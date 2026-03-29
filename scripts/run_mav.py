# run_mav.py
import asyncio
import json
import random
import os
import re
import logging
import argparse
from typing import AsyncGenerator

from typing_extensions import override
from dotenv import load_dotenv

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm

from google.genai import types
from opentelemetry import trace as otel_trace

from util import setup_tracing, setup_fault_injection, teardown_fault_injection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger("MAV")

load_dotenv("")

APP = "mav"

# ── Prompts ────────────────────────────────
GEN_PROMPT_MAIN = """
Solve the problem concisely. Include only the final answer. 
Problem: {problem}
Answer:
"""

GEN_PROMPT_HUMANEVAL = """
Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
{problem}
"""

GEN_PROMPT_MATH = """
You are a helpful assistant skilled in math problem-solving. 
Always end your solution with the final numerical answer in latex, using '\\boxed{{<answer>}}'. 
If there is no solution, reply with an empty boxed '\\boxed{{}}'.
Please solve the following math problem step by step:
QUESTION: {problem}
Provide your detailed solution below:
"""

MULTICHOICE_TEMPLATE = """
Answer the following multiple choice question. Think step by step before answering, and then output the answer in the format of "The answer is (X)" at the end, where X is the LETTER of the correct answer.
QUESTION:
{problem}
Think step by step, then end with EXACTLY "The answer is (X)", where X is the LETTER of the correct answer. Do not include the answer text itself, only the letter.
"""

SYSTEM_STR_MAIN = """
You are a critical verifier tasked with evaluating question-answering. 
You will be presented with a question and a proposed answer. 
Your job is to carefully go over and analyze the answer. Follow the instructions.
"""

SYSTEM_STR_CODE = """
You are a critical verifier tasked with evaluating code implementations. 
You will be presented with a prompt and a code implementation. 
Your job is to carefully go over and analyze the code. Follow the instructions.
"""

SYSTEM_STR_MATH = """
You are a critical verifier tasked with evaluating mathematical problem-solving. 
You will be presented with a question and a proposed solution. 
Your job is to carefully go over and analyze the solution. Follow the instructions.
"""

SYSTEM_STR_MULTIPLE_CHOICE = """
You are a critical verifier tasked with evaluating multiple-choice question-answering. 
You will be presented with a question, the multiple-choice options, and a proposed solution. 
Your job is to carefully go over and analyze the solution. Follow the instructions.
"""

VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER:"

VERA_ASK_FOR_APPROVAL_ONLY_PROMPT = (
    f"To clarify, based on the above analysis, reply with ONLY "
    f"'{VERA_ANSWER_SYMBOL}True' or ONLY '{VERA_ANSWER_SYMBOL}False'. "
    f"Do not include any other text in your response."
)

VERA_NAMES_TO_PROMPTS = {
    "math_steps": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Go over each step in the proposed solution and check whether it is mathematically correct. Think out load. "
        f"If you reach a step that is incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'."
        f"If you get to the end of all the steps and each step was correct, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "logic_steps": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Go over each step in the proposed solution and check whether it is logically sound. Think out load. "
        f"If you reach a step that is not logically sound, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If you get to the end of all the steps and each step was logically sound, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "facts_steps": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Go over each step in the proposed solution and check whether the facts presented are correct. Think out load. "
        f"If you reach a step with incorrect facts, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If you get to the end of all the steps and each step had correct facts, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "units_steps": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Check if the units are handled correctly in each step of the solution. Think out loud. "
        f"If you find any issues with the units, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If all units are handled correctly, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_direct": (
        "{prefix}"
        f"INSTRUCTIONS: \n"
        f"Is this solution correct for the given question? "
        f"Respond with ONLY '{VERA_ANSWER_SYMBOL}True' or ONLY '{VERA_ANSWER_SYMBOL}False'. Do not provide any explanation or additional text."
    ),
    "general_summarize": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Summarize the solution in your own words, explore anything you think may be incorrect. Think out load. "
        f"If you find something that's incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If you've gone over the solution and everything seems correct, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_diff": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Explain the solution in a different way than it was presented. "
        "Try to find any flaws in the solution. Think out load. "
        f"If you find something that's incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If you've gone over the solution and everything seems correct, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_edge": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Check if the solution handles edge cases and boundary conditions, test extreme values or special cases. Think out loud. "
        f"If any boundary conditions or edge cases fail, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If all boundary conditions and edge cases are handled correctly, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_mistakes": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Check if the solution has any common mistakes, calculation errors, or misconceptions that typically found in this type of problem. Think out loud. "
        f"If you find any common mistakes, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If no common mistakes are found, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_domain": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Check if the solution correctly applies relevant domain-knowledge, established theories, and standard practices for this type of problem. Think out loud. "
        f"If any domain knowledge is misapplied or violated, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If all domain-specific knowledge is correctly applied, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
}


# ── Dataset-specific config ──
DATASET_CONFIG = {
    "humaneval": {
        "gen_prompt": GEN_PROMPT_HUMANEVAL,
        "system_str": SYSTEM_STR_CODE,
        "solution_label": "PROPOSED SOLUTION",
        "veras": [
            "math_steps",
            "logic_steps",
            "facts_steps",
            "units_steps",
            "general_direct",
            "general_diff",
            "general_edge",
            "general_domain",
            "general_summarize",
        ],
    },
    "humaneval+": {
        "gen_prompt": GEN_PROMPT_HUMANEVAL,
        "system_str": SYSTEM_STR_CODE,
        "solution_label": "PROPOSED SOLUTION",
        "veras": [
            "math_steps",
            "logic_steps",
            "facts_steps",
            "units_steps",
            "general_direct",
            "general_diff",
            "general_edge",
            "general_domain",
            "general_summarize",
        ],
    },
    "mbpp": {
        "gen_prompt": GEN_PROMPT_HUMANEVAL,
        "system_str": SYSTEM_STR_CODE,
        "solution_label": "PROPOSED SOLUTION",
        "veras": [
            "math_steps",
            "logic_steps",
            "facts_steps",
            "units_steps",
            "general_direct",
            "general_diff",
            "general_edge",
            "general_domain",
            "general_summarize",
        ],
    },
    "mbpp+": {
        "gen_prompt": GEN_PROMPT_HUMANEVAL,
        "system_str": SYSTEM_STR_CODE,
        "solution_label": "PROPOSED SOLUTION",
        "veras": [
            "math_steps",
            "logic_steps",
            "facts_steps",
            "units_steps",
            "general_direct",
            "general_diff",
            "general_edge",
            "general_domain",
            "general_summarize",
        ],
    },
    "livecodebench": {
        "gen_prompt": GEN_PROMPT_HUMANEVAL,
        "system_str": SYSTEM_STR_CODE,
        "solution_label": "PROPOSED SOLUTION",
        "veras": [
            "math_steps",
            "logic_steps",
            "facts_steps",
            "units_steps",
            "general_direct",
            "general_diff",
            "general_edge",
            "general_domain",
            "general_summarize",
        ],
    },
    "math": {
        "gen_prompt": GEN_PROMPT_MATH,
        "system_str": SYSTEM_STR_MATH,
        "solution_label": "PROPOSED SOLUTION",
        "veras": [
            "units_steps",
            "general_summarize",
            "general_edge",
            "general_mistakes",
            "general_domain",
            "general_edge",
        ],
    },
    "mmlu": {
        "gen_prompt": MULTICHOICE_TEMPLATE,
        "system_str": SYSTEM_STR_MULTIPLE_CHOICE,
        "solution_label": "PROPOSED SOLUTION",
        "veras": [
            "math_steps",
            "logic_steps",
            "general_diff",
            "general_edge",
            "general_mistakes",
            "general_domain",
            "units_steps",
        ],
    },
    "gpqa": {
        "gen_prompt": MULTICHOICE_TEMPLATE,
        "system_str": SYSTEM_STR_MULTIPLE_CHOICE,
        "solution_label": "PROPOSED SOLUTION",
        "veras": ["math_steps", "logic_steps", "units_steps", "general_diff", "general_mistakes"],
    },
}

DEFAULT_CONFIG = {
    "gen_prompt": GEN_PROMPT_MAIN,
    "system_str": SYSTEM_STR_MAIN,
    "solution_label": "PROPOSED ANSWER",
    "veras": [
        "units_steps",
        "general_summarize",
        "general_edge",
        "general_mistakes",
        "general_domain",
        "math_steps",
        "logic_steps",
        "general_diff",
    ],
}


# ── Answer extraction helpers ──────────────────
def extract_verifier_approval(response: str) -> bool:
    symbol = VERA_ANSWER_SYMBOL.lower()
    idx = response.lower().rfind(symbol)
    answer = response[idx + len(symbol) :].strip() if idx != -1 else None
    if not answer:
        logger.warning(f"[extract_approval] no answer symbol found, len={len(response)}, preview={response[:200]!r}")
        return False
    answer = answer.replace("*", "").strip().lower()
    if answer in ("true", "true."):
        return True
    if answer in ("false", "false."):
        return False
    first_word = answer.split()[0] if answer.split() else ""
    if "true" in first_word:
        return True
    if "false" in first_word:
        return False
    logger.warning(f"[extract_approval] ambiguous answer={answer!r}, defaulting False")
    return False


def extract_answer_mcq(text: str) -> str:
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1)
    match = re.search(r"[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else ""


def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return ""
    i, braces, right = idx, 0, None
    while i < len(text):
        if text[i] == "{":
            braces += 1
        elif text[i] == "}":
            braces -= 1
            if braces == 0:
                right = i
                break
        i += 1
    if right is None:
        return ""
    boxed = text[idx : right + 1]
    left = "\\boxed{"
    if boxed.startswith(left) and boxed.endswith("}"):
        return boxed[len(left) : -1].replace("**", "")
    return ""


def find_code(completion: str) -> str:
    matches = re.findall(r"```python\n(.*?)```", completion, re.DOTALL)
    extracted = matches[0] if matches else completion
    sig_idx = extracted.find(":\n    ")
    if sig_idx != -1:
        extracted = extracted[sig_idx + 2 :]
    return extracted.strip()


def extract_answer_for_dataset(solution: str, dataset: str) -> str:
    ds = dataset.lower()
    if ds in ("mmlu", "gpqa"):
        return extract_answer_mcq(solution.replace("**", "")) or ""
    if ds == "math":
        return extract_boxed(solution) or ""
    if ds in ("humaneval", "humaneval+", "mbpp", "mbpp+", "livecodebench"):
        return find_code(solution)
    return solution.replace("**", "")


def compute_score(approvals: dict, veras: list) -> float:
    if not veras:
        return 0.0
    total = sum(approvals.get(v, False) for v in veras)
    return total / len(veras)


# ── ADK Tools for code verification ──────────────────

def execute_code_for_verification(code: str, test_input: str = "") -> dict:
    """Execute Python code to verify its correctness.
    
    Use this tool to actually run the proposed solution and check if it works.
    You can provide test input to see the output.
    
    Args:
        code: Python code to execute.
        test_input: Optional stdin input for the code.
    
    Returns:
        Dict with 'stdout', 'stderr', 'success', and 'execution_time'.
    """
    import subprocess
    import sys
    import tempfile
    import time
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        
        start = time.time()
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=test_input if test_input else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        elapsed = time.time() - start
        
        return {
            "stdout": proc.stdout.strip()[:1000],
            "stderr": proc.stderr.strip()[:500] if proc.stderr else "",
            "success": proc.returncode == 0,
            "execution_time": round(elapsed, 3),
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Timeout (10s)", "success": False, "execution_time": 10.0}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False, "execution_time": 0.0}
    finally:
        if tmp_path:
            try:
                import os as _os
                _os.unlink(tmp_path)
            except Exception:
                pass


def check_syntax(code: str) -> dict:
    """Check if Python code has valid syntax without executing it.
    
    Args:
        code: Python code to check.
    
    Returns:
        Dict with 'valid' (bool) and 'error' (str if invalid).
    """
    import ast
    try:
        ast.parse(code)
        return {"valid": True, "error": ""}
    except SyntaxError as e:
        return {"valid": False, "error": f"Line {e.lineno}: {e.msg}"}


# ── Model factory ──────────────────────────────
def make_model(model_name: str, base_url: str, api_key: str):
    extra = {}
    n = (model_name or "").lower()
    if any(k in n for k in ("qwen", "seed", "glm")):
        extra = {"temperature": 0.7, "extra_body": {"thinking": {"type": "disabled"}}}
    elif "claude" in n:
        extra = {"temperature": 0.7}
    return LiteLlm(model=f"openai/{model_name}", api_base=base_url, api_key=api_key, drop_params=True, **extra)


# ── MAV Orchestrator ──────────────────
class MAVOrchestrator(BaseAgent):
    """MAV: Generate n solutions, verify each with selected verifiers, pick best."""

    solver: LlmAgent
    verifier: LlmAgent
    ss: InMemorySessionService
    max_llm_calls: int = 100
    n_solutions: int = 2
    n_max_verifiers: int = 2
    dataset: str = "default"

    model_config = {"arbitrary_types_allowed": True}

    async def _call(self, agent: LlmAgent, prompt: str, history: list, ctr: list) -> tuple:
        """Call an agent in a NEW session. Supports optional history for multi-turn.
        Returns (response_text: str, exceeded: bool).
        """
        # Serialize history into prompt if present
        if history:
            lines = [f"[{m['role']}]: {m['text']}" for m in history]
            full_prompt = (
                "<conversation_history>\n"
                + "\n".join(lines)
                + "\n</conversation_history>\n\n<current_message>\n"
                + prompt
                + "\n</current_message>"
            )
        else:
            full_prompt = prompt

        session = await self.ss.create_session(app_name=APP, user_id="u")
        runner = Runner(agent=agent, app_name=APP, session_service=self.ss)

        resp, exceeded, counted = "", False, False

        async for ev in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=full_prompt)]),
        ):
            if not (ev.content and ev.content.parts and ev.author == agent.name):
                continue

            fc = ev.get_function_calls()
            if fc and ctr is not None:
                ctr[0] += 1
                counted = True
                logger.info(f"[{agent.name}] LLM #{ctr[0]}/{self.max_llm_calls} fc={[c.name for c in fc]}")
                if ctr[0] >= self.max_llm_calls:
                    exceeded = True

            t = (ev.content.parts[0].text or "").strip()
            if t:
                resp = t

        # Count non-tool LLM call
        if ctr is not None and not exceeded and not counted:
            ctr[0] += 1
            logger.info(f"[{agent.name}] LLM #{ctr[0]}/{self.max_llm_calls} (no tools)")
            if ctr[0] >= self.max_llm_calls:
                exceeded = True

        return resp, exceeded

    async def _call_in_span(self, agent, prompt, history, ctr, span_name):
        """Wrap _call inside OTel span for trace nesting."""
        tracer = otel_trace.get_tracer("mav")
        with tracer.start_as_current_span(
            span_name,
            attributes={"orchestrator.agent": agent.name, "orchestrator.llm_calls_before": ctr[0] if ctr else 0},
        ) as span:
            resp, exceeded = await self._call(agent, prompt, history, ctr)
            span.set_attribute("orchestrator.llm_calls_after", ctr[0] if ctr else 0)
            span.set_attribute("orchestrator.exceeded", exceeded)
            return resp, exceeded

    def _evt(self, ctx: InvocationContext, author: str, text: str) -> Event:
        return Event(
            invocation_id=ctx.invocation_id,
            author=author,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Extract user query
        query = ""
        for ev in reversed(ctx.session.events):
            if ev.content and ev.content.role == "user" and ev.content.parts:
                query = ev.content.parts[0].text or ""
                if query:
                    break
        if not query:
            query = ctx.session.state.get("user_query", "")
        if not query:
            yield self._evt(ctx, "orchestrator", "No query provided.")
            return

        # Get dataset-specific config
        ds_cfg = DATASET_CONFIG.get(self.dataset.lower(), DEFAULT_CONFIG)
        gen_prompt_tpl = ds_cfg["gen_prompt"]
        system_str = ds_cfg["system_str"]
        solution_label = ds_cfg["solution_label"]

        # Shuffle and select verifiers
        all_veras = list(ds_cfg["veras"])
        random.shuffle(all_veras)
        selected_veras = all_veras[: self.n_max_verifiers]

        ctr = [0]
        budget = lambda: ctr[0] < self.max_llm_calls

        logger.info(
            f"[MAV] start | dataset={self.dataset}, n_solutions={self.n_solutions}, "
            f"n_max_verifiers={self.n_max_verifiers}, verifiers={selected_veras}, query={query[:80]!r}"
        )

        # ── Step 1: Generate n candidate solutions ──
        solutions = []
        extracted_answers = []
        gen_prompt = gen_prompt_tpl.format(problem=query)

        for i in range(self.n_solutions):
            if not budget():
                logger.warning(f"[MAV] budget exceeded during solution generation at sol={i}")
                break

            solution, exceeded = await self._call_in_span(self.solver, gen_prompt, [], ctr, f"solver_call_{i}")
            answer = extract_answer_for_dataset(solution, self.dataset)
            solutions.append(solution)
            extracted_answers.append(answer or "")

            yield self._evt(ctx, "solver", f"Solution {i}: {solution[:300]}")
            logger.info(f"[MAV] solution {i}: answer_len={len(answer)}, preview={answer[:80]!r}")

            if exceeded:
                break

        if not solutions:
            logger.error("[MAV] no solutions generated")
            yield self._evt(ctx, "assistant", "")
            return

        # ── Step 2: Verify each solution with each verifier ──
        all_approvals = [{} for _ in range(len(solutions))]

        for sol_idx, solution in enumerate(solutions):
            prefix = f"{system_str}\n\nQUESTION:\n{query}\n\n{solution_label}:\n{solution}\n\n"

            for vera_name in selected_veras:
                if not budget():
                    logger.warning(f"[MAV] budget exceeded during verification at sol={sol_idx}/{vera_name}")
                    break

                user_prompt = VERA_NAMES_TO_PROMPTS[vera_name].format(prefix=prefix)

                vera_response, exceeded = await self._call_in_span(
                    self.verifier,
                    user_prompt,
                    [],
                    ctr,
                    f"verifier_analysis_{sol_idx}_{vera_name}",
                )
                yield self._evt(ctx, "verifier", f"sol{sol_idx}/{vera_name}: {vera_response[:150]}")

                if exceeded or not budget():
                    break

                approval_hist = [
                    {"role": "user", "text": user_prompt},
                    {"role": "assistant", "text": vera_response},
                ]
                approval_resp, exceeded = await self._call_in_span(
                    self.verifier,
                    VERA_ASK_FOR_APPROVAL_ONLY_PROMPT,
                    approval_hist,
                    ctr,
                    f"verifier_approval_{sol_idx}_{vera_name}",
                )

                approval_bool = extract_verifier_approval(approval_resp)
                all_approvals[sol_idx][vera_name] = approval_bool
                logger.info(f"[MAV] sol{sol_idx}/{vera_name}: approval={approval_bool}")
                yield self._evt(ctx, "verifier", f"sol{sol_idx}/{vera_name}: {approval_bool}")

                if exceeded:
                    break

        # ── Step 3: Select best solution by approval score ──
        scores = [compute_score(all_approvals[i], selected_veras) for i in range(len(solutions))]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_answer = extracted_answers[best_idx]

        logger.info(
            f"[MAV] scores={scores}, best_idx={best_idx}, answer_len={len(best_answer)}, preview={best_answer[:120]!r}"
        )

        ctx.session.state["final_response"] = best_answer
        yield self._evt(ctx, "assistant", best_answer)


# ── Build orchestrator ──────────────────────────────────────
def build_orchestrator(
    ss: InMemorySessionService,
    model_name: str,
    base_url: str,
    api_key: str,
    n_solutions: int = 2,
    n_max_verifiers: int = 2,
    max_llm_calls: int = 100,
    dataset: str = "default",
) -> MAVOrchestrator:
    mk = lambda: make_model(model_name, base_url, api_key)

    solver = LlmAgent(
        name="solver",
        model=mk(),
        instruction="You are a helpful problem-solving assistant.",
        output_key="response",
    )
    verifier = LlmAgent(
        name="verifier",
        model=mk(),
        instruction=(
            "You are a critical verifier tasked with evaluating solutions. "
            "For code solutions, use execute_code_for_verification to actually run the code and verify it works. "
            "Use check_syntax to verify the code is syntactically correct."
        ),
        output_key="response",
        tools=[execute_code_for_verification, check_syntax],
    )

    return MAVOrchestrator(
        name="orchestrator",
        solver=solver,
        verifier=verifier,
        ss=ss,
        max_llm_calls=max_llm_calls,
        n_solutions=n_solutions,
        n_max_verifiers=n_max_verifiers,
        dataset=dataset,
        sub_agents=[solver, verifier],
    )


# ── Run single sample ──────────────────────────────────────
async def run_agent(runner: Runner, ss: InMemorySessionService, sample: dict, **kwargs) -> str:
    query = sample["query"] if isinstance(sample, dict) else sample
    session = await ss.create_session(app_name=APP, user_id="user", state={"user_query": query})
    msg = types.Content(role="user", parts=[types.Part(text=query)])

    last_text = ""
    async for ev in runner.run_async(user_id="user", session_id=session.id, new_message=msg):
        if ev.content and ev.content.parts and ev.author == "assistant":
            text = (ev.content.parts[0].text or "").strip()
            if text:
                last_text = text

    if not last_text:
        final_session = await ss.get_session(app_name=APP, user_id="user", session_id=session.id)
        if final_session and final_session.state:
            last_text = final_session.state.get("final_response", "")

    if not last_text:
        logger.warning(f"[run_agent] empty response, query={query[:80]!r}")
    else:
        logger.info(f"[run_agent] response len={len(last_text)}, preview={last_text[:120]!r}")
    return last_text


async def main():
    p = argparse.ArgumentParser(description="MAV Agent")
    p.add_argument("--task_id", default="default")
    p.add_argument("--query", default="")
    p.add_argument("--max_llm_calls", type=int, default=100)
    p.add_argument("--n_solutions", type=int, default=2, help="Number of candidate solutions")
    p.add_argument("--n_max_verifiers", type=int, default=2, help="Max verifiers to use")
    p.add_argument("--dataset", default="default", help="Dataset: humaneval, mbpp, math, mmlu, gpqa, etc.")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--model", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--sample_file", default=None, help="JSON file with sample data")
    p.add_argument("--fault_name", default=None, help="Fault name (e.g. 'llm_error_single'). None=no injection.")
    args = p.parse_args()

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "sk-xxx")

    if args.sample_file and os.path.isfile(args.sample_file):
        with open(args.sample_file, "r", encoding="utf-8") as f:
            sample = json.load(f)
        if "query" not in sample:
            logger.error(f"[main] sample_file missing 'query' field: {args.sample_file}")
            return
    elif args.query:
        sample = {"query": args.query}
    else:
        logger.error("[main] must provide --query or --sample_file")
        return

    task_dir = os.path.join(args.output_dir, args.task_id)
    trace_dir = os.path.join(task_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)

    with open(os.path.join(task_dir, "input.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": args.task_id,
                "query": sample["query"],
                "model": model,
                "max_llm_calls": args.max_llm_calls,
                "dataset": args.dataset,
                "n_solutions": args.n_solutions,
                "n_max_verifiers": args.n_max_verifiers,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fh = logging.FileHandler(os.path.join(task_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    tp, _ = setup_tracing("MAV", logger=logger, trace_dir=trace_dir)

    fault_engine = setup_fault_injection(args.fault_name)

    ss = InMemorySessionService()
    orchestrator = build_orchestrator(
        ss=ss,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        n_solutions=args.n_solutions,
        n_max_verifiers=args.n_max_verifiers,
        max_llm_calls=args.max_llm_calls,
        dataset=args.dataset,
    )
    runner = Runner(agent=orchestrator, app_name=APP, session_service=ss)

    logger.info(
        f"[main] task={args.task_id}, model={model}, dataset={args.dataset}, "
        f"n_solutions={args.n_solutions}, n_max_verifiers={args.n_max_verifiers}"
    )
    logger.info(f"[main] query={sample['query'][:120]!r}")

    resp = await run_agent(runner, ss, sample)

    teardown_fault_injection(fault_engine)

    output_data = {"task_id": args.task_id, "final_answer": resp, "answer_length": len(resp)}
    if fault_engine is not None:
        output_data["_fault_name"] = args.fault_name
        output_data["_fault_fired"] = len(fault_engine.log)
        output_data["_fault_log"] = fault_engine.log

    with open(os.path.join(task_dir, "output.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

    if resp:
        logger.info(f"[main] OK, task={args.task_id}, answer_len={len(resp)}, preview={resp[:200]!r}")
    else:
        logger.warning(f"[main] EMPTY response for task={args.task_id}")

    if tp:
        tp.force_flush()


if __name__ == "__main__":
    asyncio.run(main())
