# run_mad.py
import asyncio
import json
import re
import os
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
logger = logging.getLogger("MAD")

load_dotenv("")

APP = "mad"


# ── Prompts ────────────────────────────────
# NOTE: Original PLAYER_META_PROMPT / MODERATOR_META_PROMPT contain {task}.
# ADK instruction is STATIC (set once at agent creation), so we use generic
# instructions here. The task context is injected via the user prompt in _call.
PLAYER_INSTRUCTION = "You are a debater. Hello and welcome to the debate. It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer."

MODERATOR_INSTRUCTION = "You are a moderator. There will be two debaters involved in a debate. They will present their answers and discuss their perspectives. At the end of each round, you will evaluate answers and decide which is correct."

NEGATIVE_PROMPT = "##aff_ans##\n\nYou disagree with my answer. Provide your answer and reasons."

MODERATOR_PROMPT = 'Now the ##round## round of debate for both sides has ended.\n\nAffirmative side arguing:\n##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nYou, as the moderator, will evaluate both sides\' answers and determine if there is a clear preference for an answer candidate. If so, please summarize your reasons for supporting affirmative/negative side and give the final answer that you think is correct, and the debate will conclude. If not, the debate will continue to the next round. Now please output your answer in json format, with the format as follows: {"Whether there is a preference": "Yes or No", "Supported Side": "Affirmative or Negative", "Reason": "", "debate_answer": ""}. IMPORTANT: If the topic is a coding task, the "debate_answer" field must contain the complete executable code (function definition), not a description. Please strictly output in JSON format, do not output irrelevant content.'

DEBATE_PROMPT = "##oppo_ans##\n\nDo you agree with my perspective? Please provide your reasons and answer."

JUDGE_PROMPT_LAST1 = "Affirmative side arguing: ##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nNow, what answer candidates do we have? Present them without reasons."

JUDGE_PROMPT_LAST2 = 'Therefore, ##debate_topic##\nPlease summarize your reasons and give the final answer that you think is correct. Now please output your answer in json format, with the format as follows: {"Reason": "", "debate_answer": ""}. IMPORTANT: If the topic is a coding task, the "debate_answer" field must contain the complete executable code (function definition), not a description. Please strictly output in JSON format, do not output irrelevant content.'

ROUND_NAMES = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}


# ── Helpers ─────────────────────────────────────────────────
def parse_json_response(raw: str) -> dict:
    """Parse JSON from moderator/judge LLM response."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return eval(cleaned)
    except Exception as e:
        logger.error(f"[parse_json] failed: {e}, raw={cleaned[:300]}")
        return {"debate_answer": ""}


# ── ADK Tools for debate with evidence ──────────────────────────────

def run_code_as_evidence(code: str, description: str = "") -> dict:
    """Execute Python code to provide evidence for your argument.
    
    Use this to demonstrate that your proposed solution actually works,
    or to show that the opponent's solution has flaws.
    
    Args:
        code: Python code to execute as evidence.
        description: What this code is meant to demonstrate.
    
    Returns:
        Dict with execution results that can support your argument.
    """
    import subprocess
    import sys
    import tempfile
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        return {
            "description": description,
            "stdout": proc.stdout.strip()[:1000],
            "stderr": proc.stderr.strip()[:500] if proc.stderr else "",
            "success": proc.returncode == 0,
            "evidence_supports_claim": proc.returncode == 0 and not proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"description": description, "stdout": "", "stderr": "Timeout", "success": False, "evidence_supports_claim": False}
    except Exception as e:
        return {"description": description, "stdout": "", "stderr": str(e), "success": False, "evidence_supports_claim": False}
    finally:
        if tmp_path:
            try:
                import os as _os
                _os.unlink(tmp_path)
            except Exception:
                pass


def verify_code_correctness(code: str, test_cases: str) -> dict:
    """Verify if code passes given test cases.
    
    Use this to objectively check if a proposed solution is correct.
    
    Args:
        code: The code solution to verify.
        test_cases: Test assertions to run (e.g., "assert func(1) == 2").
    
    Returns:
        Dict with verification results.
    """
    import subprocess
    import sys
    import tempfile
    
    full_code = code + "\n" + test_cases
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(full_code)
            tmp_path = f.name
        
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        return {
            "all_tests_passed": proc.returncode == 0,
            "error": proc.stderr.strip()[:500] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"all_tests_passed": False, "error": "Timeout (10s)"}
    except Exception as e:
        return {"all_tests_passed": False, "error": str(e)}
    finally:
        if tmp_path:
            try:
                import os as _os
                _os.unlink(tmp_path)
            except Exception:
                pass


def make_model(model_name: str, base_url: str, api_key: str):
    extra = {}
    n = (model_name or "").lower()
    if any(k in n for k in ("qwen", "seed", "glm")):
        extra = {"temperature": 0.7, "extra_body": {"thinking": {"type": "disabled"}}}
    elif "claude" in n:
        extra = {"temperature": 0.7}
    return LiteLlm(model=f"openai/{model_name}", api_base=base_url, api_key=api_key, drop_params=True, **extra)


# ── MAD Orchestrator ───────────────────────
class MADOrchestrator(BaseAgent):
    """Multi-Agent Debate: affirmative vs negative, moderator evaluates, judge fallback."""

    affirmative: LlmAgent
    negative: LlmAgent
    moderator: LlmAgent
    judge: LlmAgent
    ss: InMemorySessionService
    max_llm_calls: int = 100
    max_round: int = 3

    model_config = {"arbitrary_types_allowed": True}

    async def _call(self, agent: LlmAgent, prompt: str, history: list, ctr: list) -> tuple:
        """Call an agent in a NEW session with history context.
        Returns (response_text: str, exceeded: bool).
        """
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

        if ctr is not None and not exceeded and not counted:
            ctr[0] += 1
            logger.info(f"[{agent.name}] LLM #{ctr[0]}/{self.max_llm_calls} (no tools)")
            if ctr[0] >= self.max_llm_calls:
                exceeded = True

        return resp, exceeded

    async def _call_in_span(self, agent, prompt, history, ctr, span_name):
        """Wrap _call inside OTel span for trace nesting."""
        tracer = otel_trace.get_tracer("mad")
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
        # Extract debate topic
        task = ""
        for ev in reversed(ctx.session.events):
            if ev.content and ev.content.role == "user" and ev.content.parts:
                task = ev.content.parts[0].text or ""
                if task:
                    break
        if not task:
            task = ctx.session.state.get("user_query", "")
        if not task:
            yield self._evt(ctx, "orchestrator", "No task provided.")
            return

        # Independent history per agent
        aff_hist, neg_hist, mod_hist = [], [], []
        ctr = [0]
        budget = lambda: ctr[0] < self.max_llm_calls

        logger.info(f"[MAD] start | topic={task[:80]!r}, max_round={self.max_round}")

        # ── Round 1: Affirmative starts ──
        aff_ans, exceeded = await self._call_in_span(self.affirmative, task, aff_hist, ctr, "affirmative_call_0")
        aff_hist += [{"role": "user", "text": task}, {"role": "assistant", "text": aff_ans}]
        yield self._evt(ctx, "affirmative", aff_ans)
        logger.info(f"[MAD] round 1 aff len={len(aff_ans)}")
        base_answer = aff_ans
        if exceeded:
            ctx.session.state["final_response"] = base_answer
            yield self._evt(ctx, "assistant", base_answer)
            return

        # ── Round 1: Negative responds ──
        neg_prompt = NEGATIVE_PROMPT.replace("##aff_ans##", aff_ans)
        neg_ans, exceeded = await self._call_in_span(self.negative, neg_prompt, neg_hist, ctr, "negative_call_0")
        neg_hist += [{"role": "user", "text": neg_prompt}, {"role": "assistant", "text": neg_ans}]
        yield self._evt(ctx, "negative", neg_ans)
        logger.info(f"[MAD] round 1 neg len={len(neg_ans)}")
        if exceeded:
            ctx.session.state["final_response"] = base_answer
            yield self._evt(ctx, "assistant", base_answer)
            return

        # ── Round 1: Moderator evaluates ──
        mod_prompt = (
            MODERATOR_PROMPT.replace("##aff_ans##", aff_ans)
            .replace("##neg_ans##", neg_ans)
            .replace("##round##", "first")
        )
        mod_raw, exceeded = await self._call_in_span(self.moderator, mod_prompt, mod_hist, ctr, "moderator_call_0")
        mod_hist += [{"role": "user", "text": mod_prompt}, {"role": "assistant", "text": mod_raw}]
        yield self._evt(ctx, "moderator", mod_raw)
        mod_ans = parse_json_response(mod_raw)
        logger.info(f"[MAD] round 1 moderator verdict={mod_ans}")

        # ── Subsequent rounds ──
        for rnd in range(self.max_round - 1):
            if mod_ans.get("debate_answer") or not budget():
                break
            round_name = ROUND_NAMES.get(rnd + 2, f"{rnd + 2}th")

            # Affirmative debates
            aff_prompt = DEBATE_PROMPT.replace("##oppo_ans##", neg_ans)
            aff_ans, exceeded = await self._call_in_span(
                self.affirmative, aff_prompt, aff_hist, ctr, f"affirmative_call_{rnd+1}"
            )
            aff_hist += [{"role": "user", "text": aff_prompt}, {"role": "assistant", "text": aff_ans}]
            yield self._evt(ctx, "affirmative", aff_ans)
            if exceeded:
                break

            # Negative debates
            neg_prompt = DEBATE_PROMPT.replace("##oppo_ans##", aff_ans)
            neg_ans, exceeded = await self._call_in_span(
                self.negative, neg_prompt, neg_hist, ctr, f"negative_call_{rnd+1}"
            )
            neg_hist += [{"role": "user", "text": neg_prompt}, {"role": "assistant", "text": neg_ans}]
            yield self._evt(ctx, "negative", neg_ans)
            if exceeded:
                break

            # Moderator evaluates
            mod_prompt = (
                MODERATOR_PROMPT.replace("##aff_ans##", aff_ans)
                .replace("##neg_ans##", neg_ans)
                .replace("##round##", round_name)
            )
            mod_raw, exceeded = await self._call_in_span(
                self.moderator, mod_prompt, mod_hist, ctr, f"moderator_call_{rnd+1}"
            )
            mod_hist += [{"role": "user", "text": mod_prompt}, {"role": "assistant", "text": mod_raw}]
            yield self._evt(ctx, "moderator", mod_raw)
            mod_ans = parse_json_response(mod_raw)
            logger.info(f"[MAD] round {rnd+2} moderator verdict={mod_ans}")
            if exceeded:
                break

        # ── Result or Judge fallback ──
        if mod_ans.get("debate_answer"):
            debate_answer = mod_ans["debate_answer"]
            logger.info(f"[MAD] resolved by moderator: {debate_answer[:120]!r}")
        else:
            logger.info("[MAD] no consensus, invoking judge...")
            first_aff = aff_hist[1]["text"]
            first_neg = neg_hist[1]["text"]
            judge_hist = []

            j1_prompt = JUDGE_PROMPT_LAST1.replace("##aff_ans##", first_aff).replace("##neg_ans##", first_neg)
            j1_ans, _ = await self._call_in_span(self.judge, j1_prompt, judge_hist, ctr, "judge_call_0")
            judge_hist += [{"role": "user", "text": j1_prompt}, {"role": "assistant", "text": j1_ans}]
            yield self._evt(ctx, "judge", j1_ans)

            j2_prompt = JUDGE_PROMPT_LAST2.replace("##debate_topic##", task)
            j2_raw, _ = await self._call_in_span(self.judge, j2_prompt, judge_hist, ctr, "judge_call_1")
            yield self._evt(ctx, "judge", j2_raw)
            j_ans = parse_json_response(j2_raw)
            debate_answer = j_ans.get("debate_answer", base_answer)
            logger.info(f"[MAD] judge verdict={j_ans}")

        ctx.session.state["final_response"] = debate_answer
        logger.info(f"[MAD] done | answer_len={len(debate_answer)}, preview={debate_answer[:120]!r}")
        yield self._evt(ctx, "assistant", debate_answer)


# ── Build orchestrator ──────────────────────────────────────
def build_orchestrator(
    ss: InMemorySessionService,
    model_name: str,
    base_url: str,
    api_key: str,
    max_round: int = 3,
    max_llm_calls: int = 100,
) -> MADOrchestrator:
    mk = lambda: make_model(model_name, base_url, api_key)
    # Static instructions — task is injected via user prompt
    # Debaters can use code execution to support their arguments
    affirmative = LlmAgent(
        name="affirmative",
        model=mk(),
        instruction=(
            PLAYER_INSTRUCTION + 
            " For coding tasks, you can use run_code_as_evidence to demonstrate your solution works."
        ),
        output_key="response",
        tools=[run_code_as_evidence],
    )
    negative = LlmAgent(
        name="negative",
        model=mk(),
        instruction=(
            PLAYER_INSTRUCTION + 
            " For coding tasks, you can use run_code_as_evidence to demonstrate flaws in opponent's solution or show your alternative works."
        ),
        output_key="response",
        tools=[run_code_as_evidence],
    )
    # Moderator and judge can verify code to make objective decisions
    moderator = LlmAgent(
        name="moderator",
        model=mk(),
        instruction=(
            MODERATOR_INSTRUCTION + 
            " For coding tasks, use verify_code_correctness to objectively test solutions before making a verdict."
        ),
        output_key="response",
        tools=[verify_code_correctness],
    )
    judge = LlmAgent(
        name="judge",
        model=mk(),
        instruction=(
            MODERATOR_INSTRUCTION + 
            " For coding tasks, use verify_code_correctness to objectively test solutions before making the final decision."
        ),
        output_key="response",
        tools=[verify_code_correctness],
    )

    return MADOrchestrator(
        name="orchestrator",
        affirmative=affirmative,
        negative=negative,
        moderator=moderator,
        judge=judge,
        ss=ss,
        max_llm_calls=max_llm_calls,
        max_round=max_round,
        sub_agents=[affirmative, negative, moderator, judge],
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
        logger.info(f"[run_agent] len={len(last_text)}, preview={last_text[:120]!r}")
    return last_text


# ── Main entry ──────────────
async def main():
    p = argparse.ArgumentParser(description="MAD Agent")
    p.add_argument("--task_id", default="default")
    p.add_argument("--query", default="")
    p.add_argument("--max_llm_calls", type=int, default=100)
    p.add_argument("--max_round", type=int, default=3, help="Max debate rounds")
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
            logger.error(f"[main] sample_file missing 'query': {args.sample_file}")
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
                "max_round": args.max_round,
                "max_llm_calls": args.max_llm_calls,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fh = logging.FileHandler(os.path.join(task_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    tp, _ = setup_tracing("MAD", logger=logger, trace_dir=trace_dir)
    fault_engine = setup_fault_injection(args.fault_name)

    ss = InMemorySessionService()
    orchestrator = build_orchestrator(ss, model, base_url, api_key, args.max_round, args.max_llm_calls)
    runner = Runner(agent=orchestrator, app_name=APP, session_service=ss)

    logger.info(f"[main] task={args.task_id}, model={model}, max_round={args.max_round}")
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
