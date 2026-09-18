# run_mapcoder.py
import asyncio
import json
import os
import re
import xml.etree.ElementTree as ET
import logging
from typing import AsyncGenerator

from typing_extensions import override
from pydantic import BaseModel

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.tools import ToolContext
from opentelemetry import trace as otel_trace
from util import setup_tracing, setup_fault_injection, teardown_fault_injection
import argparse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger("MapCoder")

APP = "mapcoder"

# ── Number mapping for prompt ──
MAPPING = {
    1: "one (01)",
    2: "two (02)",
    3: "three (03)",
    4: "four (04)",
    5: "five (05)",
    6: "six (06)",
    7: "seven (07)",
    8: "eight (08)",
    9: "nine (09)",
}

# ── Prompts ──
INPUT_KB_EXEMPLARS = """Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.

# Problem:
{query}

# Exemplars:
Recall {k_str} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate {language} code step by step to solve that problem
3. finally generate a planning to solve that problem

# Algorithm:

----------------
Important:
Your response must follow the following xml format-

<root>
<problem>
# Recall {k_str} relevant and distinct problems (different from problem mentioned above). Write each problem in the following format.
<description>
# Describe the problem.
</description>
<code>
# Let's think step by step to solve this problem in {language} programming language.
</code>
<planning>
# Planning to solve this problem.
</planning>
</problem>
# similarly add more problems here...

<algorithm>
# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.
# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.
</algorithm>
</root>
"""

PLANNING_PROMPT = """Given a competitive programming problem generate a concrete planning to solve the problem.

# Problem:
{example_problem}

# Planning:
{example_planning}

{algorithm_prompt}

## Problem to be solved:
{prompt}

{sample_io_prompt}

## Planning:
----------------
Important: You should give only the planning to solve the problem. Do not add extra explanation or words."""

PLANNING_FOR_VERIFICATION = """Given a competitive programming problem and a plan to solve the problem in {language}, tell whether the plan is correct to solve this problem.

# Problem:
{query}

# Planning:
{planning}

----------------
Important: Your response must follow the following xml format-```
<root>
<explanation> Discuss whether the given competitive programming problem is solvable by using the above mentioned planning.</explanation>
<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>
</root>"""

FINAL_CODE_GENERATION = """Given a competitive programming problem generate {language} code to solve the problem.

{algorithm_prompt}

## Problem to be solved:
{prompt}

## Planning:
{planning}

{sample_io_prompt}

## Let's think step by step.
----------------
Important:
{std_input_prompt}
## Your response must contain only the {language} code to solve this problem. Do not add extra explanation or words."""

IMPROVING_CODE = """Given a competitive programming problem you have generated {language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.

{algorithm_prompt}

## Problem to be solved:
{prompt}

{response}

## Test Report:
{test_log}

## Modified Planning:

## Let's think step by step to modify {language} Code for solving this problem.
----------------
Important:
{std_input_prompt}
## Your response must contain the modified planning and then the {language} code inside ``` block to solve this problem."""


# ── Utility functions ──
def run_subprocess(code: str, stdin_input: str = "") -> dict:
    """Run Python code in subprocess, return dict with stdout/error."""
    import subprocess, sys, tempfile

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            f.flush()
            tmp_path = f.name
        r = subprocess.run(
            [sys.executable, tmp_path],
            input=stdin_input or None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        os.unlink(tmp_path)
        out = {"stdout": r.stdout.strip()}
        if r.stderr.strip():
            out["error"] = r.stderr.strip()
        return out
    except subprocess.TimeoutExpired:
        return {"stdout": "", "error": "Timeout (10s)"}
    except Exception as e:
        return {"stdout": "", "error": str(e)}


def evaluate_tests(test_cases: list, code: str, entry_point: str = "") -> tuple:
    """Run code against test cases, return (passed: bool, test_log: str)."""
    test_log = ""
    all_passed = True
    for tc in test_cases:
        # stdin/stdout format (LiveCodeBench)
        if isinstance(tc, dict) and "input" in tc:
            stdin_data = tc["input"]
            expected = tc.get("output", "").strip()
            result = run_subprocess(code, stdin_input=stdin_data)
            actual = result.get("stdout", "").strip()
            if result.get("error") or actual != expected:
                all_passed = False
                test_log += f"failed: expected={expected!r}, got={actual!r}\n"
            else:
                test_log += f"passed: input={stdin_data[:40]!r}\n"
            continue
        # assertion format (HumanEval, MBPP)
        full = ("from typing import *\n" if "from typing import *" not in code else "") + code + "\n" + tc + "\n"
        if "def check(candidate)" in tc and entry_point:
            full += f"\ncheck({entry_point})\n"
        result = run_subprocess(full)
        if result.get("error"):
            all_passed = False
            test_log += f"failed in test case: {tc}\n"
        else:
            test_log += f"passed in test case: {tc}\n"
    return all_passed, test_log


def get_sample_io(query: str, dataset: str, test_cases_json: str = "", input_output: str = "") -> list:
    """Extract sample test cases from various sources."""
    # 1. Structured test_cases (HumanEval, MBPP)
    if test_cases_json:
        try:
            tc = json.loads(test_cases_json)
            if tc and isinstance(tc, list):
                logger.info(f"[get_sample_io] using test_cases field, count={len(tc)}")
                return tc
        except json.JSONDecodeError as e:
            logger.error(f"[get_sample_io] test_cases_json parse error: {e}")

    # 2. stdin/stdout from input_output (LiveCodeBench)
    if input_output:
        try:
            io_list = json.loads(input_output) if isinstance(input_output, str) else input_output
            if io_list and isinstance(io_list, list):
                logger.info(f"[get_sample_io] using input_output field, count={len(io_list)}")
                return io_list
        except json.JSONDecodeError as e:
            logger.error(f"[get_sample_io] input_output parse error: {e}")

    # 3. Parse from doctest (HumanEval-style)
    pattern = r">>> (.*?)\n\s*([^\n>]*)"
    matches = re.findall(pattern, query)
    if matches:
        assertions = []
        for func_call, expected in matches:
            func_call, expected = func_call.strip(), expected.strip()
            if expected and not expected.startswith(">>>"):
                assertions.append(f"assert {func_call} == {expected}")
            elif not expected:
                assertions.append(f"assert {func_call} is None")
        if assertions:
            logger.info(f"[get_sample_io] parsed from doctest, count={len(assertions)}")
            return assertions

    # 4. MBPP-style assert lines
    for line in query.split("\n"):
        if line.strip().startswith("assert "):
            assertions = [l.strip() for l in query.split("\n") if l.strip().startswith("assert ")]
            logger.info(f"[get_sample_io] parsed assert lines, count={len(assertions)}")
            return assertions

    logger.warning(f"[get_sample_io] no test cases found for dataset={dataset}")
    return []


def get_sample_io_str(sample_io: list) -> str:
    """Format sample I/O as string for prompt."""
    if sample_io and isinstance(sample_io[0], str):
        return "\n".join(sample_io)
    if sample_io and isinstance(sample_io[0], dict):
        return "\n".join(f"Input:\n{io.get('input','')}\nExpected output:\n{io.get('output','')}" for io in sample_io)
    return str(sample_io)


def parse_code(response: str) -> str:
    """Extract code from markdown code block."""
    if "```" not in response:
        return response
    for lang in ["python3", "python", "Python3", "Python"]:
        pat = rf"```{re.escape(lang)}((.|\n)*?)```"
        blocks = re.findall(pat, response, re.DOTALL)
        if blocks:
            return "\n".join(blocks[-1]) if isinstance(blocks[-1], (tuple, list)) else blocks[-1]
    blocks = re.findall(r"```((.|\n)*?)```", response, re.DOTALL)
    if blocks:
        return "\n".join(blocks[-1]) if isinstance(blocks[-1], (tuple, list)) else blocks[-1]
    return response


def trim_text(text: str, trimmed: str) -> str:
    return text.replace(trimmed, "").strip()


def replace_tag(text: str, tag: str) -> str:
    if f"<{tag}><![CDATA[" in text and f"]]></{tag}>" in text:
        return text
    return text.replace(f"<{tag}>", f"<{tag}><![CDATA[").replace(f"</{tag}>", f"]]></{tag}>").strip()


def xml_to_dict(element) -> dict:
    result = {}
    for child in element:
        child_data = xml_to_dict(child) if len(child) else child.text
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    return result


def parse_xml(response: str) -> dict:
    response = response.replace("```xml", "").replace("```", "")
    for wrapper in [response, "<root>\n" + response + "\n</root>", "<root>\n" + response]:
        try:
            return xml_to_dict(ET.fromstring(wrapper))
        except ET.ParseError:
            continue
    logger.error(f"[parse_xml] all parse attempts failed, response={response[:200]!r}")
    return {}


# ── Model factory (consistent with standard) ──
def make_model(model_name: str, base_url: str, api_key: str):
    extra = {}
    n = (model_name or "").lower()
    if any(k in n for k in ("qwen", "seed", "glm")):
        extra = {"temperature": 0.7, "extra_body": {"thinking": {"type": "disabled"}}}
    elif "claude" in n:
        extra = {"temperature": 0.7}
    return LiteLlm(model=f"openai/{model_name}", api_base=base_url, api_key=api_key, drop_params=True, **extra)


# ── Tool for coder/debugger agents ──
def run_tests_tool(code: str, tool_context: ToolContext) -> dict:
    """Run the generated code against sample test cases to verify correctness.

    Args:
        code: The complete code solution to test.

    Returns:
        Dict with 'passed' (bool) and 'test_log' (str).
    """
    sample_io = json.loads(tool_context.state.get("sample_io_json", "[]"))
    entry_point = tool_context.state.get("entry_point", "")
    clean_code = parse_code(code)
    passed, test_log = evaluate_tests(sample_io, clean_code, entry_point)
    tool_context.state["test_passed"] = passed
    tool_context.state["test_log"] = test_log
    tool_context.state["current_code"] = clean_code
    return {"passed": passed, "test_log": test_log}


# ── MapCoder Orchestrator ──
class MapCoderOrchestrator(BaseAgent):
    """Custom orchestrator: Retrieval → Planning(×k) → Coding → Debugging(×t)."""

    retriever: LlmAgent
    planner: LlmAgent
    verifier: LlmAgent
    coder_agent: LlmAgent
    debugger_agent: LlmAgent
    ss: InMemorySessionService
    max_llm_calls: int = 100
    k: int = 3
    t: int = 5
    language: str = "Python3"
    dataset: str = "humaneval"

    model_config = {"arbitrary_types_allowed": True}

    async def _call(self, agent: LlmAgent, prompt: str, ctr: list, extra_state: dict = None) -> tuple:
        """Call an agent in a NEW session (stateless).
        Returns (response_text: str, session_state: dict, exceeded: bool).
        """
        init_state = dict(extra_state) if extra_state else {}
        session = await self.ss.create_session(app_name=APP, user_id="u", state=init_state)
        runner = Runner(agent=agent, app_name=APP, session_service=self.ss)

        resp, exceeded, counted = "", False, False
        async for ev in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
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

        # Count non-tool LLM call (consistent with standard: only if no fc was counted)
        if ctr is not None and not exceeded and not counted:
            ctr[0] += 1
            logger.info(f"[{agent.name}] LLM #{ctr[0]}/{self.max_llm_calls} (no tools)")
            if ctr[0] >= self.max_llm_calls:
                exceeded = True

        final_session = await self.ss.get_session(app_name=APP, user_id="u", session_id=session.id)
        final_state = final_session.state if final_session else {}
        return resp, final_state, exceeded

    async def _call_in_span(self, agent, prompt, ctr, span_name, extra_state=None):
        """Wrap _call inside OTel span for trace nesting."""
        tracer = otel_trace.get_tracer("mapcoder")
        with tracer.start_as_current_span(
            span_name,
            attributes={"orchestrator.agent": agent.name, "orchestrator.llm_calls_before": ctr[0] if ctr else 0},
        ) as span:
            resp, state, exceeded = await self._call(agent, prompt, ctr, extra_state)
            span.set_attribute("orchestrator.llm_calls_after", ctr[0] if ctr else 0)
            span.set_attribute("orchestrator.exceeded", exceeded)
            return resp, state, exceeded

    def _evt(self, ctx: InvocationContext, author: str, text: str) -> Event:
        """Create trace event."""
        return Event(
            invocation_id=ctx.invocation_id,
            author=author,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Extract user query from session events
        query = ""
        for ev in reversed(ctx.session.events):
            if ev.content and ev.content.role == "user" and ev.content.parts:
                query = ev.content.parts[0].text or ""
                if query:
                    break
        if not query:
            query = ctx.session.state.get("query", "")
        if not query:
            yield self._evt(ctx, "orchestrator", "No query provided.")
            return

        state = ctx.session.state
        sample_io = get_sample_io(
            query,
            self.dataset,
            test_cases_json=state.get("test_cases_json", ""),
            input_output=state.get("input_output", ""),
        )
        entry_point = state.get("entry_point", "")
        logger.info(f"[MapCoder] start, query={query[:80]!r}, sample_io_count={len(sample_io)}, k={self.k}, t={self.t}")

        # State injected into coder/debugger sessions for run_tests_tool
        tool_state = {"sample_io_json": json.dumps(sample_io), "entry_point": entry_point}
        ctr = [0]
        budget = lambda: ctr[0] < self.max_llm_calls
        tracer = otel_trace.get_tracer("mapcoder")

        # ── Step 1: Retrieval Agent — find k similar exemplars + algorithm ──
        k_str = MAPPING.get(self.k, str(self.k))
        retrieval_prompt = INPUT_KB_EXEMPLARS.format(query=query, k_str=k_str, language=self.language)

        raw, _, exceeded = await self._call_in_span(self.retriever, retrieval_prompt, ctr, "retrieval_call_0")
        yield self._evt(ctx, "retriever", raw)
        if exceeded:
            state["final_code"] = ""
            return

        # Post-process retrieval output (consistent with original)
        for trim_str in [
            "# Identify the algorithm (Brute-force, Dynamic Programming, "
            "Divide-and-conquer, Greedy, Backtracking, Recursive, Binary "
            "search, and so on) that needs to be used to solve the original problem.",
            "# Write a useful tutorial about the above mentioned algorithms. "
            "Provide a high level generic tutorial for solving this types of "
            "problem. Do not generate code.",
            "# Planning to solve this problem:",
            f"# Let's think step by step to solve this problem in {self.language} programming language.",
        ]:
            raw = trim_text(raw, trim_str)
        for tag in ["algorithm", "description", "code", "planning"]:
            raw = replace_tag(raw, tag)

        parsed = parse_xml(raw)
        algorithm = parsed.get("algorithm", "")
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{algorithm}"
        sample_io_prompt = f"## Sample Test cases: \n{get_sample_io_str(sample_io)}\n"
        problems = parsed.get("problem", [])
        if isinstance(problems, dict):
            problems = [problems]
        logger.info(f"[MapCoder] retrieved {len(problems)} exemplars, algorithm={str(algorithm)[:80]!r}")

        # ── Step 2: Planning Agent × k ──
        plannings = []
        for i, example in enumerate(problems):
            if not budget():
                break
            desc = example.get("description", "") if isinstance(example, dict) else ""
            plan = example.get("planning", "") if isinstance(example, dict) else ""

            planning_prompt = PLANNING_PROMPT.format(
                example_problem=desc,
                example_planning=plan,
                algorithm_prompt=algorithm_prompt,
                prompt=query,
                sample_io_prompt=sample_io_prompt,
            )
            # Fixed: removed outer planning_call span wrapping _call_in_span
            planning_text, _, exceeded = await self._call_in_span(
                self.planner, planning_prompt, ctr, f"planner_call_{i}"
            )
            yield self._evt(ctx, "planner", planning_text)
            if exceeded:
                break

            verify_prompt = PLANNING_FOR_VERIFICATION.format(
                language=self.language,
                query=query,
                planning=planning_text,
            )
            ver_raw, _, exceeded = await self._call_in_span(self.verifier, verify_prompt, ctr, f"verifier_call_{i}")
            yield self._evt(ctx, "verifier", ver_raw)

            ver_raw = replace_tag(ver_raw, "explanation")
            ver_raw = replace_tag(ver_raw, "confidence")
            ver_parsed = parse_xml(ver_raw)
            try:
                confidence = int(str(ver_parsed.get("confidence", "0")).strip())
            except (ValueError, TypeError):
                confidence = 0

            plannings.append((planning_text, confidence, example))
            logger.info(f"[MapCoder] planning {i}: confidence={confidence}")
            if exceeded:
                break

        # Sort by confidence descending
        plannings.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            f"[MapCoder] {len(plannings)} plannings sorted, top confidence={plannings[0][1] if plannings else 'N/A'}"
        )

        # ── Steps 3+4: Coding + Debugging per planning ──
        final_code = ""
        solved = False
        std_input_prompt = ""

        for plan_idx, (planning_text, confidence, example) in enumerate(plannings):
            if not budget():
                break

            # Fixed: removed outer code_debug_plan span wrapping _call_in_span
            coder_prompt = FINAL_CODE_GENERATION.format(
                language=self.language,
                algorithm_prompt=algorithm_prompt,
                prompt=query,
                planning=planning_text,
                sample_io_prompt=sample_io_prompt,
                std_input_prompt=std_input_prompt,
            )
            coder_resp, coder_st, exceeded = await self._call_in_span(
                self.coder_agent,
                coder_prompt,
                ctr,
                f"coder_call_{plan_idx}",
                extra_state=tool_state,
            )
            yield self._evt(ctx, "coder", coder_resp)

            code = coder_st.get("current_code", "") or parse_code(coder_resp)
            response_text = f"## Planning: {planning_text}\n## Code:\n```\n{code}\n```"
            passed = coder_st.get("test_passed", False)
            test_log = coder_st.get("test_log", "")
            if not test_log:
                passed, test_log = evaluate_tests(sample_io, code, entry_point)
            if exceeded:
                final_code = code
                break

            for dbg_i in range(1, self.t + 1):
                logger.info(f"[MapCoder] plan={plan_idx}, debug={dbg_i}/{self.t}, passed={passed}, llm_calls={ctr[0]}")
                if passed or not budget():
                    break
                debug_prompt = IMPROVING_CODE.format(
                    language=self.language,
                    algorithm_prompt=algorithm_prompt,
                    prompt=query,
                    response=response_text,
                    test_log=test_log,
                    std_input_prompt=std_input_prompt,
                )
                debug_resp, debug_st, exceeded = await self._call_in_span(
                    self.debugger_agent,
                    debug_prompt,
                    ctr,
                    f"debugger_call_{plan_idx}_{dbg_i}",
                    extra_state=tool_state,
                )
                yield self._evt(ctx, "debugger", debug_resp)
                code = debug_st.get("current_code", "") or parse_code(debug_resp)
                response_text = debug_resp
                passed = debug_st.get("test_passed", False)
                test_log = debug_st.get("test_log", "")
                if not test_log:
                    passed, test_log = evaluate_tests(sample_io, code, entry_point)
                if exceeded:
                    break

            if passed:
                solved = True
                final_code = code
                logger.info(f"[MapCoder] SOLVED with plan {plan_idx}, confidence={confidence}")
                break
            final_code = code

        state["final_code"] = final_code
        logger.info(f"[MapCoder] done, solved={solved}, code_len={len(final_code)}, total_llm_calls={ctr[0]}")
        yield self._evt(ctx, "assistant", final_code)


# ── Build orchestrator ──
def build_orchestrator(
    ss: InMemorySessionService,
    model_name: str,
    base_url: str,
    api_key: str,
    k: int = 3,
    t: int = 5,
    max_llm_calls: int = 100,
    dataset: str = "humaneval",
) -> MapCoderOrchestrator:
    """Build MapCoder pipeline: 4 LlmAgents + 1 custom orchestrator."""
    mk = lambda: make_model(model_name, base_url, api_key)
    language = "Python3"

    retriever = LlmAgent(
        name="retriever",
        model=mk(),
        instruction="You are a retrieval agent that finds relevant exemplar problems and identifies algorithms.",
        output_key="retriever_output",
    )
    planner = LlmAgent(
        name="planner",
        model=mk(),
        instruction="You are a planning agent that generates step-by-step plans for competitive programming problems.",
        output_key="planner_output",
    )
    verifier = LlmAgent(
        name="verifier",
        model=mk(),
        instruction="You are a verification agent that evaluates planning correctness.",
        output_key="verifier_output",
    )
    coder_agent = LlmAgent(
        name="coder",
        model=mk(),
        instruction=(
            "You are a coding agent that generates code from plans. "
            "After generating the code, you MUST call run_tests_tool with the complete code to verify it passes the test cases."
        ),
        output_key="coder_output",
        tools=[run_tests_tool],
    )
    debugger_agent = LlmAgent(
        name="debugger",
        model=mk(),
        instruction=(
            "You are a debugging agent that fixes code based on test reports. "
            "After generating the fixed code, you MUST call run_tests_tool with the complete code to verify correctness."
        ),
        output_key="debugger_output",
        tools=[run_tests_tool],
    )

    return MapCoderOrchestrator(
        name="orchestrator",
        retriever=retriever,
        planner=planner,
        verifier=verifier,
        coder_agent=coder_agent,
        debugger_agent=debugger_agent,
        ss=ss,
        max_llm_calls=max_llm_calls,
        k=k,
        t=t,
        language=language,
        dataset=dataset,
        sub_agents=[retriever, planner, verifier, coder_agent, debugger_agent],
    )


# ── Run single sample ──
async def run_agent(runner: Runner, ss: InMemorySessionService, sample: dict, **kwargs) -> str:
    """Run MapCoder on a single sample, return final code."""
    query = sample["query"] if isinstance(sample, dict) else sample
    init_state = {"query": query}

    if isinstance(sample, dict):
        if sample.get("test_cases"):
            init_state["test_cases_json"] = json.dumps(sample["test_cases"])
        if sample.get("input_output"):
            init_state["input_output"] = (
                sample["input_output"]
                if isinstance(sample["input_output"], str)
                else json.dumps(sample["input_output"])
            )
        if sample.get("entry_point"):
            init_state["entry_point"] = sample["entry_point"]

    session = await ss.create_session(app_name=APP, user_id="user", state=init_state)
    msg = types.Content(role="user", parts=[types.Part(text=query)])

    last_assistant_text = ""
    last_code_text = ""
    async for ev in runner.run_async(user_id="user", session_id=session.id, new_message=msg):
        if ev.content and ev.content.parts:
            text = (ev.content.parts[0].text or "").strip()
            if text:
                if ev.author == "assistant":
                    last_assistant_text = text
                elif ev.author in ("coder", "debugger"):
                    last_code_text = text

    # Get final_code from session state (set by orchestrator)
    final_session = await ss.get_session(app_name=APP, user_id="user", session_id=session.id)
    resp = last_assistant_text
    if not resp and final_session and final_session.state:
        resp = final_session.state.get("final_code", "")
    if not resp and last_code_text:
        resp = parse_code(last_code_text)
        logger.info(f"[run_agent] fallback to parsed event text, len={len(resp)}")

    if not resp:
        logger.warning(f"[run_agent] empty response, session={session.id}, query={query[:80]!r}")
    else:
        logger.info(f"[run_agent] response len={len(resp)}, preview={resp[:120]!r}")
    return resp


# ── Main entry ──
async def main():

    p = argparse.ArgumentParser(description="MapCoder Agent")
    p.add_argument("--task_id", default="default")
    p.add_argument("--query", default="")
    p.add_argument("--max_llm_calls", type=int, default=100)
    p.add_argument("--k", type=int, default=3, help="Number of exemplar problems to retrieve")
    p.add_argument("--t", type=int, default=5, help="Max debug iterations per planning")
    p.add_argument("--dataset", default="humaneval", help="Dataset name: humaneval, mbpp, livecodebench")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--model", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    # Support passing sample as JSON file
    p.add_argument("--sample_file", default=None, help="JSON file with sample data (query, test_cases, etc.)")
    p.add_argument("--fault_name", default=None, help="Fault name (e.g. 'llm_error_single'). None=no injection.")
    args = p.parse_args()

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "sk-xxx")

    # Build sample from args or file
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

    # Setup output directory
    task_dir = os.path.join(args.output_dir, args.task_id)
    trace_dir = os.path.join(task_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)

    # Save input
    with open(os.path.join(task_dir, "input.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": args.task_id,
                "query": sample["query"],
                "model": model,
                "max_llm_calls": args.max_llm_calls,
                "k": args.k,
                "t": args.t,
                "dataset": args.dataset,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # File logger
    fh = logging.FileHandler(os.path.join(task_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    # Setup tracing
    tp, _ = setup_tracing("MapCoder", logger=logger, trace_dir=trace_dir)
    fault_engine = setup_fault_injection(args.fault_name)

    # Build and run
    ss = InMemorySessionService()
    orchestrator = build_orchestrator(
        ss=ss,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        k=args.k,
        t=args.t,
        max_llm_calls=args.max_llm_calls,
        dataset=args.dataset,
    )
    runner = Runner(agent=orchestrator, app_name=APP, session_service=ss)

    logger.info(f"[main] task={args.task_id}, model={model}, k={args.k}, t={args.t}, dataset={args.dataset}")
    logger.info(f"[main] query={sample['query'][:120]!r}")

    resp = await run_agent(runner, ss, sample)

    teardown_fault_injection(fault_engine)

    output_data = {"task_id": args.task_id, "final_answer": resp, "code_length": len(resp)}
    if fault_engine is not None:
        output_data["_fault_name"] = args.fault_name
        output_data["_fault_fired"] = len(fault_engine.log)
        output_data["_fault_log"] = fault_engine.log

    with open(os.path.join(task_dir, "output.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

    if resp:
        logger.info(f"[main] OK, task={args.task_id}, code_len={len(resp)}, preview={resp[:120]!r}")
    else:
        logger.warning(f"[main] EMPTY response for task={args.task_id}")

    if tp:
        tp.force_flush()


if __name__ == "__main__":
    asyncio.run(main())
