# run_autogen.py
import asyncio
import json
import os
import io
import sys
import subprocess
import logging
import argparse
from typing import AsyncGenerator, List, Dict, Tuple
from dotenv import load_dotenv
from typing_extensions import override

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
logger = logging.getLogger("Agent")

load_dotenv("")

APP = "Agent"
TERM_MSG = "TERMINATE"

# ── Prompts ──────────────────────

ASSISTANT_AGENT_SYSTEM_MESSAGE_CODER = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
In every response, please restate what you think is the most recent and corresponding block of code (like in a python coding block or in a sh coding block) if the initial question is code related or you use code in previous or current response.
In every response, whether you use code or not, please conclude with the current final answer or current possible code output in a specific format at the end(like "The final answer is" with the final answer in the box or "The possible output is" with the possible output in the box at the end) to ensure the answer or output is available even if the conversation is interrupted.
Reply "TERMINATE" in the end when everything is done."""

ASSISTANT_AGENT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your language skills.
You should not write any code and should solve the problem using your language skills as far as you can.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
In every response, please conclude with the current final answer in a specific format at the end(like "The final answer is" with the final answer in the box at the end) to ensure the answer is available even if the conversation is interrupted.
Reply "TERMINATE" in the end when everything is done."""

DEFAULT_USER_PROXY_AGENT_SYSTEM_MESSAGE = ""


# ── ADK Tools for code execution ──────────────────────────────

def execute_python_code(code: str) -> dict:
    """Execute Python code and return the output.
    
    Args:
        code: Python code to execute.
    
    Returns:
        Dict with 'stdout' and optionally 'error' fields.
    """
    try:
        local_vars = {}
        stdout_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        exec(code, {}, local_vars)
        sys.stdout = sys.__stdout__
        output = stdout_buffer.getvalue()
        result = {"stdout": output}
        logger.info(f"[execute_python_code] success, output_len={len(output)}")
        return result
    except Exception as e:
        sys.stdout = sys.__stdout__
        logger.error(f"[execute_python_code] error: {e}")
        return {"stdout": "", "error": str(e)}


def execute_shell_command(command: str) -> dict:
    """Execute a shell command and return the output.
    
    Args:
        command: Shell command to execute.
    
    Returns:
        Dict with 'stdout' and optionally 'error' fields.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        out = {"stdout": result.stdout.strip()}
        if result.stderr.strip():
            out["error"] = result.stderr.strip()
        logger.info(f"[execute_shell_command] done, stdout_len={len(result.stdout)}")
        return out
    except subprocess.TimeoutExpired:
        logger.error("[execute_shell_command] timeout")
        return {"stdout": "", "error": "Timeout (30s)"}
    except Exception as e:
        logger.error(f"[execute_shell_command] error: {e}")
        return {"stdout": "", "error": str(e)}


# ── Model factory ──────────────────────────────────────────


def make_model(model_name: str, base_url: str, api_key: str):
    extra = {}
    n = (model_name or "").lower()
    if any(k in n for k in ("qwen", "seed", "glm")):
        extra = {"temperature": 0.7, "extra_body": {"thinking": {"type": "disabled"}}}
    elif "claude" in n:
        extra = {"temperature": 0.7}
    return LiteLlm(
        model=f"openai/{model_name}",
        api_base=base_url,
        api_key=api_key,
        drop_params=True,
        **extra,
    )


# ── Orchestrator ───────────────────────────────────────────


class AutoGenOrchestrator(BaseAgent):
    """
    Two-agent orchestrator faithful to original AutoGen_Main.inference().

    Loop logic:
      1. Send query to assistant
      2. If TERMINATE in response → done
      3. For each turn:
         a. Assistant response → user_proxy (either code execution or LLM)
         b. If TERMINATE in proxy response → done (return assistant's last response)
         c. Proxy response → assistant
         d. If TERMINATE in assistant response → done
    """

    assistant: LlmAgent
    user_proxy: LlmAgent
    max_turn: int = 3
    max_llm_calls: int = 100
    code_execute: bool = False
    ss: InMemorySessionService

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        ss: InMemorySessionService,
        max_turn: int = 3,
        max_llm_calls: int = 100,
        code_execute: bool = False,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "sk-xxx",
    ):
        mk = lambda: make_model(model_name, base_url, api_key)

        assistant_prompt = ASSISTANT_AGENT_SYSTEM_MESSAGE_CODER if code_execute else ASSISTANT_AGENT_SYSTEM_MESSAGE

        assistant = LlmAgent(
            name="assistant",
            model=mk(),
            instruction=assistant_prompt,
        )
        # user_proxy always has tools for code execution when code_execute is enabled
        # The LLM will decide when to call tools based on the assistant's response
        user_proxy_tools = [execute_python_code, execute_shell_command] if code_execute else []
        user_proxy_instruction = (
            "You are a code executor agent. When you receive a message containing code blocks "
            "(```python or ```sh), you MUST execute the code using the appropriate tool:\n"
            "- For Python code (```python), use execute_python_code tool\n"
            "- For shell commands (```sh), use execute_shell_command tool\n"
            "Extract the code from the code block and pass it to the tool. "
            "Report the execution results back to the assistant."
            if code_execute
            else DEFAULT_USER_PROXY_AGENT_SYSTEM_MESSAGE
        )
        user_proxy = LlmAgent(
            name="user_proxy",
            model=mk(),
            instruction=user_proxy_instruction,
            tools=user_proxy_tools,
        )

        super().__init__(
            name="autogen_orchestrator",
            assistant=assistant,
            user_proxy=user_proxy,
            max_turn=max_turn,
            max_llm_calls=max_llm_calls,
            code_execute=code_execute,
            ss=ss,
            sub_agents=[assistant, user_proxy],
        )

    # ── Core _call ─────────────

    async def _call(
        self,
        agent: LlmAgent,
        prompt: str,
        history: List[Dict],
        ctr: List[int],
    ) -> Tuple[str, bool]:
        """
        Call an agent within the CURRENT OTel context.
        Creates a fresh session per call (like standard).
        History is formatted as conversation context.

        Returns (response_text, exceeded).
        """
        if history:
            lines = [f"[{m['role']}]: {m['content']}" for m in history]
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

        resp = ""
        exceeded = False
        counted = False

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

    async def _call_in_span(
        self,
        agent: LlmAgent,
        prompt: str,
        history: List[Dict],
        ctr: List[int],
        span_name: str,
    ) -> Tuple[str, bool]:
        """Wrap _call inside a manual OTel span for trace nesting."""
        tracer = otel_trace.get_tracer("autogen_orchestrator")
        with tracer.start_as_current_span(
            span_name,
            attributes={
                "orchestrator.agent": agent.name,
                "orchestrator.llm_calls_before": ctr[0] if ctr else 0,
            },
        ) as span:
            result, exceeded = await self._call(agent, prompt, history, ctr)
            span.set_attribute("orchestrator.llm_calls_after", ctr[0] if ctr else 0)
            span.set_attribute("orchestrator.exceeded", exceeded)
            return result, exceeded

    # ── Event helper ───────────────────────────────────────

    def _evt(self, ctx: InvocationContext, author: str, text: str) -> Event:
        return Event(
            invocation_id=ctx.invocation_id,
            author=author,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
        )

    # ── Main orchestration loop ────────────────────────────

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Faithful reproduction of original AutoGen_Main.inference():

        1. query → assistant (with assistant_agent_history)
        2. If TERMINATE → return
        3. Loop max_turn-1 times:
           a. assistant_response → user_proxy
              - If code_execute: check for code, execute directly
              - Else (or no code found): call user_proxy LLM
           b. If TERMINATE in proxy_response → return assistant_response
           c. proxy_response → assistant
           d. If TERMINATE in assistant_response → return
        4. Return last assistant_response
        """
        # Extract user query from session
        query = ""
        for ev in reversed(ctx.session.events):
            if ev.content and ev.content.role == "user" and ev.content.parts:
                query = ev.content.parts[0].text or ""
                if query:
                    break
        if not query:
            yield self._evt(ctx, "autogen_orchestrator", "No query provided.")
            return

        # Independent histories per agent (faithful to original)
        user_proxy_history: List[Dict] = []
        assistant_agent_history: List[Dict] = []
        ctr = [0]
        budget = lambda: ctr[0] < self.max_llm_calls

        tracer = otel_trace.get_tracer("autogen_orchestrator")
        assistant_call_idx = 0
        proxy_call_idx = 0

        # ── Turn 0: query → assistant ──
        assistant_agent_response, exceeded = await self._call_in_span(
            self.assistant,
            query,
            assistant_agent_history,
            ctr,
            f"assistant_call_{assistant_call_idx}",
        )
        assistant_call_idx += 1

        # Update histories (mirrors original)
        user_proxy_history.append({"role": "assistant", "content": query})
        assistant_agent_history.append({"role": "user", "content": query})
        assistant_agent_history.append({"role": "assistant", "content": assistant_agent_response})

        yield self._evt(ctx, "assistant", assistant_agent_response)
        logger.info(
            f"[Turn 0] assistant len={len(assistant_agent_response)}, "
            f"TERMINATE={TERM_MSG in assistant_agent_response}"
        )

        if TERM_MSG in assistant_agent_response:
            ctx.session.state["final"] = assistant_agent_response.replace(TERM_MSG, "").strip()
            return

        if exceeded:
            ctx.session.state["final"] = assistant_agent_response.replace(TERM_MSG, "").strip()
            return

        # ── Main loop (max_turn - 1 iterations, like original) ──
        for turn in range(1, self.max_turn):
            if not budget():
                break
            logger.info(f"[autogen_orchestrator] turn {turn}, LLM calls: {ctr[0]}/{self.max_llm_calls}")

            # ── Assistant response → User Proxy ──
            # user_proxy will use tools to execute code if code_execute is enabled
            with tracer.start_as_current_span(
                f"turn_{turn}",
                attributes={"orchestrator.turn": turn},
            ):
                user_proxy_response, exceeded = await self._call_in_span(
                    self.user_proxy,
                    assistant_agent_response,
                    user_proxy_history,
                    ctr,
                    f"user_proxy_call_{proxy_call_idx}",
                )
                proxy_call_idx += 1
                if exceeded:
                    ctx.session.state["final"] = assistant_agent_response.replace(TERM_MSG, "").strip()
                    return

                # Update histories
                user_proxy_history.append({"role": "user", "content": assistant_agent_response})
                assistant_agent_history.append({"role": "assistant", "content": assistant_agent_response})

                yield self._evt(ctx, "user_proxy", user_proxy_response)
                logger.info(
                    f"[Turn {turn}] user_proxy len={len(user_proxy_response)}, "
                    f"TERMINATE={TERM_MSG in user_proxy_response}"
                )

                if TERM_MSG in user_proxy_response:
                    ctx.session.state["final"] = assistant_agent_response.replace(TERM_MSG, "").strip()
                    return

                # ── User Proxy response → Assistant ──
                assistant_agent_response, exceeded = await self._call_in_span(
                    self.assistant,
                    user_proxy_response,
                    assistant_agent_history,
                    ctr,
                    f"assistant_call_{assistant_call_idx}",
                )
                assistant_call_idx += 1

                # Update histories
                user_proxy_history.append({"role": "assistant", "content": user_proxy_response})
                assistant_agent_history.append({"role": "user", "content": user_proxy_response})
                assistant_agent_history.append({"role": "assistant", "content": assistant_agent_response})

                yield self._evt(ctx, "assistant", assistant_agent_response)
                logger.info(
                    f"[Turn {turn}] assistant len={len(assistant_agent_response)}, "
                    f"TERMINATE={TERM_MSG in assistant_agent_response}"
                )

                if TERM_MSG in assistant_agent_response:
                    ctx.session.state["final"] = assistant_agent_response.replace(TERM_MSG, "").strip()
                    return

                if exceeded:
                    ctx.session.state["final"] = assistant_agent_response.replace(TERM_MSG, "").strip()
                    return

        # Loop exhausted — return last assistant response
        ctx.session.state["final"] = assistant_agent_response.replace(TERM_MSG, "").strip()


# ── Main ───────────────────────────────────────────────────


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", default="default")
    p.add_argument("--query", default="Calculate the sum of the first 20 prime numbers using Python code.")
    p.add_argument("--max_turn", type=int, default=3)
    p.add_argument("--max_llm_calls", type=int, default=100)
    p.add_argument(
        "--code_execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable code execution (default on)",
    )
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--model", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    # ── fault injection args ──
    p.add_argument("--fault_name", default=None, help="Fault name (e.g. 'llm_error_single'). None=no injection.")
    args = p.parse_args()

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "sk-xxx")

    task_dir = os.path.join(args.output_dir, args.task_id)
    trace_dir = os.path.join(task_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)

    with open(os.path.join(task_dir, "input.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": args.task_id,
                "query": args.query,
                "model": model,
                "max_turn": args.max_turn,
                "max_llm_calls": args.max_llm_calls,
                "code_execute": args.code_execute,
                "fault_name": args.fault_name,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fh = logging.FileHandler(os.path.join(task_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    tp, _ = setup_tracing("AutoGen", logger=logger, trace_dir=trace_dir)

    # ── install fault injection before any LLM call ──
    fault_engine = setup_fault_injection(args.fault_name)

    ss = InMemorySessionService()
    runner = Runner(
        agent=AutoGenOrchestrator(
            ss=ss,
            max_turn=args.max_turn,
            max_llm_calls=args.max_llm_calls,
            code_execute=args.code_execute,
            model_name=model,
            base_url=base_url,
            api_key=api_key,
        ),
        app_name=APP,
        session_service=ss,
    )

    logger.info(
        f"Task: [{args.task_id}] {args.query} | Model: {model} | "
        f"code_execute: {args.code_execute} | fault: {args.fault_name or 'none'}"
    )

    session = await ss.create_session(app_name=APP, user_id="user")
    events_log = []
    async for ev in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=args.query)]),
    ):
        if ev.content and ev.content.parts:
            text = (ev.content.parts[0].text or "").strip()
            if text:
                logger.info(f"[{args.task_id}][{(ev.author or '?').upper()}] {text[:200]}")
                events_log.append({"author": ev.author or "?", "text": text})

    # ── teardown fault injection and collect metadata ──
    teardown_fault_injection(fault_engine)

    s = await ss.get_session(app_name=APP, user_id="user", session_id=session.id)
    final = (s.state or {}).get("final", "")
    if not final:
        for e in reversed(events_log):
            if e["author"] == "assistant":
                final = e["text"].replace(TERM_MSG, "").strip()
                if final:
                    break

    if final:
        logger.info(f"[OK] [{args.task_id}] Final: {final[:200]}")

    # ── output includes fault metadata ──
    output_data = {"task_id": args.task_id, "final_answer": final, "events": events_log}
    if fault_engine is not None:
        output_data["_fault_name"] = args.fault_name
        output_data["_fault_fired"] = len(fault_engine.log)
        output_data["_fault_log"] = fault_engine.log

    with open(os.path.join(task_dir, "output.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

    if tp:
        tp.force_flush()


if __name__ == "__main__":
    asyncio.run(main())
