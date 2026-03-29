# run_evomac.py
import asyncio
import json
import os
import re
import time
import signal
import tempfile
import subprocess
import logging
import argparse
from collections import defaultdict, deque
from typing import AsyncGenerator

from typing_extensions import override
from dotenv import load_dotenv

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from google.genai import types
from opentelemetry import trace as otel_trace
from util import setup_tracing, setup_fault_injection, teardown_fault_injection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger("EvoMAC")

load_dotenv("")

APP = "evomac"


# ── Prompts ────────────────────────────────
INITIAL_CODING_ROLE = """EvoMAC is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'."""

INITIAL_CODING = """Here is a function completion task:
Task: "{task}".
Please think step by step and complete the function.
Your answer should must strictly follow a markdown code block format, where the following tokens must be replaced such that "FILENAME" is the lowercase file name including the file extension, "LANGUAGE" in the programming language, "DOCSTRING" is a string literal specified in source code that is used to document a specific segment of code, and "CODE" is the original code:
FILENAME
```LANGUAGE
\'\'\'
DOCSTRING
\'\'\'
CODE
```"""

ORGANIZING = """Here is a function completion task:
Task: "{task}".
Programming Language: "{language}"
The implemention of the task(source codes) are: "{codes}"
Your goal is to organize a coding team to complete the function completion task.
You should follow the following format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. Each task is assigned to a programmer, and the workflow shows the dependencies between tasks. 
### COMPOSITION
```
Task 1: Task 1 description
Task 2: Task 2 description
```
### WORKFLOW
```
Task 1: []
Task 2: [Task 1]
```
Please note that the decomposition should be both effective and efficient.
1) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].
2) The WORKFLOW should not contain circles!
3) The programmer number and the task number should be as small as possible.
4) Your task should not include anything related to testing, writing document or computation cost optimizing."""

SUBCODECOMPLETE = """Here is a function completion task:
Task: "{task}".
Programming Language: "{language}"
The implemention of the task(source codes) are: "{codes}"
I will give you a subtask below, you should carefully read the subtask and do the following things: 
1) If the subtask is a specific task related to the function completion, please think step by step and reason yourself to finish the task.
2) If the subtask is a test report of the code, please check the source code and the test report, and then think step by step and reason yourself to fix the bug. 
Subtask description: "{subtask}"
3) You should output the COMPLETE code content in each file. Each file must strictly follow a markdown code block format, where the following tokens must be replaced such that "FILENAME" is the lowercase file name including the file extension, "LANGUAGE" in the programming language, "DOCSTRING" is a string literal specified in source code that is used to document a specific segment of code, and "CODE" is the original code. Format:
FILENAME
```LANGUAGE
\'\'\'
DOCSTRING
\'\'\'
CODE
```
Note that no placeholder (such as 'pass' in Python) and you should strictly following the required format.
!!! This message has the highest priority: DO NOT write main() function or implement any testcase(such as assert) in your code. You just need to write or modified the task function itself."""

TESTORGANIZING = """According to the function completion requirements listed below: 
Task: "{task}".
Programming Language: "{language}"
Your goal is to organize a testing team to complete the function completion task.
There are one default tasks: 
1) use some simplest case to test the logic. The case must be as simple as possible, and you should ensure every 'assert' you write is 100% correct
Follow the format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. 
### COMPOSITION
```
Task 1: Task 1 description
Task 2: Task 2 description
```
### WORKFLOW
```
Task 1: []
Task 2: [Task 1]
```
Note that:
1) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].
2) DO NOT include things like implement the code in your task description.
3) The task number should be as small as possible. Only one task is also acceptable."""

TESTCODECOMPLETE = """According to the function completion requirements listed below: 
Task: "{task}".
Please locate the example test case given in the function definition, these test case will be used latter.
The implemention of the function is:
"{codes}"
Testing Task description: "{subtask}"
According to example test case in the Task description, please write these test cases to locate the bugs. You should not add any other testcases except for the example test case given in the Task description
The output must strictly follow a markdown code block format, where the following tokens must be replaced such that "FILENAME" is "{test_file_name}", "LANGUAGE" in the programming language,"REQUIREMENTS" is the targeted requirement of the test case, and "CODE" is the test code that is used to test the specific requirement of the file. Format:
FILENAME
```LANGUAGE
\'\'\'
REQUIREMENTS
\'\'\'
CODE
```
You will start with the "{test_file_name}" and finish the code follows in the strictly defined format.
Please note that:
1) The code should be fully functional. No placeholders (such as 'pass' in Python).
2) You should write the test file with 'unittest' python library. Import the functions you need to test if necessary.
3) The test case should be as simple as possible, and the test case number should be less than 5.
4) According to example test case in the Task description, please only write these test cases to locate the bugs. You should not add any other testcases by yourself except for the example test case given in the Task description"""

UPDATING = """Here is a function completion task:
Task:
{task}.
Source Codes:
{codes}
Current issues: 
{issues}.
According to the task, source codes and current issues given above, design a programmmer team to solve current issues.
You should follow the following format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. Each task is assigned to a programmer, and the workflow shows the dependencies between tasks.
### COMPOSITION
```
Programmer 1: Task 1 description
Programmer 2: Task 2 description
```
### WORKFLOW
```
Programmer 1: []
Programmer 2: [Programmer 1]
```
Please note that:
1) You should repeat exactly the current issues in the task description of module COMPOSITION in a line. For example: Programmer 1: AssertionError: function_name(input) != expected_output. The actual output is: actual_output.
2) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].
3) The WORKFLOW should not contain circles!
4) The programmer number and the task number should be as small as possible. One programmer is also acceptable.
5) DO NOT include things like implement the code in your task description."""


# ── Helper: Codes ──────────────────────────────
class Codes:
    """Parse and manage code files from LLM markdown output."""

    def __init__(self, generated_content="", target_file=None):
        self.codebooks = {}
        if generated_content:
            self._parse(generated_content, target_file)

    def _parse(self, content, target_file):
        regex = r"(.*?)```[Pp](?:ython3?|y)\s*\n(.*?)```"
        matches = list(re.finditer(regex, content, re.DOTALL))
        if not matches:
            regex = r"(.*?)```\w*\s*\n(.*?)```"
            matches = list(re.finditer(regex, content, re.DOTALL))
        if not matches:
            logger.warning(f"[Codes._parse] no code block found, content_len={len(content)}")
            return
        for match in matches:
            code = match.group(2)
            if "CODE" in code:
                continue
            if target_file is None:
                filename = self._extract_filename(match.group(1), code)
            else:
                filename = target_file
            if not filename:
                filename = "main.py"
                logger.warning(f"[Codes._parse] filename extraction failed, using '{filename}'")
            if code and code.strip():
                self.codebooks[filename] = self._format_code(code)

    @staticmethod
    def _extract_filename(header, code):
        name = ""
        for m in re.finditer(r"(\w+\.\w+)", header):
            name = m.group().lower()
        if "__main__" in code and "test" not in code and "check" not in code:
            name = "main.py"
        if not name:
            for m in re.finditer(r"class (\S+?):\n", code, re.DOTALL):
                name = m.group(1).lower().split("(")[0] + ".py"
        return name

    @staticmethod
    def _format_code(code):
        lines = code.split("\n")
        start = ""
        if lines and lines[0].strip() and lines[0].lower() != "python":
            start = lines[0] + "\n"
        return start + "\n".join(l for l in lines[1:] if l.strip())

    def update(self, generated_content, target_file=None):
        new = Codes(generated_content, target_file)
        for k, v in new.codebooks.items():
            self.codebooks[k] = v

    def get_codes(self) -> str:
        parts = []
        for fn, code in self.codebooks.items():
            ext = "python" if fn.endswith(".py") else fn.split(".")[-1]
            parts.append(f"{fn}\n```{ext}\n{code}\n```")
        return "\n\n".join(parts)

    def get_raw_codes(self) -> str:
        for fn, code in self.codebooks.items():
            if not fn.startswith("test_requirement_"):
                return code
        return ""


# ── Helper: Organization ───────────────────────
class Organization:
    """Parse workflow organization from LLM output."""

    def __init__(self):
        self.organization = {}

    def update(self, generated_content, filename="organization.json"):
        parsed = self._parse(generated_content)
        self.organization[filename] = parsed

    def get_orgs(self):
        return list(self.organization.values())

    def _parse(self, content):
        composition, workflow = {}, {}
        for match in re.finditer(r"(.+?)\n```.*?\n(.*?)```", content, re.DOTALL):
            header, body = match.group(1), match.group(2).strip()
            if "COMPOSITION" in header:
                composition = self._parse_entries(body, as_workflow=False)
            elif "WORKFLOW" in header:
                workflow = self._parse_entries(body, as_workflow=True)
        return {"composition": composition, "workflow": workflow}

    @staticmethod
    def _parse_entries(content, as_workflow=False):
        sep = r"\n\n" if "\n\n" in content else r"\n"
        pattern = rf"(?:Programmer|Task) \d+:.*?(?:{sep}|\Z)"
        result = {}
        all_deps = []
        for match in re.finditer(pattern, content, re.DOTALL):
            parts = match.group().strip().split(":")
            key = parts[0].strip().replace("Task", "Programmer")
            value = ":".join(parts[1:]).strip()
            if as_workflow:
                deps_str = value.strip().strip("[]").replace("'", "").replace('"', "")
                deps = [d.replace("Task", "Programmer").strip() for d in deps_str.split(",") if d.strip()]
                result[key] = deps
                all_deps.extend(deps)
            else:
                result[key] = value
        if as_workflow:
            for dep in set(all_deps):
                if dep not in result:
                    result[dep] = []
        return result


# ── Helper functions ─────────────────────────────────────────
def topological_sort(workflow: dict) -> list:
    """Topological sort of workflow DAG."""
    in_deg = defaultdict(int)
    adj = defaultdict(list)
    for node, deps in workflow.items():
        for d in deps:
            adj[d].append(node)
            in_deg[node] += 1
    queue = deque(n for n in workflow if in_deg[n] == 0)
    order = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nb in adj[cur]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    if len(order) != len(workflow):
        logger.warning(f"[topo_sort] cycle detected: sorted {len(order)}/{len(workflow)} nodes")
    return order


def execute_tests(codebooks: dict, test_codebooks: dict, test_file: str) -> tuple:
    """Run test file against code in a temp directory. Returns (has_bug, result)."""
    success = "The codes run successfully without errors."
    if test_file not in test_codebooks:
        logger.warning(f"[execute_tests] test file {test_file} not found in test_codebooks")
        return False, success

    with tempfile.TemporaryDirectory() as tmp:
        for fn, code in codebooks.items():
            with open(os.path.join(tmp, fn), "w") as f:
                f.write(code)
        with open(os.path.join(tmp, test_file), "w") as f:
            f.write(test_codebooks[test_file])
        try:
            command = f"cd {tmp}; python3 {test_file};"
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            time.sleep(3)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception as e:
                    logger.error(f"[execute_tests] kill failed: {e}")
                    try:
                        proc.kill()
                    except Exception as e2:
                        logger.error(f"[execute_tests] force kill failed: {e2}")
            proc.wait()
            if proc.returncode == 0:
                return False, success
            err = proc.stderr.read().decode("utf-8", errors="replace")
            if err and "traceback" in err.lower():
                return True, err.replace(tmp + "/", "")
            return False, success
        except Exception as e:
            logger.error(f"[execute_tests] error: {e}")
            return True, f"Error: {e}"


# ── ADK Tool for code execution ──────────────────────────────

def run_python_code(code: str) -> dict:
    """Execute Python code in a subprocess and return the result.
    
    Args:
        code: Complete Python code to execute.
    
    Returns:
        Dict with 'stdout', 'stderr', and 'success' fields.
    """
    import sys
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        os.unlink(tmp_path)
        
        return {
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "success": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Timeout (30s)", "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False}


# ── Model factory ──────────────────────────────
def make_model(model_name: str, base_url: str, api_key: str):
    extra = {}
    n = (model_name or "").lower()
    if any(k in n for k in ("qwen", "seed", "glm")):
        extra = {"temperature": 0.7, "extra_body": {"thinking": {"type": "disabled"}}}
    elif "claude" in n:
        extra = {"temperature": 0.7}
    return LiteLlm(model=f"openai/{model_name}", api_base=base_url, api_key=api_key, drop_params=True, **extra)


# ── EvoMAC Orchestrator ───────────────
class EvoMACOrchestrator(BaseAgent):
    """EvoMAC pipeline:
    Phase 1: Initial coding
    Phase 2: CTO organizes workflow → programmers execute in topo order
    Phase 3: Iterative test-and-fix loop (organize tests → generate → run → fix)
    """

    initial_coder: LlmAgent
    cto: LlmAgent
    programmer: LlmAgent
    test_writer: LlmAgent
    ss: InMemorySessionService
    max_llm_calls: int = 100
    iteration: int = 5
    language: str = "python"

    model_config = {"arbitrary_types_allowed": True}

    async def _call(self, agent: LlmAgent, prompt: str, ctr: list) -> tuple:
        """Call an agent in a NEW session (stateless).
        Returns (response_text: str, exceeded: bool).
        """
        session = await self.ss.create_session(app_name=APP, user_id="u")
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

        if ctr is not None and not exceeded and not counted:
            ctr[0] += 1
            logger.info(f"[{agent.name}] LLM #{ctr[0]}/{self.max_llm_calls} (no tools)")
            if ctr[0] >= self.max_llm_calls:
                exceeded = True

        return resp, exceeded

    async def _call_in_span(self, agent, prompt, ctr, span_name):
        """Wrap _call inside OTel span for trace nesting."""
        tracer = otel_trace.get_tracer("evomac")
        with tracer.start_as_current_span(
            span_name,
            attributes={"orchestrator.agent": agent.name, "orchestrator.llm_calls_before": ctr[0] if ctr else 0},
        ) as span:
            resp, exceeded = await self._call(agent, prompt, ctr)
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

        codes = Codes()
        test_codes = Codes()
        org = Organization()
        ctr = [0]
        budget = lambda: ctr[0] < self.max_llm_calls

        logger.info(f"[EvoMAC] start | task={task[:80]!r}, iteration={self.iteration}, language={self.language}")

        # ── Phase 1: Initial coding ──
        resp, exceeded = await self._call_in_span(
            self.initial_coder,
            INITIAL_CODING.format(task=task),
            ctr,
            "phase1_initial_coding",
        )
        codes.update(resp)
        logger.info(f"[EvoMAC] Phase1 done | files={list(codes.codebooks.keys())}")
        yield self._evt(ctx, "initial_coder", f"Initial files: {list(codes.codebooks.keys())}")
        if exceeded:
            ctx.session.state["final_response"] = codes.get_raw_codes()
            yield self._evt(ctx, "assistant", codes.get_raw_codes())
            return

        # ── Phase 2: CTO organizes coding workflow ──
        resp, exceeded = await self._call_in_span(
            self.cto,
            ORGANIZING.format(task=task, language=self.language, codes=codes.get_codes()),
            ctr,
            "phase2_organize_workflow",
        )
        org.update(resp)
        org_data = org.get_orgs()[0] if org.get_orgs() else {"composition": {}, "workflow": {}}
        logger.info(f"[EvoMAC] Phase2 done | tasks={list(org_data.get('composition', {}).keys())}")
        yield self._evt(ctx, "cto", f"Organized: {list(org_data.get('composition', {}).keys())}")
        if exceeded:
            ctx.session.state["final_response"] = codes.get_raw_codes()
            yield self._evt(ctx, "assistant", codes.get_raw_codes())
            return

        # ── Phase 3: Execute coding workflow ──
        async for ev in self._execute_workflow(ctx, task, codes, org, ctr):
            yield ev

        # ── Phase 4: Iterative test-and-fix ──
        for i in range(self.iteration):
            if not budget():
                logger.warning(f"[EvoMAC] budget exceeded at iteration {i+1}")
                break

            logger.info(f"[EvoMAC] Iteration {i+1}/{self.iteration}: testing...")
            has_bug, reports = await self._execute_test_workflow(ctx, task, codes, test_codes, ctr)

            if not has_bug:
                logger.info(f"[EvoMAC] All tests passed at iteration {i+1}")
                yield self._evt(ctx, "evomac", f"All tests passed at iteration {i+1}")
                break

            logger.info(f"[EvoMAC] Bugs at iteration {i+1}: {reports[:200]!r}")
            yield self._evt(ctx, "evomac", f"Iter {i+1} bugs: {reports[:300]}")

            if i < self.iteration - 1 and budget():
                resp, exceeded = await self._call_in_span(
                    self.cto,
                    UPDATING.format(task=task, codes=codes.get_codes(), issues=reports),
                    ctr,
                    f"cto_update_call_{i}",
                )
                org.update(resp)
                updated_data = org.get_orgs()[0] if org.get_orgs() else {}
                logger.info(f"[EvoMAC] Updated tasks: {list(updated_data.get('composition', {}).keys())}")
                yield self._evt(ctx, "cto", f"Updated: {list(updated_data.get('composition', {}).keys())}")
                if exceeded:
                    break
                async for ev in self._execute_workflow(ctx, task, codes, org, ctr):
                    yield ev

        # ── Final output ──
        result = codes.get_raw_codes()
        ctx.session.state["final_response"] = result
        logger.info(f"[EvoMAC] done | code_len={len(result)}, total_llm_calls={ctr[0]}")
        yield self._evt(ctx, "assistant", result)

    async def _execute_workflow(self, ctx, task, codes, org, ctr):
        """Execute coding subtasks in topological order."""
        org_data = org.get_orgs()[0] if org.get_orgs() else {"composition": {}, "workflow": {}}
        comp = org_data.get("composition", {})
        wf = org_data.get("workflow", {})
        order = topological_sort(wf)
        logger.info(f"[EvoMAC] Executing coding workflow: {order}")

        for idx, phase in enumerate(order):
            if ctr[0] >= self.max_llm_calls:
                logger.warning(f"[EvoMAC] budget exceeded during workflow at phase={phase}")
                break
            subtask = comp.get(phase, phase)
            resp, exceeded = await self._call_in_span(
                self.programmer,
                SUBCODECOMPLETE.format(task=task, language=self.language, codes=codes.get_codes(), subtask=subtask),
                ctr,
                f"programmer_{phase}",
            )
            codes.update(resp)
            logger.info(f"[EvoMAC] {phase} done | files={list(codes.codebooks.keys())}")
            yield self._evt(ctx, "programmer", f"{phase} done: {list(codes.codebooks.keys())}")
            if exceeded:
                break

    async def _execute_test_workflow(self, ctx, task, codes, test_codes, ctr) -> tuple:
        """Organize tests → generate test code → execute → return (has_bug, reports)."""
        resp, exceeded = await self._call_in_span(
            self.cto,
            TESTORGANIZING.format(task=task, language=self.language),
            ctr,
            "cto_test_organize",
        )
        if exceeded:
            return False, "Budget exceeded during test organization."

        test_org = Organization()
        test_org.update(resp, filename="test_organization.json")
        test_data = test_org.get_orgs()[0] if test_org.get_orgs() else {"composition": {}, "workflow": {}}
        test_comp = test_data.get("composition", {})
        test_order = topological_sort(test_data.get("workflow", {}))
        logger.info(f"[EvoMAC] Test workflow: {test_order}")

        has_bug_any = False
        all_reports = ""
        for idx, phase in enumerate(test_order):
            if ctr[0] >= self.max_llm_calls:
                break
            test_file = f"test_requirement_{idx}.py"
            subtask = test_comp.get(phase, phase)
            resp, exceeded = await self._call_in_span(
                self.test_writer,
                TESTCODECOMPLETE.format(task=task, codes=codes.get_codes(), subtask=subtask, test_file_name=test_file),
                ctr,
                f"test_writer_{test_file}",
            )
            test_codes.update(resp, target_file=test_file)
            if exceeded:
                break
            has_bug, result = execute_tests(codes.codebooks, test_codes.codebooks, test_file)
            logger.info(f"[EvoMAC] {test_file}: has_bug={has_bug}, result={result[:150]!r}")
            if has_bug:
                has_bug_any = True
                all_reports += f"Requirement: {subtask}\nTest Report: {result}\n\n"

        if not has_bug_any:
            all_reports = "No bugs found in the test cases."
        return has_bug_any, all_reports


# ── Build orchestrator ──────────────────────────────────────
def build_orchestrator(
    ss: InMemorySessionService,
    model_name: str,
    base_url: str,
    api_key: str,
    iteration: int = 5,
    language: str = "python",
    max_llm_calls: int = 100,
) -> EvoMACOrchestrator:
    mk = lambda: make_model(model_name, base_url, api_key)

    initial_coder = LlmAgent(
        name="initial_coder",
        model=mk(),
        instruction=INITIAL_CODING_ROLE,
        output_key="response",
    )
    cto = LlmAgent(
        name="cto",
        model=mk(),
        instruction="You are the Chief Technology Officer at EvoMAC. You organize coding teams and workflows.",
        output_key="response",
    )
    programmer = LlmAgent(
        name="programmer",
        model=mk(),
        instruction="You are a Programmer at EvoMAC. You write and fix code based on subtask descriptions. You can use run_python_code tool to test your code.",
        output_key="response",
        tools=[run_python_code],
    )
    test_writer = LlmAgent(
        name="test_writer",
        model=mk(),
        instruction="You are a Programmer at EvoMAC. You write test cases for code validation.",
        output_key="response",
    )

    return EvoMACOrchestrator(
        name="orchestrator",
        initial_coder=initial_coder,
        cto=cto,
        programmer=programmer,
        test_writer=test_writer,
        ss=ss,
        max_llm_calls=max_llm_calls,
        iteration=iteration,
        language=language,
        sub_agents=[initial_coder, cto, programmer, test_writer],
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


# ── Main entry ──────────
async def main():
    p = argparse.ArgumentParser(description="EvoMAC Agent")
    p.add_argument("--task_id", default="default")
    p.add_argument("--query", default="")
    p.add_argument("--max_llm_calls", type=int, default=100)
    p.add_argument("--iteration", type=int, default=5, help="Max test-and-fix iterations")
    p.add_argument("--language", type=str, default="python", help="Target programming language")
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
                "iteration": args.iteration,
                "language": args.language,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fh = logging.FileHandler(os.path.join(task_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    tp, _ = setup_tracing("EvoMAC", logger=logger, trace_dir=trace_dir)
    fault_engine = setup_fault_injection(args.fault_name)

    ss = InMemorySessionService()
    orchestrator = build_orchestrator(
        ss=ss,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        iteration=args.iteration,
        language=args.language,
        max_llm_calls=args.max_llm_calls,
    )
    runner = Runner(agent=orchestrator, app_name=APP, session_service=ss)

    logger.info(f"[main] task={args.task_id}, model={model}, iteration={args.iteration}, language={args.language}")
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
        logger.info(f"[main] OK, task={args.task_id}, code_len={len(resp)}, preview={resp[:200]!r}")
    else:
        logger.warning(f"[main] EMPTY response for task={args.task_id}")

    if tp:
        tp.force_flush()


if __name__ == "__main__":
    asyncio.run(main())
