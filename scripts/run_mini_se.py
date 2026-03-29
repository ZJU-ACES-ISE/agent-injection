# run_mini_se.py
import asyncio
import json
import os
import subprocess
import logging
import argparse
from pathlib import Path
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=__import__("sys").stdout)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger("MiniSE")
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

APP = "MiniSE"
SNIPPET_LINES = 4
TERM_MSG = "TERMINATE"

# System prompt for Mini-SE agent
SYSTEM_PROMPT = """You are an expert software engineer. Fix the GitHub issue using the provided tools.

Tools:
1. **grep_search**(pattern): Search codebase. Supports multiple patterns separated by | (e.g. "confirmByUid|confirmByCode|validateEmail").
2. **view_files**(paths): View MULTIPLE files at once. Comma-separated paths (e.g. "src/a.py,src/b.js").
3. **edit_tool**(path, old_str, new_str): Find-and-replace. old_str must be exact unique match.
4. **patch_submission**(): Generate diff patch when done.

Workflow:
1. Read the issue. Extract SPECIFIC keywords (function names, error messages, variable names).
2. ONE grep_search with multiple patterns joined by | (e.g. "funcA|funcB|ClassName"). This finds all relevant files at once.
3. ONE view_files call with ALL relevant files from grep results.
4. edit_tool to make minimal fixes.
5. patch_submission, then say TERMINATE.

IMPORTANT: Minimize tool calls. Combine multiple searches into ONE grep with | patterns. Read ALL relevant files in ONE view_files call.

Rules:
- Search for SPECIFIC terms from the issue, not generic words.
- old_str must match file content EXACTLY (whitespace, indentation matter).
- Do NOT create new files or add tests.
- Say TERMINATE when done.
"""


# Workspace manager
class WorkspaceContext:
    """Manages a git workspace for a single SWE-bench task."""

    def __init__(self, repo_dir: str, temp_dir: str, instance_id: str, base_commit: str, model_name: str = ""):
        self.instance_id = instance_id
        self.base_commit = base_commit
        self.repo_dir = repo_dir
        safe_id = instance_id.replace("/", "__")
        safe_model = (model_name or "default").replace("/", "_")
        self.path = os.path.join(temp_dir, safe_model, f"ws_{safe_id}")

    def setup(self):
        """Create workspace from repo at base_commit. Reuses if already built."""
        self.repo_dir = os.path.abspath(self.repo_dir)
        self.path = os.path.abspath(self.path)

        # Reuse existing workspace if it has the right commit marker
        marker = os.path.join(self.path, ".mini_se_commit")
        if os.path.exists(marker):
            with open(marker) as f:
                if f.read().strip() == self.base_commit:
                    logger.info(f"Reusing workspace: {self.path}")
                    subprocess.run(["git", "checkout", "."], cwd=self.path, capture_output=True)
                    return

        if os.path.exists(self.path):
            subprocess.run(["rm", "-rf", self.path], check=True)
        os.makedirs(self.path, exist_ok=True)

        # Ensure the commit is available locally
        r = subprocess.run(
            ["git", "-C", self.repo_dir, "cat-file", "-t", self.base_commit],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.info(f"Fetching commit {self.base_commit[:12]} for {self.instance_id}...")
            subprocess.run(
                ["git", "-C", self.repo_dir, "fetch", "--quiet", "--depth=1", "origin", self.base_commit],
                capture_output=True,
                text=True,
            )
            r2 = subprocess.run(
                ["git", "-C", self.repo_dir, "cat-file", "-t", self.base_commit],
                capture_output=True,
                text=True,
            )
            if r2.returncode != 0:
                logger.info(f"Depth fetch failed, trying full fetch...")
                subprocess.run(
                    ["git", "-C", self.repo_dir, "fetch", "--quiet", "--unshallow"],
                    capture_output=True,
                    text=True,
                )

        # Archive + extract at base_commit
        archive = os.path.join(self.path, "_archive.tar")
        subprocess.run(
            ["git", "-C", self.repo_dir, "archive", "--format=tar", f"--output={archive}", self.base_commit],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(["tar", "-xf", archive, "-C", self.path], check=True, capture_output=True)
        os.remove(archive)

        # Init git for diff tracking (set local user to avoid 'Please tell me who you are' on fresh Linux)
        subprocess.run(
            f"cd {self.path} && git init && git config user.email 'mini-se@local' && git config user.name 'mini-se' && git add . && git commit -m 'init' --quiet",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

        # Write commit marker for reuse detection
        with open(os.path.join(self.path, ".mini_se_commit"), "w") as f:
            f.write(self.base_commit)

        logger.info(f"Workspace ready: {self.path}")

    def get_diff(self) -> str:
        """Get git diff of all changes."""
        try:
            r = subprocess.run(["git", "diff", "HEAD"], cwd=self.path, capture_output=True, text=True)
            patch = r.stdout
            if not patch.strip():
                return "The patch is empty. No edits have been made yet. Use edit_tool to make changes first."
            return f"Patch generated successfully:\n```diff\n{patch}\n```"
        except Exception as e:
            return f"Error generating patch: {e}"


# Tool globals (set per-task before agent run)
_active_workspace: WorkspaceContext = None


# ADK Tool: grep_search
def grep_search(pattern: str) -> dict:
    """Search the codebase for a text pattern using grep.

    Args:
        pattern: Text or regex pattern to search for.

    Returns:
        Dict with 'result' containing matching file:line:content, or 'error'.
    """
    global _active_workspace
    if _active_workspace is None:
        return {"error": "No active workspace."}
    if not pattern:
        return {"error": "pattern is required."}
    logger.info(f"[grep_search] pattern='{pattern}'")
    try:
        r = subprocess.run(
            [
                "grep",
                "-rn",
                "--include=*.py",
                "--include=*.js",
                "--include=*.ts",
                "--include=*.jsx",
                "--include=*.tsx",
                "--include=*.java",
                "--include=*.go",
                "--include=*.rb",
                "--include=*.rs",
                "--include=*.c",
                "--include=*.cpp",
                "--include=*.h",
                "--include=*.json",
                "--include=*.yaml",
                "--include=*.yml",
                "--include=*.tpl",
                "--include=*.html",
                "--include=*.ejs",
                "--include=*.hbs",
                "--include=*.pug",
                "--include=*.jade",
                "--include=*.vue",
                "--include=*.svelte",
                "-l" if len(pattern) <= 3 else "-n",  # short patterns: list files only
                "--exclude-dir=node_modules",
                "--exclude-dir=.git",
                "--exclude-dir=vendor",
                "--exclude-dir=dist",
                "--exclude-dir=build",
                "--exclude-dir=public",
                "--exclude-dir=static",
                "--exclude-dir=assets",
                "--exclude-dir=locale",
                "--exclude-dir=locales",
                "--exclude-dir=language",
                "--exclude-dir=languages",
                "--exclude-dir=i18n",
                "--exclude-dir=l10n",
                "--exclude-dir=translations",
                "--exclude-dir=test",
                "--exclude-dir=tests",
                "--exclude-dir=__tests__",
                "--exclude-dir=spec",
                "--exclude-dir=__pycache__",
                "--exclude-dir=.tox",
                "--exclude-dir=.mypy_cache",
                "--exclude-dir=.pytest_cache",
                "--exclude-dir=coverage",
                "--exclude-dir=.coverage",
                "--exclude-dir=htmlcov",
                "--exclude-dir=docs",
                "--exclude-dir=doc",
                "--exclude-dir=.next",
                "--exclude-dir=.nuxt",
                "--exclude-dir=.output",
                "--exclude-dir=target",
                "--exclude-dir=bin",
                "--exclude-dir=obj",
                "--exclude-dir=logs",
                "--exclude-dir=tmp",
                "--exclude-dir=temp",
                "--exclude-dir=fixtures",
                "--exclude-dir=migrations",
                "--exclude-dir=.eggs",
                "--exclude-dir=.venv",
                "--exclude-dir=venv",
                "--exclude-dir=env",
                "--exclude=*.min.js",
                "--exclude=*.min.css",
                "--exclude=*.bundle.js",
                "--exclude=*.chunk.js",
                "--exclude=*.map",
                "--exclude=*.svg",
                "--exclude=*.png",
                "--exclude=*.ico",
                "--exclude=*.lock",
                "--exclude=package-lock.json",
                "--exclude=yarn.lock",
                "--exclude=pnpm-lock.yaml",
                "--exclude=Gemfile.lock",
                "--exclude=poetry.lock",
                "--exclude=composer.lock",
                "--exclude=go.sum",
                "--exclude=Cargo.lock",
                "-E",
                pattern,
                ".",
            ],
            cwd=_active_workspace.path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = r.stdout.strip()
        if not output:
            logger.info(f"[grep] '{pattern}' -> 0 results")
            return {"result": f"No matches for '{pattern}'. Try a different search term."}
        lines = output.split("\n")
        logger.info(f"[grep] '{pattern}' -> {len(lines)} matches, result={output[:10000]}")
        return {"result": "\n".join(lines)}
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out (30s). Use a more specific pattern."}
    except Exception as e:
        return {"error": str(e)}


# ADK Tool: view_files
def view_files(paths: str) -> dict:
    """View multiple files at once. Pass comma-separated relative paths.

    Args:
        paths: Comma-separated file paths (e.g. "src/a.py,src/b.js,lib/c.ts").

    Returns:
        Dict with 'result' containing all file contents concatenated, or 'error'.
    """
    global _active_workspace
    if _active_workspace is None:
        return {"error": "No active workspace."}
    if not paths:
        return {"error": "paths is required."}
    file_list = [p.strip() for p in paths.split(",") if p.strip()]
    if not file_list:
        return {"error": "No valid paths provided."}
    logger.info(f"[view_files] {len(file_list)} files: {file_list}")
    results = []
    for p in file_list:
        full_path = Path(_active_workspace.path) / p
        if not full_path.exists():
            results.append(f"=== {p} === FILE NOT FOUND")
            continue
        if not full_path.is_file():
            results.append(f"=== {p} === NOT A FILE")
            continue
        try:
            content = full_path.read_text(errors="replace")
            lines = content.splitlines()
            numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
            results.append(f"=== {p} ({len(lines)} lines) ===\n" + "\n".join(numbered))
        except Exception as e:
            results.append(f"=== {p} === ERROR: {e}")
    return {"result": "\n\n".join(results)}


# ADK Tool: edit_tool
def edit_tool(path: str, old_str: str, new_str: str) -> dict:
    """Replace an exact string in a file with a new string.

    Args:
        path: Relative file path (e.g., "src/module/file.py").
        old_str: The exact text to find and replace. Must appear exactly once in the file.
        new_str: The replacement text. Use empty string to delete the old text.

    Returns:
        Dict with 'result' showing the edit outcome, or 'error' on failure.
    """
    global _active_workspace
    if _active_workspace is None:
        return {"error": "No active workspace."}

    if not path:
        return {"error": "`path` is required."}
    if old_str is None:
        return {"error": "`old_str` is required."}
    if new_str is None:
        new_str = ""

    ws_path = _active_workspace.path
    full_path = Path(ws_path) / path
    if not full_path.exists():
        return {"error": f"File '{path}' does not exist."}
    if not full_path.is_file():
        return {"error": f"'{path}' is a directory, not a file."}

    content = full_path.read_text(errors="replace")
    old_str = old_str
    new_str = new_str

    logger.info(f"[edit_tool] path='{path}' old_str='{old_str[:100]}' new_str='{new_str[:100]}'")
    count = content.count(old_str)
    if count == 0:
        return {"error": f"`old_str` not found verbatim in {path}. Check whitespace and special chars."}
    if count > 1:
        lines_with = [i + 1 for i, line in enumerate(content.split("\n")) if old_str in line]
        return {
            "error": f"Multiple occurrences ({count}) of `old_str` in {path} at lines {lines_with}. Make it more specific."
        }

    new_content = content.replace(old_str, new_str)

    # Syntax check for Python files
    if path.endswith(".py"):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=ws_path, delete=True) as tmp:
            tmp.write(new_content)
            tmp.flush()
            r = subprocess.run(
                f"python3 -c \"import ast; ast.parse(open('{tmp.name}').read())\"",
                shell=True,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                return {"error": f"Edit would introduce syntax errors:\n{r.stderr}\nEdit aborted."}

    full_path.write_text(new_content)

    # Build snippet for confirmation
    rep_start = content.split(old_str)[0].count("\n") + 1
    num_new = new_str.count("\n") + 1
    mod_lines = new_content.split("\n")
    snip_start = max(0, rep_start - 1 - SNIPPET_LINES)
    snip_end = min(len(mod_lines), rep_start - 1 + num_new + SNIPPET_LINES)
    snippet = "\n".join(mod_lines[snip_start:snip_end])

    return {
        "result": f"Edit applied to {path}:{rep_start}-{rep_start + num_new - 1}\n{snippet}\n\nReview and edit again if needed."
    }


# ADK Tool: patch_submission
def patch_submission() -> dict:
    """Generate a diff patch from all edits made so far and submit it.

    Returns:
        Dict with 'result' containing the generated patch diff.
    """
    global _active_workspace
    if _active_workspace is None:
        return {"error": "No active workspace."}
    logger.info("[patch_submission] generating diff")
    diff = _active_workspace.get_diff()
    logger.info(f"[patch_submission] result: {diff[:10000]}")
    return {"result": diff}


# Model factory (same pattern as run_autogen.py)
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


# Mini-SE Orchestrator (follows AutoGen pattern: _call + _call_in_span + OTel traces)
class MiniSEOrchestrator(BaseAgent):
    """Single-agent orchestrator with OTel tracing, matching AutoGen pattern."""

    agent: LlmAgent
    max_llm_calls: int = 30
    ss: InMemorySessionService

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, ss, max_llm_calls=30, model_name=None, base_url=None, api_key=None):
        _model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
        _base = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        _key = api_key or os.getenv("OPENAI_API_KEY", "sk-xxx")
        agent = LlmAgent(
            name="mini_se",
            model=make_model(_model, _base, _key),
            instruction=SYSTEM_PROMPT,
            tools=[grep_search, view_files, edit_tool, patch_submission],
        )
        super().__init__(
            name="mini_se_orchestrator",
            agent=agent,
            max_llm_calls=max_llm_calls,
            ss=ss,
            sub_agents=[agent],
        )

        # ── Build prompt from history ──

    def _build_prompt(self, prompt: str, history: List[Dict]) -> str:
        if not history:
            return prompt
        lines = [f"[{m['role']}]: {m['content']}" for m in history]
        return (
            "<conversation_history>\n"
            + "\n".join(lines)
            + "\n</conversation_history>\n\n<current_message>\n"
            + prompt
            + "\n</current_message>"
        )

    # ── Summarize history when context is too long ──
    async def _summarize_history(self, history: List[Dict], ctr: List[int]) -> List[Dict]:
        """Compress history into a single summary message using LLM (traced + counted)."""
        logger.info(f"[summarize] compressing {len(history)} messages...")
        lines = [f"[{m['role']}]: {m['content']}" for m in history]
        summary_prompt = (
            "Summarize the following conversation history concisely. "
            "Keep all file paths, function names, code changes, and key findings. "
            "Drop verbose tool outputs and repeated searches.\n\n" + "\n".join(lines)
        )
        tracer = otel_trace.get_tracer("mini_se_orchestrator")
        with tracer.start_as_current_span(
            "summarize_history",
            attributes={"orchestrator.history_len": len(history)},
        ):
            session = await self.ss.create_session(app_name=APP, user_id="u")
            runner = Runner(agent=self.agent, app_name=APP, session_service=self.ss)
            summary = ""
            async for ev in runner.run_async(
                user_id="u",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=summary_prompt)]),
            ):
                if ev.content and ev.content.parts and ev.author == self.agent.name:
                    t = (ev.content.parts[0].text or "").strip()
                    if t:
                        summary = t
            if ctr is not None:
                ctr[0] += 1
                logger.info(f"[{self.agent.name}] LLM #{ctr[0]}/{self.max_llm_calls} (summarize)")
        logger.info(f"[summarize] compressed to {len(summary)} chars: {summary[:500]}")
        return [{"role": "assistant", "content": f"[Previous conversation summary]: {summary}"}]

    # ── Core _call (same pattern as AutoGen, with context overflow retry) ──
    async def _call(
        self,
        agent: LlmAgent,
        prompt: str,
        history: List[Dict],
        ctr: List[int],
    ) -> Tuple[str, bool]:
        """Call agent via ADK Runner. Returns (response_text, exceeded)."""
        full_prompt = self._build_prompt(prompt, history)
        logger.info(f"[_call] input: {full_prompt[:10000]}")

        for attempt in range(2):  # attempt 0 = normal, attempt 1 = after summarize
            try:
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
                logger.info(f"[_call] output: exceeded={exceeded} resp={resp[:10000]}")
                return resp, exceeded

            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and (
                    "context_length" in err_str
                    or "too many tokens" in err_str
                    or "maximum context" in err_str
                    or "token" in err_str
                    and "limit" in err_str
                ):
                    logger.warning(f"[_call] context overflow, summarizing history and retrying: {e}")
                    history[:] = await self._summarize_history(history, ctr)
                    full_prompt = self._build_prompt(prompt, history)
                    continue
                logger.error(f"[_call] error: {e}")
                raise

    async def _call_in_span(
        self,
        agent: LlmAgent,
        prompt: str,
        history: List[Dict],
        ctr: List[int],
        span_name: str,
    ) -> Tuple[str, bool]:
        """Wrap _call in OTel span for trace export."""
        tracer = otel_trace.get_tracer("mini_se_orchestrator")
        with tracer.start_as_current_span(
            span_name,
            attributes={"orchestrator.agent": agent.name, "orchestrator.llm_calls_before": ctr[0]},
        ) as span:
            result, exceeded = await self._call(agent, prompt, history, ctr)
            span.set_attribute("orchestrator.llm_calls_after", ctr[0])
            span.set_attribute("orchestrator.exceeded", exceeded)
            return result, exceeded

    def _evt(self, ctx, author, text):
        return Event(
            invocation_id=ctx.invocation_id,
            author=author,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
        )

    # ── Main orchestration loop ──
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        query = ""
        for ev in reversed(ctx.session.events):
            if ev.content and ev.content.role == "user" and ev.content.parts:
                query = ev.content.parts[0].text or ""
                if query:
                    break
        if not query:
            yield self._evt(ctx, "mini_se_orchestrator", "No query.")
            return

        history: List[Dict] = []
        ctr = [0]
        call_idx = 0

        # Turn 0: initial query
        resp, exceeded = await self._call_in_span(self.agent, query, history, ctr, f"mini_se_call_{call_idx}")
        call_idx += 1
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": resp})
        yield self._evt(ctx, "mini_se", resp)
        logger.info(f"[Turn 0] calls={ctr[0]} TERM={TERM_MSG in resp} resp={resp[:10000]}")

        if TERM_MSG in resp or exceeded:
            ctx.session.state["final"] = resp.replace(TERM_MSG, "").strip()
            return

        # Subsequent turns: feed response back as next prompt
        for turn in range(1, self.max_llm_calls):
            if ctr[0] >= self.max_llm_calls:
                break

            resp, exceeded = await self._call_in_span(self.agent, resp, history, ctr, f"mini_se_call_{call_idx}")
            call_idx += 1
            history.append({"role": "user", "content": resp})
            history.append({"role": "assistant", "content": resp})
            yield self._evt(ctx, "mini_se", resp)
            logger.info(f"[Turn {turn}] calls={ctr[0]} TERM={TERM_MSG in resp} resp={resp[:10000]}")

            if TERM_MSG in resp or exceeded:
                ctx.session.state["final"] = resp.replace(TERM_MSG, "").strip()
                return

        ctx.session.state["final"] = resp.replace(TERM_MSG, "").strip()


# Main
async def main():
    p = argparse.ArgumentParser(description="Mini-SE agent for SWE-bench Pro")
    p.add_argument("--task_id", required=True, help="SWE-bench instance ID")
    p.add_argument("--query", required=True, help="Issue description")
    p.add_argument("--repo_dir", required=True, help="Path to cloned repo")
    p.add_argument("--base_commit", required=True, help="Base commit SHA")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--max_llm_calls", type=int, default=100)
    p.add_argument("--model", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument(
        "--workspaces_dir", default=None, help="Shared workspaces directory (default: output_dir/_workspaces)"
    )
    p.add_argument("--fault_name", default=None, help="Fault name for injection. None=no injection.")
    args = p.parse_args()

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "sk-xxx")

    task_dir = os.path.join(args.output_dir, args.task_id)
    trace_dir = os.path.join(task_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)

    # Save input (same format as run_autogen.py)
    with open(os.path.join(task_dir, "input.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": args.task_id,
                "query": args.query,
                "model": model,
                "max_llm_calls": args.max_llm_calls,
                "repo_dir": args.repo_dir,
                "base_commit": args.base_commit,
                "fault_name": args.fault_name,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fh = logging.FileHandler(os.path.join(task_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    tp, _ = setup_tracing("MiniSE", logger=logger, trace_dir=trace_dir)
    fault_engine = setup_fault_injection(args.fault_name)

    # Setup workspace
    temp_dir = args.workspaces_dir or os.path.join(args.output_dir, "_workspaces")
    os.makedirs(temp_dir, exist_ok=True)

    global _active_workspace
    ws = WorkspaceContext(args.repo_dir, temp_dir, args.task_id, args.base_commit, model)
    ws.setup()
    _active_workspace = ws

    try:
        query = (
            f"Please resolve the following GitHub issue by searching the codebase, "
            f"understanding the root cause, making targeted edits, and submitting a patch.\n\n"
            f"<issue>\n{args.query}\n</issue>"
        )

        ss = InMemorySessionService()
        runner = Runner(
            agent=MiniSEOrchestrator(
                ss=ss,
                max_llm_calls=args.max_llm_calls,
                model_name=model,
                base_url=base_url,
                api_key=api_key,
            ),
            app_name=APP,
            session_service=ss,
        )

        logger.info(
            f"Task: [{args.task_id}] model={model} max_llm_calls={args.max_llm_calls} fault={args.fault_name or 'none'}"
        )

        session = await ss.create_session(app_name=APP, user_id="user")
        events_log = []

        async for ev in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        ):
            if ev.content and ev.content.parts:
                text = (ev.content.parts[0].text or "").strip()
                if text:
                    logger.info(f"[{args.task_id}][{(ev.author or '?').upper()}] {text[:10000]}")
                    events_log.append({"author": ev.author or "?", "text": text})

        teardown_fault_injection(fault_engine)

        # Get final answer from session state
        s = await ss.get_session(app_name=APP, user_id="user", session_id=session.id)
        final = (s.state or {}).get("final", "")
        if not final:
            for e in reversed(events_log):
                if e["author"] == "mini_se":
                    final = e["text"].replace(TERM_MSG, "").strip()
                    if final:
                        break

        # Get final patch
        patch = ""
        try:
            r = subprocess.run(["git", "diff", "HEAD"], cwd=ws.path, capture_output=True, text=True)
            patch = r.stdout
        except Exception as e:
            logger.error(f"Failed to get diff: {e}")

        # Save outputs (same structure as run_autogen.py)
        output_data = {
            "task_id": args.task_id,
            "final_answer": final,
            "patch": patch,
            "events": events_log,
        }
        if fault_engine is not None:
            output_data["_fault_name"] = args.fault_name
            output_data["_fault_fired"] = len(fault_engine.log)
            output_data["_fault_log"] = fault_engine.log

        with open(os.path.join(task_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"[DONE] {args.task_id} patch_len={len(patch)}")

        if tp:
            tp.force_flush()

    finally:
        _active_workspace = None
        # Don't cleanup — workspace and DB are reusable across reruns


if __name__ == "__main__":
    asyncio.run(main())
