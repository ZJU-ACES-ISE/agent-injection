"""
Fault detection: rule-based + LLM-as-a-Judge (All-at-once).
Predicts fault type and fault step for each fault-injected failed task.
Metrics: Fault Type Accuracy, Fault Step Accuracy, Cohen's Kappa.

Judge models loaded via MODEL_PREFIXES env vars (same as run_all_method_dataset.py):
    {PREFIX}_OPENAI_MODEL, {PREFIX}_OPENAI_BASE_URL, {PREFIX}_OPENAI_API_KEY, {PREFIX}_MAX_WORKERS

Results saved per-task:
    {task_dir}/fault_detected/rule.json                       (rule-based detection)
    {task_dir}/fault_detected/all-at-once-{model_name}.json   (LLM-as-a-Judge, one per judge)

Usage:
    uv run python scripts/fault_detect.py                          # rule-based only
    uv run python scripts/fault_detect.py --method llm             # LLM judge (all configured judges)
    uv run python scripts/fault_detect.py --method llm --judges DEEPSEEK GPT
    uv run python scripts/fault_detect.py --method both            # rule + LLM
    uv run python scripts/fault_detect.py --metrics-only           # just print metrics from saved results
"""

import os
import sys
import re
import json
import csv
import argparse
import logging
import asyncio
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import multiprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fault_detect")

# ==============================================================================
# Constants
# ==============================================================================
ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "results_all" / "results_fault"
EVAL_CSV = ROOT / "results_paper" / "eval_detail.csv"

FAULT_TYPES = ["error", "timeout", "empty", "truncate", "corrupt", "schema", "compound", "unknown"]

# Judge model env-var prefixes (same pattern as run_all_method_dataset.py)
MODEL_PREFIXES = ["DEEPSEEK", "GPT", "SEED", "KIMI", "CLAUDE"]


# ==============================================================================
# Judge model builder (reads {PREFIX}_OPENAI_MODEL/BASE_URL/API_KEY from env)
# ==============================================================================
def _build_judge_models(prefixes: list) -> list:
    judges = []
    for p in prefixes:
        name = os.getenv(f"{p}_OPENAI_MODEL")
        if not name:
            logger.info(f"[judge] skip {p}: {p}_OPENAI_MODEL not set")
            continue
        judges.append(
            {
                "prefix": p,
                "name": name,
                "base_url": os.getenv(f"{p}_OPENAI_BASE_URL"),
                "api_key": os.getenv(f"{p}_OPENAI_API_KEY"),
                "max_workers": int(os.getenv(f"{p}_MAX_WORKERS", "5")),
            }
        )
    logger.info(f"[judge] {len(judges)} judges: {[(j['prefix'], j['name']) for j in judges]}")
    return judges


# ==============================================================================
# Collect failed fault-injected cases from eval_detail.csv
# ==============================================================================
def collect_failed_cases(datasets=None) -> list:
    """Read eval_detail.csv, return cases where fault_type=results_fault, fired>0, eval_score=0.
    If datasets is given, only include those datasets."""
    cases = []
    with open(EVAL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Only fault-injected experiment rows
            if row.get("fault_type") != "results_fault":
                continue
            # Filter by dataset if specified
            if datasets and row.get("dataset") not in datasets:
                continue
            # Only tasks where fault actually fired
            fired = float(row.get("_fault_fired") or 0)
            if fired <= 0:
                continue
            # Only failed tasks (eval_score = 0)
            score = row.get("eval_score", "1")
            if score not in ("0", "0.0", ""):
                continue
            # Build task_dir path: results_all/results_fault/{dataset}/{model}/{method}/{task_id}
            task_dir = BASE_DIR / row["dataset"] / row["model"] / row["method"] / row["task_id"]
            if not (task_dir / "output.json").is_file():
                continue
            cases.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "system": row["method"],
                    "task_id": row["task_id"],
                    "task_dir": str(task_dir),
                    "fault_name": row.get("_fault_name", ""),
                }
            )
    logger.info(f"[collect] {len(cases)} failed fault-injected cases from {EVAL_CSV}")
    return cases


# ==============================================================================
# Ground-truth extraction
# ==============================================================================
def parse_fault_type(fault_name: str) -> str:
    """Extract canonical fault type: 'llm_error_single' -> 'error', 'compound_xxx' -> 'compound'."""
    if not fault_name:
        return ""
    if fault_name.startswith("compound_"):
        return "compound"
    # llm_error_single / tool_timeout_burst / llm_schema_pos_early -> 2nd part
    parts = fault_name.split("_")
    return parts[1] if len(parts) >= 2 else fault_name


def find_fault_step(fault_log: list, steps: list) -> int:
    """Map first fault firing timestamp to step number (1-based). Returns 0 if mapping fails."""
    if not fault_log or not steps:
        return 0
    # fault_log[0]["t"] is Unix seconds; steps have start_time/end_time in nanoseconds
    t_ns = int(fault_log[0]["t"] * 1e9)
    for s in steps:
        try:
            if int(s["start_time"]) <= t_ns <= int(s["end_time"]):
                return s["step"]
        except (KeyError, ValueError, TypeError):
            continue
    # Fallback: fault likely fired before/during first step
    return steps[0].get("step", 1) if steps else 0


# ==============================================================================
# Build step texts for detection (from agent_steps.json or output.json events)
# ==============================================================================
def build_step_texts(output_data: dict, steps: list) -> list:
    """Return [{step, agent_name, text}, ...]. Prefer agent_steps; fall back to output.json events."""
    if steps:
        return [
            {
                "step": s["step"],
                "agent_name": s.get("agent_name", ""),
                "text": s.get("agent", {}).get("output", "") or "",
            }
            for s in steps
        ]
    if output_data.get("events"):
        return [
            {"step": i, "agent_name": ev.get("author", ""), "text": ev.get("text", "")}
            for i, ev in enumerate(output_data["events"], 1)
        ]
    return []


# ==============================================================================
# Rule-based detection
# ==============================================================================
# Ordered patterns: first match wins
RULE_PATTERNS = [
    (re.compile(r"\[API ERROR\]", re.IGNORECASE), "error"),
    (re.compile(r"HTTP 500.*Internal Server Error", re.IGNORECASE), "error"),
    (re.compile(r"\[SERVICE UNAVAILABLE\]", re.IGNORECASE), "compound"),
    (re.compile(r"HTTP 503", re.IGNORECASE), "compound"),
    (re.compile(r"\[TIMEOUT\]", re.IGNORECASE), "timeout"),
    (re.compile(r"timed out.*server did not respond", re.IGNORECASE), "timeout"),
    (re.compile(r"Connection dropped.*server did not respond", re.IGNORECASE), "timeout"),
    (re.compile(r"\[CONTENT FILTERED\]", re.IGNORECASE), "compound"),
    (re.compile(r"content_policy_violation", re.IGNORECASE), "schema"),
    (re.compile(r"This response has been filtered", re.IGNORECASE), "schema"),
    (re.compile(r"<!DOCTYPE html>.*502 Bad Gateway", re.IGNORECASE | re.DOTALL), "compound"),
    (re.compile(r"502 Bad Gateway.*nginx", re.IGNORECASE), "compound"),
    # corrupt: injected unicode symbol block ☀☁☂☃★☆
    (re.compile(r"[\u2600-\u26FF]{3,}"), "corrupt"),
]
# Mojibake: UTF-8 bytes decoded as Latin-1, produces â followed by control chars
_MOJIBAKE_RE = re.compile(r"â[\x80-\xbf€‚ƒ„…†‡ˆ‰Š‹ŒŽ''" "•–—˜™š›œžŸ]")


def rule_detect(step_texts: list) -> dict:
    """Rule-based fault detection. Returns {fault_type, fault_step}."""
    # 1) Explicit pattern matches (error, timeout, schema, compound)
    for s in step_texts:
        for pattern, ftype in RULE_PATTERNS:
            if pattern.search(s["text"]):
                return {"fault_type": ftype, "fault_step": s["step"]}

    # 2) Mojibake corruption (â patterns, >=2 hits)
    for s in step_texts:
        if len(_MOJIBAKE_RE.findall(s["text"])) >= 2:
            return {"fault_type": "corrupt", "fault_step": s["step"]}

    # 3) Empty response
    for s in step_texts:
        if s["text"].strip() == "":
            return {"fault_type": "empty", "fault_step": s["step"]}

    # 4) Truncation: text ends mid-word without proper termination
    for s in step_texts:
        t = s["text"].strip()
        if len(t) < 20:
            continue
        if t[-1] not in ".!?`\"')]}>;\n" and t[-1].isalnum():
            last_line = t.split("\n")[-1].strip()
            if len(last_line) > 5 and not last_line.endswith(("TERMINATE", "exitcode")):
                return {"fault_type": "truncate", "fault_step": s["step"]}

    return {"fault_type": "unknown", "fault_step": 0}


# ==============================================================================
# LLM-as-a-Judge (All-at-once) — uses util.acall_llm + Pydantic schema
# ==============================================================================
class FaultDetectResult(BaseModel):
    fault_type: str = Field(
        description="Detected fault type: error, timeout, empty, truncate, corrupt, schema, compound, or unknown"
    )
    fault_step: int = Field(description="Step number (1-based) where the fault first manifests")
    reason: str = Field(description="Brief explanation of the diagnosis")


def make_llm_config(judge: dict):
    """Create a util.Config for a specific judge model. Concurrency from {PREFIX}_MAX_WORKERS."""
    from util import Config

    cfg = Config(
        openai_base_url=judge["base_url"],
        openai_api_key=judge["api_key"],
        openai_model=judge["name"],
        llm_max_concurrency=judge["max_workers"],
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg


# Cache for merged system context (settings + dependency per system group)
_merged_cache = {}


def load_system_context(case: dict) -> str:
    """Load merged_settings.json + merged_dependency.json for this system, format as text for LLM prompt."""
    key = (case["dataset"], case["model"], case["system"])
    if key in _merged_cache:
        return _merged_cache[key]

    sys_dir = BASE_DIR / case["dataset"] / case["model"] / case["system"]
    parts = []

    # Load agent prompts and tool definitions
    s_file = sys_dir / "merged_settings.json"
    if s_file.is_file():
        try:
            settings = json.loads(s_file.read_text(encoding="utf-8"))
            # Agent roles (name + first 200 chars of prompt)
            agents = settings.get("prompt", {})
            if agents:
                parts.append(
                    "Agents:\n"
                    + "\n".join(
                        (
                            f"  - {name}: {prompt[:200].replace(chr(10), ' ')}..."
                            if len(prompt) > 200
                            else f"  - {name}: {prompt.replace(chr(10), ' ')}"
                        )
                        for name, prompt in agents.items()
                    )
                )
            # Tool names + short descriptions
            tools = settings.get("tool", [])
            if tools:
                parts.append(
                    "Tools:\n" + "\n".join(f"  - {t['name']}: {t.get('description', '')[:100]}" for t in tools)
                )
        except Exception as e:
            logger.warning(f"Bad merged_settings {s_file}: {e}")

    # Load agent->tool/agent dependency graph
    d_file = sys_dir / "merged_dependency.json"
    if d_file.is_file():
        try:
            dep = json.loads(d_file.read_text(encoding="utf-8"))
            dep_lines = [
                f"  - {agent} uses tools={d.get('tool', [])}, delegates_to={d.get('agent', [])}"
                for agent, d in dep.items()
                if d.get("tool") or d.get("agent")
            ]
            if dep_lines:
                parts.append("Dependencies:\n" + "\n".join(dep_lines))
        except Exception as e:
            logger.warning(f"Bad merged_dependency {d_file}: {e}")

    result = "\n".join(parts)
    _merged_cache[key] = result
    return result


async def llm_detect(step_texts: list, task_id: str, config, system_context: str = "") -> dict:
    """LLM-as-a-Judge: feed system context + conversation to LLM, return structured result."""
    from util import acall_llm

    # Format conversation: each step as "agent_name: text" (truncate long texts)
    chat_lines = [f"Step {s['step']} [{s['agent_name']}]: {s['text'][:2000]}" for s in step_texts]
    chat_content = "\n".join(chat_lines)

    fault_types_str = ", ".join(FAULT_TYPES)
    ctx_block = f"\nSystem Architecture:\n{system_context}\n" if system_context else ""

    # Prompt aligned with WHO baseline all_at_once style
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history to diagnose injected faults. "
        "A fault was injected into the LLM API or tool calls during this conversation. "
        "Identify the fault type and the step where it first manifests.\n\n"
        f"The fault type is one of [{fault_types_str}]:\n"
        "  - error: API server error (HTTP 500, [API ERROR])\n"
        "  - timeout: Request timeout or connection drop ([TIMEOUT])\n"
        "  - empty: Empty or missing LLM response\n"
        "  - truncate: Response cut short / incomplete mid-sentence\n"
        "  - corrupt: Garbled text, mojibake, unicode symbols replacing normal text\n"
        "  - schema: Invalid schema / content policy violation / wrong JSON structure\n"
        "  - compound: Multiple or realistic compound faults (503, content filter, HTML error page, wrong entity, stale cache)\n"
        "  - unknown: Cannot determine\n\n"
        f"{ctx_block}"
        f"Here's the conversation ({len(step_texts)} steps total):\n\n{chat_content}\n\n"
        "Based on this conversation, predict the fault type and the step (1-based) where it first manifested.\n"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant skilled in analyzing multi-agent conversations and diagnosing faults.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        result_json = await acall_llm(messages, config, output_schema=FaultDetectResult)
        parsed = json.loads(result_json)
        pred_type = parsed.get("fault_type", "unknown").lower().strip()
        # Normalize to canonical types
        if pred_type not in FAULT_TYPES:
            pred_type = "unknown"
        return {"fault_type": pred_type, "fault_step": parsed.get("fault_step", 0), "reason": parsed.get("reason", "")}
    except Exception as e:
        logger.error(f"LLM failed for {task_id}: {e}")
        return {"fault_type": "unknown", "fault_step": 0, "reason": str(e)}


# ==============================================================================
# Per-task result I/O: fault_detected/rule.json and fault_detected/all-at-once.json
# ==============================================================================
def _result_path(task_dir: str, filename: str) -> Path:
    return Path(task_dir) / "fault_detected" / filename


def load_json(path: Path) -> dict:
    """Load a JSON file, return empty dict on missing/bad file."""
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Bad JSON {path}: {e}")
    return {}


def save_json(path: Path, data: dict):
    """Write JSON with auto-mkdir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ==============================================================================
# Metrics
# ==============================================================================
def compute_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's Kappa between two label lists."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    all_labels = set(labels_a) | set(labels_b)
    pe = sum((labels_a.count(l) / n) * (labels_b.count(l) / n) for l in all_labels)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def print_metrics(cases: list):
    """Compute and print metrics from saved fault_detected/ files."""
    # Collect all results: merge rule.json + all all-at-once-*.json per task
    results = []
    for c in cases:
        rule_data = load_json(_result_path(c["task_dir"], "rule.json"))
        # Merge all per-judge LLM files into one llm dict
        llm_all = {}
        fd_dir = Path(c["task_dir"]) / "fault_detected"
        if fd_dir.is_dir():
            for f in fd_dir.glob("all-at-once-*.json"):
                data = load_json(f)
                for jname, jresult in data.get("llm", {}).items():
                    llm_all[jname] = jresult
        merged = {**rule_data}
        if llm_all:
            merged["llm"] = llm_all
        if not merged:
            continue
        results.append(merged)
    logger.info(f"[metrics] {len(results)} results loaded from {len(cases)} cases")
    if not results:
        return

    # Rule metrics
    rule_results = [
        (r["gt_fault_type"], r["rule"]["fault_type"], str(r["gt_fault_step"]), str(r["rule"]["fault_step"]))
        for r in results
        if r.get("rule")
    ]
    if rule_results:
        gt_t, pred_t, gt_s, pred_s = zip(*rule_results)
        type_acc = sum(g == p for g, p in zip(gt_t, pred_t)) / len(gt_t)
        step_acc = sum(g == p for g, p in zip(gt_s, pred_s)) / len(gt_s)
        logger.info(f"[Rule] TypeAcc={type_acc:.4f} StepAcc={step_acc:.4f} ({len(rule_results)} cases)")
        # Per-type breakdown
        type_total, type_correct = Counter(), Counter()
        for g, p in zip(gt_t, pred_t):
            type_total[g] += 1
            if g == p:
                type_correct[g] += 1
        for t in sorted(type_total):
            logger.info(f"  [Rule] {t}: {type_correct[t]}/{type_total[t]}={type_correct[t]/type_total[t]:.4f}")

    # LLM metrics per judge
    judge_names = set()
    for r in results:
        for k in r.get("llm", {}):
            judge_names.add(k)

    for jname in sorted(judge_names):
        llm_results = [
            (
                r["gt_fault_type"],
                r["llm"][jname]["fault_type"],
                str(r["gt_fault_step"]),
                str(r["llm"][jname]["fault_step"]),
            )
            for r in results
            if r.get("llm", {}).get(jname)
        ]
        if not llm_results:
            continue
        gt_t, pred_t, gt_s, pred_s = zip(*llm_results)
        type_acc = sum(g == p for g, p in zip(gt_t, pred_t)) / len(gt_t)
        step_acc = sum(g == p for g, p in zip(gt_s, pred_s)) / len(gt_s)
        logger.info(f"[LLM|{jname}] TypeAcc={type_acc:.4f} StepAcc={step_acc:.4f} ({len(llm_results)} cases)")

        # Kappa with rule (if both exist)
        shared = [
            (r["rule"]["fault_type"], r["llm"][jname]["fault_type"])
            for r in results
            if r.get("rule") and r.get("llm", {}).get(jname)
        ]
        if shared:
            rule_l, llm_l = zip(*shared)
            logger.info(
                f"[Kappa|{jname}] TypeKappa={compute_kappa(list(rule_l), list(llm_l)):.4f} ({len(shared)} shared)"
            )


# ==============================================================================
# LLM judge worker: one process per judge model (runs asyncio internally)
# ==============================================================================
def _run_judge_worker(judge: dict, cases: list):
    """Process entry point: run one judge model on all cases. Each judge saves to its own file."""
    asyncio.run(_run_judge_async(judge, cases))


async def _run_judge_async(judge: dict, cases: list):
    """Async loop for one judge: iterate cases, call LLM, save per-task."""
    jname = judge["name"]
    llm_config = make_llm_config(judge)

    # Each judge writes to its own file: all-at-once-{model_name}.json (no file conflict)
    filename = f"all-at-once-{jname}.json"

    # Skip cases that already have this judge's result
    pending = [c for c in cases if not load_json(_result_path(c["task_dir"], filename)).get("llm", {}).get(jname)]
    logger.info(f"[LLM|{jname}] {len(cases)-len(pending)} done, {len(pending)} pending")

    for i, case in enumerate(pending):
        task_dir = Path(case["task_dir"])
        # Load output.json
        try:
            output_data = json.loads((task_dir / "output.json").read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"[LLM|{jname}] load failed {case['task_id']}: {e}")
            continue
        # Load agent_steps.json
        steps = []
        steps_file = task_dir / "trace_parsed" / "agent_steps.json"
        if steps_file.is_file():
            try:
                steps = json.loads(steps_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"[LLM|{jname}] load steps failed {case['task_id']}: {e}")

        step_texts = build_step_texts(output_data, steps)
        if not step_texts:
            continue

        # Ground truth
        gt_type = parse_fault_type(case["fault_name"])
        gt_step = find_fault_step(output_data.get("_fault_log", []), steps)

        # System architecture context for LLM prompt
        system_context = load_system_context(case)

        # Call LLM judge
        llm_result = await llm_detect(step_texts, case["task_id"], llm_config, system_context)

        # Save incrementally to per-judge file
        save_json(
            _result_path(case["task_dir"], filename),
            {
                "dataset": case["dataset"],
                "model": case["model"],
                "system": case["system"],
                "task_id": case["task_id"],
                "fault_name": case["fault_name"],
                "gt_fault_type": gt_type,
                "gt_fault_step": gt_step,
                "llm": {jname: llm_result},
            },
        )

        if (i + 1) % 200 == 0:
            logger.info(f"[LLM|{jname}] {i+1}/{len(pending)} done")

    logger.info(f"[LLM|{jname}] completed")


# ==============================================================================
# Main processing
# ==============================================================================
async def run(args):
    method = args.method
    cases = collect_failed_cases(datasets=args.datasets)
    if not cases:
        logger.error("No cases found")
        return

    # === Rule-based pass: save to fault_detected/rule.json ===
    if method in ("rule", "both"):
        # Skip cases that already have rule.json
        pending = [c for c in cases if not load_json(_result_path(c["task_dir"], "rule.json")).get("rule")]
        logger.info(f"[Rule] {len(cases)-len(pending)} done, {len(pending)} pending")

        for i, case in enumerate(pending):
            task_dir = Path(case["task_dir"])
            # Load output.json (contains fault_log, events fallback)
            try:
                output_data = json.loads((task_dir / "output.json").read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"[Rule] load output.json failed {case['task_id']}: {e}")
                continue
            # Load agent_steps.json (from trace parsing)
            steps = []
            steps_file = task_dir / "trace_parsed" / "agent_steps.json"
            if steps_file.is_file():
                try:
                    steps = json.loads(steps_file.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.error(f"[Rule] load agent_steps failed {case['task_id']}: {e}")

            # Build per-step text for pattern matching
            step_texts = build_step_texts(output_data, steps)
            if not step_texts:
                continue

            # Ground truth from output.json
            gt_type = parse_fault_type(case["fault_name"])
            gt_step = find_fault_step(output_data.get("_fault_log", []), steps)

            # Run rule detection and save incrementally
            save_json(
                _result_path(case["task_dir"], "rule.json"),
                {
                    "dataset": case["dataset"],
                    "model": case["model"],
                    "system": case["system"],
                    "task_id": case["task_id"],
                    "fault_name": case["fault_name"],
                    "gt_fault_type": gt_type,
                    "gt_fault_step": gt_step,
                    "rule": rule_detect(step_texts),
                },
            )

            if (i + 1) % 1000 == 0:
                logger.info(f"[Rule] {i+1}/{len(pending)} done")

        logger.info(f"[Rule] completed")

    # === LLM-as-Judge pass: each judge runs in a separate process (parallel) ===
    if method in ("llm", "both"):
        judge_names = args.judges or MODEL_PREFIXES
        judges = _build_judge_models(judge_names)
        if not judges:
            logger.error("No judge models configured. Set env vars like DEEPSEEK_OPENAI_MODEL etc.")
            return

        # Launch one process per judge model (parallel, like run_all_method_dataset.py)
        # Each judge uses its own {PREFIX}_MAX_WORKERS as LLM concurrency
        procs = []
        for judge in judges:
            p = multiprocessing.Process(
                target=_run_judge_worker,
                args=(judge, [dict(c) for c in cases]),
            )
            p.start()
            procs.append((judge["name"], p))
            logger.info(f"[LLM] started {judge['name']} (pid={p.pid}, concurrency={judge['max_workers']})")

        # Wait for all judge processes to finish
        for jname, p in procs:
            p.join()
            logger.info(f"[LLM] {jname} exited (code={p.exitcode})")

    # Print metrics
    print_metrics(cases)


# ==============================================================================
# Entry point
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Fault detection: rule-based + LLM-as-a-Judge")
    ap.add_argument("--method", choices=["rule", "llm", "both"], default="rule", help="Detection method")
    ap.add_argument("--datasets", nargs="+", default=None, help="Filter by dataset names (default: all)")
    ap.add_argument("--judges", nargs="+", default=None, help=f"Judge model prefixes (default: {MODEL_PREFIXES})")
    ap.add_argument("--metrics-only", action="store_true", help="Only print metrics from saved results")
    args = ap.parse_args()

    if args.metrics_only:
        cases = collect_failed_cases(datasets=args.datasets)
        print_metrics(cases)
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
