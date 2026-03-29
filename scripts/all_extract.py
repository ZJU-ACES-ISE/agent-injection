# scripts/all_extract.py
import argparse
import json
import os
import sys
import threading
import logging
import traceback
import time
import multiprocessing as mp
from pathlib import Path
from collections import Counter

import pandas as pd

# ── Import evaluation primitives from run_all_eval.py ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all_eval import (
    extract_code,
    _run_humaneval_test,
    _run_assert_test,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Anchor all paths to project root (works regardless of CWD)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
ROOT = _PROJECT_ROOT / "results_all"
DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "paper_outputs")
SYSTEMS = ["autogen", "evomac", "mad", "mapcoder", "mav", "mini_se"]  # for HE cache loading
CODE_DATASETS = {"HumanEval", "HumanEval+", "MBPP", "MBPP+"}  # datasets with code execution eval

# ── Module-level caches (set by _worker_init in each subprocess) ────────
_HE_TEST_CACHE = {}  # {"HumanEval_0": test_str, "HumanEval/0": test_str}
_FAULT_ASSIGN = {}  # {"HumanEval": {"HumanEval_0": "llm_error_single"}}


def _worker_init(he_cache: dict, fault_assign: dict):
    global _HE_TEST_CACHE, _FAULT_ASSIGN
    _HE_TEST_CACHE = he_cache
    _FAULT_ASSIGN = fault_assign


# ═══════════════════════════════════════════════════════════════════════
#  Per-task evaluation (delegates to run_all_eval primitives)
# ═══════════════════════════════════════════════════════════════════════
def _evaluate_task_code(final_answer: str, raw: dict, dir_task_id: str, dataset: str):
    """Return 1 (pass), 0 (fail), or 'skip' (non-code dataset, use eval_detail.csv instead)."""
    # Only evaluate code datasets here; MATH/MMLU-Pro/SWE use run_all_eval.py
    if dataset not in CODE_DATASETS:
        return "skip"

    source = raw.get("source", "")
    entry_point = raw.get("entry_point", "")
    test_cases = raw.get("test_cases") or []

    if not final_answer:
        return 0

    code = extract_code(final_answer)

    if source.startswith("HumanEval"):
        if not entry_point:
            return 0
        # HumanEval+ has no test_cases → fallback to HumanEval base cache
        if not test_cases:
            cached = _HE_TEST_CACHE.get(dir_task_id) or _HE_TEST_CACHE.get(raw.get("task_id", ""))
            if cached:
                test_cases = [cached]
        if not test_cases or not test_cases[0]:
            return 0
        return 1 if _run_humaneval_test(code, test_cases[0], entry_point) else 0

    if source.startswith("MBPP"):
        if not test_cases:
            return 0
        return 1 if all(_run_assert_test(code, tc) for tc in test_cases) else 0

    return "skip"


# ═══════════════════════════════════════════════════════════════════════
#  Process one task (worker subprocess)
# ═══════════════════════════════════════════════════════════════════════
def process_task(args) -> dict:
    exp_type, dataset, model, system, task_id, task_dir = args
    fault_assignments = _FAULT_ASSIGN.get(dataset, {})
    row = dict(
        dataset=dataset,
        model=model,
        system=system,
        task_id=task_id,
        orig_task_id="",
        source="",
        entry_point="",
        experiment=exp_type,
        fault_name="",
        fault_fired="",
        fault_trigger_at="",
        llm_step_count="",
        tool_call_steps="",
        total_tool_calls="",
        total_input_tokens="",
        total_output_tokens="",
        total_llm_time="",
        total_exec_time="",
        n_agents="",
        n_events="",
        answer_length="",
        max_llm_calls="",
        has_final_answer="",
        has_error="",
        pass_at_1="",
    )
    try:
        out_path = os.path.join(task_dir, "output.json")
        raw_path = os.path.join(task_dir, "input_raw.json")
        steps_path = os.path.join(task_dir, "trace_parsed", "agent_steps.json")

        # input.json — system configuration parameters
        input_path = os.path.join(task_dir, "input.json")
        if os.path.exists(input_path):
            try:
                with open(input_path) as f:
                    inp = json.load(f)
                row["max_llm_calls"] = inp.get("max_llm_calls", "")
            except Exception as e:
                logger.warning(f"Failed reading {input_path}: {e}")

        if not os.path.exists(out_path):
            row["pass_at_1"] = "missing"
            return row

        with open(out_path) as f:
            out = json.load(f)

        # Basic output metadata
        row["has_final_answer"] = 1 if "final_answer" in out else 0
        row["has_error"] = 1 if "error" in out else 0
        # Event count (number of conversation turns)
        events = out.get("events", [])
        row["n_events"] = len(events) if isinstance(events, list) else ""
        # Answer length — character length of final_answer
        fa = out.get("final_answer", "")
        row["answer_length"] = len(fa) if fa else 0

        # Fault metadata (from output.json + fault_assignment.json fallback)
        if exp_type == "fault":
            row["fault_name"] = out.get("_fault_name", "") or fault_assignments.get(task_id, "")
            ff = out.get("_fault_fired")
            row["fault_fired"] = "" if ff is None else int(ff)
            fault_log = out.get("_fault_log")
            if fault_log:
                last_entry = fault_log[-1]
                row["fault_trigger_at"] = last_entry.get("count", "")

        # System crashed — no final_answer
        if "error" in out and "final_answer" not in out:
            row["pass_at_1"] = "error"

        # LLM step count and detailed trace metrics
        if os.path.exists(steps_path):
            try:
                with open(steps_path) as f:
                    steps = json.load(f)
                row["llm_step_count"] = len(steps)
                tool_call_steps = 0
                total_tool_calls = 0
                total_in_tok = 0
                total_out_tok = 0
                total_llm_t = 0.0
                total_exec_t = 0.0
                agents_seen = set()
                for s in steps:
                    agents_seen.add(s.get("agent_name", ""))
                    tools = s.get("agent", {}).get("tools_called", [])
                    if tools:
                        tool_call_steps += 1
                        total_tool_calls += len(tools)
                    u = s.get("step_usage", {})
                    total_in_tok += u.get("input_tokens", 0) or 0
                    total_out_tok += u.get("output_tokens", 0) or 0
                    total_llm_t += u.get("llm_inference_time", 0) or 0
                    total_exec_t += u.get("step_execution_time", 0) or 0
                row["tool_call_steps"] = tool_call_steps
                row["total_tool_calls"] = total_tool_calls
                row["total_input_tokens"] = total_in_tok
                row["total_output_tokens"] = total_out_tok
                row["total_llm_time"] = round(total_llm_t, 2)
                row["total_exec_time"] = round(total_exec_t, 2)
                row["n_agents"] = len(agents_seen)
            except Exception as e:
                logger.warning(f"Failed reading {steps_path}: {e}")

        # Load input_raw.json for metadata + test cases
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                raw = json.load(f)
            row["orig_task_id"] = raw.get("task_id", "")
            row["source"] = raw.get("source", "")
            row["entry_point"] = raw.get("entry_point", "")
        else:
            raw = None

        # Evaluate pass@1 (skip if already marked error)
        if row["pass_at_1"] == "error":
            return row
        if raw is None:
            row["pass_at_1"] = "no_raw"
            return row

        final_answer = out.get("final_answer", "")
        row["pass_at_1"] = _evaluate_task_code(final_answer, raw, task_id, dataset)

    except Exception as e:
        logger.error(f"Error processing {task_dir}: {e}\n{traceback.format_exc()}")
        row["pass_at_1"] = "error"

    return row


# ═══════════════════════════════════════════════════════════════════════
#  CSV checkpoint (append-mode, thread-safe)
# ═══════════════════════════════════════════════════════════════════════
_CSV_COLUMNS = [
    "dataset",
    "model",
    "system",
    "task_id",
    "orig_task_id",
    "source",
    "entry_point",
    "experiment",
    "fault_name",
    "fault_fired",
    "fault_trigger_at",
    "llm_step_count",
    "tool_call_steps",
    "total_tool_calls",
    "total_input_tokens",
    "total_output_tokens",
    "total_llm_time",
    "total_exec_time",
    "n_agents",
    "n_events",
    "answer_length",
    "max_llm_calls",
    "has_final_answer",
    "has_error",
    "pass_at_1",
]
_csv_lock = threading.Lock()


def load_completed_from_csv(csv_path: str) -> tuple:
    """Load existing CSV checkpoint. Returns (df, set_of_completed_keys)."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.DataFrame(columns=_CSV_COLUMNS), set()
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        keys = set(zip(df["dataset"], df["model"], df["system"], df["task_id"], df["experiment"]))
        return df, keys
    except Exception as e:
        logger.warning(f"Failed loading checkpoint {csv_path}: {e}")
        return pd.DataFrame(columns=_CSV_COLUMNS), set()


def append_batch_to_csv(csv_path: str, rows: list, write_header: bool = False):
    """Append rows to CSV file (thread-safe)."""
    if not rows:
        return
    with _csv_lock:
        df = pd.DataFrame(rows)[_CSV_COLUMNS]
        df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ═══════════════════════════════════════════════════════════════════════
#  Task collection and cache loading
# ═══════════════════════════════════════════════════════════════════════
def collect_all_tasks() -> list:
    """Auto-discover all tasks from results_all/{results_nofault,results_fault}/."""
    tasks = []
    for exp_type in ["nofault", "fault"]:
        base = ROOT / f"results_{exp_type}"
        if not base.is_dir():
            continue
        # Auto-discover datasets (all subdirs under base)
        for ds_dir in sorted(base.iterdir()):
            if ds_dir.name.startswith(".") or not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            for model_dir in sorted(ds_dir.iterdir()):
                if model_dir.name.startswith(".") or not model_dir.is_dir():
                    continue
                # Auto-discover systems (all subdirs under model)
                for sys_dir in sorted(model_dir.iterdir()):
                    if sys_dir.name.startswith(".") or not sys_dir.is_dir():
                        continue
                    system = sys_dir.name
                    for task_dir in sorted(sys_dir.iterdir()):
                        if task_dir.name.startswith(".") or not task_dir.is_dir():
                            continue
                        tasks.append((exp_type, ds, model_dir.name, system, task_dir.name, str(task_dir)))
    return tasks


def load_he_test_cache() -> dict:
    """Load HumanEval test cases for HumanEval+ fallback."""
    cache = {}
    he_base = ROOT / "results_nofault" / "HumanEval"
    if not he_base.is_dir():
        logger.warning(f"HumanEval base dir not found: {he_base}")
        return cache
    for model_dir in sorted(he_base.iterdir()):
        if model_dir.name.startswith(".") or not model_dir.is_dir():
            continue
        for sys_name in SYSTEMS:
            sys_path = model_dir / sys_name
            if not sys_path.is_dir():
                continue
            for task_dir in sorted(sys_path.iterdir()):
                if task_dir.name.startswith(".") or not task_dir.is_dir():
                    continue
                raw_path = task_dir / "input_raw.json"
                if not raw_path.exists():
                    continue
                try:
                    r = json.loads(raw_path.read_text())
                    tcs = r.get("test_cases") or []
                    if tcs:
                        cache[task_dir.name] = tcs[0]
                        orig_id = r.get("task_id", "")
                        if orig_id:
                            cache[orig_id] = tcs[0]
                except Exception as e:
                    logger.warning(f"HE cache load fail {raw_path}: {e}")
            if cache:
                logger.info(f"HE cache loaded from {model_dir.name}/{sys_name}")
                return cache
    return cache


def load_fault_assignments() -> dict:
    """Load fault_assignment.json from all dataset dirs under results_fault/."""
    fa = {}
    fault_base = ROOT / "results_fault"
    if not fault_base.is_dir():
        return fa
    for ds_dir in sorted(fault_base.iterdir()):
        if not ds_dir.is_dir():
            continue
        p = ds_dir / "fault_assignment.json"
        if p.exists():
            try:
                fa[ds_dir.name] = json.loads(p.read_text())
            except Exception as e:
                logger.error(f"Failed loading {p}: {e}")
    return fa


# ═══════════════════════════════════════════════════════════════════════
#  Main extraction
# ═══════════════════════════════════════════════════════════════════════
def extract_raw(output_dir: str, force: bool = False, workers: int = None, swe_whitelist: set = None) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("  Extracting raw results")
    logger.info("=" * 60)

    csv_path = os.path.join(output_dir, "raw_results.csv")

    fault_assign = load_fault_assignments()
    logger.info(f"Fault assignments: {', '.join(f'{k}={len(v)}' for k, v in fault_assign.items())}")

    he_cache = load_he_test_cache()
    logger.info(f"HumanEval test cache: {len(he_cache)} entries")

    all_tasks = collect_all_tasks()
    # If swe_whitelist is set, filter SWE-bench_Pro tasks to only those in the whitelist
    if swe_whitelist is not None:
        before = len(all_tasks)
        all_tasks = [t for t in all_tasks if t[1] != "SWE-bench_Pro" or t[4] in swe_whitelist]
        logger.info(
            f"SWE whitelist filter: {before} -> {len(all_tasks)} tasks (kept {sum(1 for t in all_tasks if t[1]=='SWE-bench_Pro')} SWE tasks)"
        )
    logger.info(f"Total tasks discovered: {len(all_tasks)}")

    if force:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        completed = set()
        logger.info("Force mode: cleared existing CSV")
    else:
        _, completed = load_completed_from_csv(csv_path)
        logger.info(f"Already in CSV: {len(completed)} tasks")

    pending = [t for t in all_tasks if (t[1], t[2], t[3], t[4], t[0]) not in completed]

    if not pending:
        logger.info("All tasks already extracted.")
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        logger.info(f"CSV: {csv_path} ({len(df)} rows)")
        return df

    logger.info(f"Pending: {len(pending)} tasks")
    n_workers = workers or min(mp.cpu_count(), 8)
    logger.info(f"Workers: {n_workers}")

    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    BATCH = 200
    t0 = time.time()
    batch, done = [], 0

    with mp.Pool(n_workers, initializer=_worker_init, initargs=(he_cache, fault_assign)) as pool:
        try:
            for result in pool.imap_unordered(process_task, pending, chunksize=32):
                if result is None:
                    continue
                batch.append(result)
                done += 1
                if len(batch) >= BATCH:
                    append_batch_to_csv(csv_path, batch, write_header=need_header)
                    need_header = False
                    batch = []
                    elapsed = time.time() - t0
                    speed = done / elapsed
                    eta = (len(pending) - done) / speed if speed > 0 else 0
                    logger.info(
                        f"  {done}/{len(pending)} ({done*100//len(pending)}%) " f"| {speed:.0f} t/s | ETA {eta:.0f}s"
                    )
        except KeyboardInterrupt:
            logger.warning(f"Interrupted! Flushing {len(batch)} buffered rows ...")
            append_batch_to_csv(csv_path, batch, write_header=need_header)
            pool.terminate()
            pool.join()
            logger.info(f"Checkpoint saved ({done} done). Restart to resume.")
            sys.exit(1)

    if batch:
        append_batch_to_csv(csv_path, batch, write_header=need_header)

    elapsed = time.time() - t0
    logger.info(f"Extraction: {done} tasks in {elapsed:.1f}s ({done/max(elapsed,1):.0f} t/s)")

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    logger.info(f"CSV: {csv_path} ({len(df)} rows)")
    logger.info(f"pass@1 distribution: {dict(Counter(df['pass_at_1'].tolist()))}")
    return df


def _load_swe_range(start: int, end: int) -> set:
    """Load SWE-bench_Pro task_ids in [start, end) from HuggingFace."""
    from datasets import load_dataset as hf_load

    ds = hf_load("ScaleAI/SWE-bench_Pro", split="test")
    tids = set()
    for i, task in enumerate(ds):
        if i >= end:
            break
        if i >= start:
            tids.add(task["instance_id"].replace("/", "__"))
    return tids


def main():
    ap = argparse.ArgumentParser(description="Extract raw results into CSV")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--force", action="store_true", help="Clear CSV, re-extract all")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--swe-start", type=int, default=None, help="SWE-bench_Pro start index (inclusive), e.g. 100")
    ap.add_argument("--swe-end", type=int, default=None, help="SWE-bench_Pro end index (exclusive), e.g. 200")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "raw_results.csv")

    # If --swe-start/--swe-end specified: load whitelist and remove old SWE rows
    swe_whitelist = None
    if args.swe_start is not None or args.swe_end is not None:
        s, e = args.swe_start or 0, args.swe_end or 300
        swe_whitelist = _load_swe_range(s, e)
        logger.info(f"--swe-range [{s},{e}): {len(swe_whitelist)} task_ids, sample={sorted(swe_whitelist)[:2]}")
        # Remove existing SWE-bench_Pro rows from CSV so they get re-extracted
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
            before = len(df)
            df = df[df["dataset"] != "SWE-bench_Pro"]
            df.to_csv(csv_path, index=False)
            logger.info(f"Removed {before - len(df)} SWE rows from CSV ({before} -> {len(df)})")

    t0 = time.time()
    extract_raw(args.output_dir, force=args.force, workers=args.workers, swe_whitelist=swe_whitelist)
    logger.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
