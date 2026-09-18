"""
Batch-parse all traces under results_all/ using ADKPhoenixParser.

For each task directory, reads trace/*.json, parses with ADKPhoenixParser,
and saves agent_steps.json, agent_settings.json, agent_dependency.json
into a sibling trace_parsed/ directory.

After parsing, aggregates settings/dependency per system (dataset/model/system),
saving merged_settings.json and merged_dependency.json at the system directory level.

Usage:
    uv run python scripts/parse_trace_all.py [--workers 8]
    uv run python scripts/parse_trace_all.py --merge-only       # skip parsing, only aggregate
"""

import json
import os
import sys
import time
import logging
import traceback
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

# ── import parser from same package ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from parse_trace import ADKPhoenixParser, merge_settings, merge_dependencies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path("../results_all")
DATASETS = ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MMLU-Pro", "MATH", "SWE-bench_Pro"]
SYSTEMS = ["autogen", "evomac", "mad", "mapcoder", "mav", "mini_se"]


# ── load spans from a trace directory (flat, no case sub-dirs) ──────────
def load_spans_flat(trace_dir: Path) -> list:
    """Load all spans from *.json files directly under trace_dir."""
    spans = []
    for f in sorted(trace_dir.glob("*.json")):
        if f.name.startswith("._"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")
            continue

        container = data.get("data", data)
        for rs in container.get("resource_spans", []):
            for ss in rs.get("scope_spans", []):
                scope_name = ss.get("scope", {}).get("name", "")
                for span in ss.get("spans", []):
                    span["_scope"] = scope_name
                    spans.append(span)

    if not spans:
        return []

    # If multiple trace_ids exist, keep only the largest trace
    by_trace = defaultdict(list)
    for span in spans:
        by_trace[span.get("trace_id", "")].append(span)
    if len(by_trace) > 1:
        logger.warning(f"{trace_dir}: {len(by_trace)} trace_ids found, using largest")
    return max(by_trace.values(), key=len)


# ── process one task directory ──────────────────────────────────────────
def process_task(task_dir_str: str) -> dict:
    """Parse trace for a single task. Returns a status dict."""
    task_dir = Path(task_dir_str)
    trace_dir = task_dir / "trace"
    out_dir = task_dir / "trace_parsed"

    result = {"task_dir": task_dir_str, "status": "ok", "steps": 0, "spans": 0}

    # Skip if already parsed
    if (out_dir / "agent_steps.json").exists():
        result["status"] = "skipped"
        return result

    if not trace_dir.is_dir():
        result["status"] = "no_trace_dir"
        return result

    try:
        spans = load_spans_flat(trace_dir)
        if not spans:
            result["status"] = "no_spans"
            return result

        result["spans"] = len(spans)

        parser = ADKPhoenixParser(spans)
        steps, settings, dependency = parser.parse_all()
        result["steps"] = len(steps)

        # Save results
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "agent_steps.json").write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "agent_settings.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "agent_dependency.json").write_text(
            json.dumps(dependency, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{e.__class__.__name__}: {e}"
        logger.error(f"Error processing {task_dir}: {e}\n{traceback.format_exc()}")

    return result


# ── collect all task directories ────────────────────────────────────────
def collect_task_dirs() -> list:
    """Walk results_all/ and return all task directory paths."""
    task_dirs = []
    for exp in ["results_nofault", "results_fault"]:
        base = ROOT / exp
        if not base.is_dir():
            continue
        for dataset in DATASETS:
            ds_dir = base / dataset
            if not ds_dir.is_dir():
                continue
            for model in sorted(os.listdir(ds_dir)):
                if model.startswith(".") or model.endswith(".json"):
                    continue
                for system in SYSTEMS:
                    sys_dir = ds_dir / model / system
                    if not sys_dir.is_dir():
                        continue
                    for task_id in sorted(os.listdir(sys_dir)):
                        if task_id.startswith("."):
                            continue
                        task_dir = sys_dir / task_id
                        if task_dir.is_dir():
                            task_dirs.append(str(task_dir))
    return task_dirs


# ── aggregate settings/dependency per system ─────────────────────────────
def aggregate_system_level():
    """Walk all system dirs, load per-task settings/dependency, merge, save."""
    count = 0
    for exp in ["results_nofault", "results_fault"]:
        base = ROOT / exp
        if not base.is_dir():
            continue
        for dataset in DATASETS:
            ds_dir = base / dataset
            if not ds_dir.is_dir():
                continue
            for model in sorted(os.listdir(ds_dir)):
                if model.startswith(".") or model.endswith(".json"):
                    continue
                for system in SYSTEMS:
                    sys_dir = ds_dir / model / system
                    if not sys_dir.is_dir():
                        continue
                    # Collect all per-task settings and dependencies
                    all_settings, all_deps = [], []
                    for task_id in sorted(os.listdir(sys_dir)):
                        if task_id.startswith("."):
                            continue
                        parsed_dir = sys_dir / task_id / "trace_parsed"
                        s_file = parsed_dir / "agent_settings.json"
                        d_file = parsed_dir / "agent_dependency.json"
                        if s_file.is_file():
                            try:
                                all_settings.append(json.loads(s_file.read_text(encoding="utf-8")))
                            except Exception as e:
                                logger.warning(f"Bad settings {s_file}: {e}")
                        if d_file.is_file():
                            try:
                                all_deps.append(json.loads(d_file.read_text(encoding="utf-8")))
                            except Exception as e:
                                logger.warning(f"Bad dependency {d_file}: {e}")
                    if not all_settings and not all_deps:
                        continue
                    # Merge and save at system directory level
                    if all_settings:
                        merged_s = merge_settings(all_settings)
                        (sys_dir / "merged_settings.json").write_text(
                            json.dumps(merged_s, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                    if all_deps:
                        merged_d = merge_dependencies(all_deps)
                        (sys_dir / "merged_dependency.json").write_text(
                            json.dumps(merged_d, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                    count += 1
    logger.info(f"Aggregated settings/dependency for {count} system groups")


# ── main ────────────────────────────────────────────────────────────────
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Batch parse all traces in results_all/")
    ap.add_argument(
        "--workers",
        type=int,
        default=min(mp.cpu_count(), 8),
        help="Number of parallel workers (default: min(cpu_count, 8))",
    )
    ap.add_argument("--force", action="store_true", help="Re-parse even if trace_parsed/ already exists")
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip per-task parsing, only aggregate system-level settings/dependency",
    )
    args = ap.parse_args()

    if not args.merge_only:
        logger.info("Collecting task directories ...")
        task_dirs = collect_task_dirs()
        logger.info(f"Found {len(task_dirs)} task directories")

        if not task_dirs:
            logger.error("No task directories found. Is results_all/ in the right place?")
            return

        # Stats counters
        stats = defaultdict(int)
        t0 = time.time()

        logger.info(f"Starting {args.workers} workers ...")
        with mp.Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_task, task_dirs, chunksize=32), 1):
                stats[result["status"]] += 1
                if result["status"] == "error":
                    logger.error(f"  FAILED: {result['task_dir']}: {result.get('error', '?')}")

                if i % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (len(task_dirs) - i) / rate
                    logger.info(
                        f"  progress: {i}/{len(task_dirs)} ({i*100//len(task_dirs)}%) "
                        f"| {rate:.0f} tasks/s | ETA {eta:.0f}s "
                        f"| ok={stats['ok']} skip={stats['skipped']} err={stats['error']}"
                    )

        elapsed = time.time() - t0
        logger.info(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"Stats: {dict(stats)}")
        logger.info(f"Total: {sum(stats.values())} tasks")

    # Always run system-level aggregation
    logger.info("Aggregating settings/dependency per system ...")
    aggregate_system_level()


if __name__ == "__main__":
    main()
