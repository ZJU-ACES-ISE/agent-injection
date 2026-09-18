# run_all_method_dataset.py
import json
import os
import re
import sys
import subprocess
import concurrent.futures
import logging
import argparse
import shutil
from dotenv import load_dotenv
import multiprocessing

load_dotenv("../scripts/.env")

logger = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PREFIXES = ["DEEPSEEK", "GPT", "SEED", "KIMI", "CLAUDE"]

METHOD_CONFIGS = {
    "autogen": {
        "script": "run_autogen.py",
        "extra_args": ["--max_llm_calls", "50", "--max_turn", "5", "--code_execute"],
    },
    "mapcoder": {
        "script": "run_mapcoder.py",
        "extra_args": ["--max_llm_calls", "50", "--k", "2", "--t", "3"],
    },
    "evomac": {
        "script": "run_evomac.py",
        "extra_args": ["--max_llm_calls", "50", "--iteration", "3"],
    },
    "mad": {
        "script": "run_mad.py",
        "extra_args": ["--max_llm_calls", "50", "--max_round", "3"],
    },
    "mav": {
        "script": "run_mav.py",
        "extra_args": ["--max_llm_calls", "50", "--n_solutions", "2", "--n_max_verifiers", "2"],
    },
    "mini_se": {
        "script": "run_mini_se.py",
        "extra_args": ["--max_llm_calls", "100"],
        "swe_bench": True,
    },
}

DEFAULT_DATASETS = ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MMLU-Pro", "MATH"]
SWE_DATASETS = ["SWE-bench_Pro"]
ALL_DATASETS = DEFAULT_DATASETS + SWE_DATASETS
DEFAULT_DATA_DIR = "../datasets/data"
DEFAULT_OUTPUT_DIR = "../results"
SWE_MAX_TASKS = 300
_swe_max_tasks = SWE_MAX_TASKS  # mutable at runtime via --swe_max_tasks

# ── Model builder ─────────────────────────────────────────────────


def _build_models(prefixes: list) -> list:
    """Build model configs from env var prefixes. Skip if model name not set."""
    models = []
    for p in prefixes:
        name = os.getenv(f"{p}_OPENAI_MODEL")
        if not name:
            logger.info(f"[config] skip prefix={p}, {p}_OPENAI_MODEL not set")
            continue
        models.append(
            {
                "name": name,
                "base_url": os.getenv(f"{p}_OPENAI_BASE_URL"),
                "api_key": os.getenv(f"{p}_OPENAI_API_KEY"),
                "max_workers": int(os.getenv(f"{p}_MAX_WORKERS", "3")),
            }
        )
    logger.info(f"[config] loaded {len(models)} models: {[(m['name'], m['max_workers']) for m in models]}")
    return models


# ── Dataset loading (cached) ─────────────────────────────────────

_dataset_cache = {}


def _load_swe_dataset(dataset_name: str, max_tasks: int = SWE_MAX_TASKS) -> list:
    """Load SWE-bench dataset from HuggingFace. Cached."""
    mt = max_tasks
    cache_key = ("__swe__", dataset_name, mt)
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]
    try:
        from datasets import load_dataset as hf_load

        hf_name = {"SWE-bench_Pro": "ScaleAI/SWE-bench_Pro"}.get(dataset_name, dataset_name)
        ds = hf_load(hf_name, split="test")
        logger.info(f"[swe] Loaded {dataset_name}: {len(ds)} tasks, using first {mt}")
    except Exception as e:
        logger.error(f"[swe] Failed to load {dataset_name}: {e}")
        _dataset_cache[cache_key] = None
        return None
    result = []
    for i, task in enumerate(ds):
        if i >= mt:
            break
        tid = task["instance_id"].replace("/", "__")
        # Preserve all HF fields for input_raw.json and downstream use
        raw = dict(task)
        result.append(
            {
                "task_id": tid,
                "query": task["problem_statement"],
                "index": i,
                "_raw": raw,
            }
        )
    _dataset_cache[cache_key] = result
    return result


def _load_dataset(data_dir: str, dataset: str) -> list:
    """Load dataset JSON or SWE-bench from HF. Cached."""
    if dataset in SWE_DATASETS:
        return _load_swe_dataset(dataset, _swe_max_tasks)

    cache_key = (data_dir, dataset)
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]

    ds_path = os.path.join(data_dir, f"{dataset}.json")
    if not os.path.exists(ds_path):
        _dataset_cache[cache_key] = None
        return None

    with open(ds_path) as f:
        items = json.load(f)

    result = []
    for i, item in enumerate(items):
        tid = item.get("task_id") or item.get("case_id") or item.get("id")
        if not tid:
            query = item.get("query") or item.get("prompt") or item.get("question") or ""
            clean = re.sub(r"[^a-zA-Z0-9]", "", query)[:10]
            tid = clean if clean else f"task_{i}"
            logger.warning(f"[dataset] {dataset}[{i}]: no id field, using fallback '{tid}'")
        tid = re.sub(r"[^a-zA-Z0-9_-]", "_", str(tid))
        query = item.get("query") or item.get("prompt") or item.get("question") or ""
        result.append({"task_id": tid, "query": query, "index": i, "_raw": item})

    _dataset_cache[cache_key] = result
    return result


def _make_task_dir(output_dir: str, dataset: str, model_name: str, method: str, task_id: str) -> str:
    """Build the output directory path for a single task."""
    return os.path.join(output_dir, dataset, model_name, method, task_id)


def _is_task_done(output_dir: str, dataset: str, model_name: str, method: str, task_id: str) -> bool:
    """Check if a task already has a non-error output."""
    task_dir = _make_task_dir(output_dir, dataset, model_name, method, task_id)
    output_file = os.path.join(task_dir, "output.json")
    if not os.path.isfile(output_file):
        return False
    try:
        with open(output_file) as f:
            obj = json.load(f)
        return "error" not in obj
    except Exception:
        return False


def _get_pending_tasks(data_dir: str, output_dir: str, dataset: str, model_name: str, method: str) -> list:
    """Return list of pending task dicts. None if dataset not found."""
    items = _load_dataset(data_dir, dataset)
    if items is None:
        return None
    return [item for item in items if not _is_task_done(output_dir, dataset, model_name, method, item["task_id"])]


# ── Error cleanup ─────────────────────────────────────────────────


# Errors matching these patterns are retryable (task dir will be removed for retry)
RETRYABLE_ERRORS = ["RateLimitError", "RateLimit", "429", "TooManyRequests", "TPM", "RPM", "timeout"]


def _is_retryable_error(obj: dict) -> bool:
    """Check if an error output.json contains a retryable error (e.g. rate limit)."""
    err_str = str(obj.get("error", "")) + str(obj.get("stderr_tail", ""))
    return any(pat.lower() in err_str.lower() for pat in RETRYABLE_ERRORS)


def _clean_error_entries(output_dir: str, dataset: str, model_name: str, method: str):
    """Remove task dirs that contain retryable error markers (e.g. RateLimitError)."""
    method_dir = os.path.join(output_dir, dataset, model_name, method)
    if not os.path.isdir(method_dir):
        return

    removed = 0
    for task_dir_name in os.listdir(method_dir):
        task_dir = os.path.join(method_dir, task_dir_name)
        if not os.path.isdir(task_dir):
            continue
        output_file = os.path.join(task_dir, "output.json")
        if os.path.isfile(output_file):
            try:
                with open(output_file) as f:
                    obj = json.load(f)
                if "error" in obj and _is_retryable_error(obj):
                    shutil.rmtree(task_dir)
                    removed += 1
            except Exception as e:
                logger.warning(f"[clean] failed to read {output_file}: {e}")

    if removed:
        logger.info(f"[clean] {method}|{dataset}|{model_name}: removed {removed} retryable error dirs for retry")


# ── SWE-bench repo helpers ────────────────────────────────────────


def _ensure_swe_repo(repo: str, repos_dir: str) -> str:
    """Clone or reuse a SWE-bench repo."""
    safe_name = repo.replace("/", "__")
    repo_path = os.path.join(repos_dir, safe_name)
    if os.path.exists(repo_path):
        return repo_path
    os.makedirs(repos_dir, exist_ok=True)
    logger.info(f"[swe] Cloning {repo}...")
    subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{repo}.git", repo_path],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return repo_path


def _ensure_swe_commit(repo_path: str, commit: str):
    """Ensure commit is available in local repo."""
    r = subprocess.run(["git", "-C", repo_path, "cat-file", "-t", commit], capture_output=True, text=True)
    if r.returncode == 0:
        return
    subprocess.run(
        ["git", "-C", repo_path, "fetch", "--quiet", "--depth=1", "origin", commit], capture_output=True, text=True
    )
    r2 = subprocess.run(["git", "-C", repo_path, "cat-file", "-t", commit], capture_output=True, text=True)
    if r2.returncode != 0:
        subprocess.run(["git", "-C", repo_path, "fetch", "--quiet", "--unshallow"], capture_output=True, text=True)


def _run_single_sample(args_tuple):
    """Top-level function for multiprocessing.Pool. Run one sample."""
    base_cmd, method, dataset, model_name, output_dir, fault_map, item = args_tuple

    task_id = item["task_id"]
    query = item["query"]
    tag = f"[{method}|{dataset}|{model_name}]"

    task_output_dir = _make_task_dir(output_dir, dataset, model_name, method, task_id)
    method_output_dir = os.path.join(output_dir, dataset, model_name, method)

    cmd = base_cmd + ["--task_id", task_id, "--query", query, "--output_dir", method_output_dir]

    # SWE-bench: add repo_dir, base_commit, workspaces_dir, --docker
    raw = item.get("_raw", {})
    if dataset in SWE_DATASETS and raw.get("repo"):
        repos_dir = os.path.join(output_dir, "_repos")
        workspaces_dir = os.path.join(output_dir, "_workspaces")
        os.makedirs(workspaces_dir, exist_ok=True)
        try:
            repo_path = _ensure_swe_repo(raw["repo"], repos_dir)
            _ensure_swe_commit(repo_path, raw["base_commit"])
        except Exception as e:
            os.makedirs(task_output_dir, exist_ok=True)
            with open(os.path.join(task_output_dir, "output.json"), "w") as f:
                json.dump({"task_id": task_id, "error": f"repo setup: {e}"}, f)
            return tag, task_id
        cmd += ["--repo_dir", repo_path, "--base_commit", raw["base_commit"], "--workspaces_dir", workspaces_dir]

    if method in ("mapcoder", "mav"):
        cmd += ["--dataset", dataset]
    if fault_map and task_id in fault_map:
        cmd += ["--fault_name", fault_map[task_id]]

    os.makedirs(task_output_dir, exist_ok=True)
    with open(os.path.join(task_output_dir, "input_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw or item, f, ensure_ascii=False, indent=2)

    # Log the full command for debugging (mask query to keep it short)
    cmd_short = [c if len(c) < 200 else c[:80] + "...TRUNCATED" for c in cmd]
    logger.info(f"{tag} {task_id} cmd={cmd_short}")

    timeout = 1800 if dataset in SWE_DATASETS else 600
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        if proc.returncode != 0:
            logger.error(f"{tag} {task_id} exit code {proc.returncode}, output: {proc.stdout}")
            with open(os.path.join(task_output_dir, "output.json"), "w") as f:
                json.dump(
                    {"task_id": task_id, "error": f"exit code {proc.returncode}", "stderr_tail": proc.stdout or ""}, f
                )
    except subprocess.TimeoutExpired:
        logger.error(f"{tag} {task_id} timeout {timeout}s")
        with open(os.path.join(task_output_dir, "output.json"), "w") as f:
            json.dump({"task_id": task_id, "error": f"timeout {timeout}s"}, f)
    except Exception as e:
        logger.error(f"{tag} {task_id} exception: {e}")
        with open(os.path.join(task_output_dir, "output.json"), "w") as f:
            json.dump({"task_id": task_id, "error": str(e)}, f)

    return tag, task_id


# ── Batch runner ──────────────────────────────────────────────────


def _run_model(model_name, max_workers, samples):
    if not samples:
        print(f"  [{model_name}] nothing to do")
        return
    print(f"  [{model_name}] {len(samples)} samples, {max_workers} processes")
    with multiprocessing.Pool(processes=max_workers) as pool:
        for i, (tag, tid) in enumerate(pool.imap_unordered(_run_single_sample, samples), 1):
            if i % 20 == 0 or i == len(samples):
                print(f"  [{model_name}] [{i}/{len(samples)}] {tag} {tid} done")
    print(f"  [{model_name}] ALL DONE")


def batch_run():
    p = argparse.ArgumentParser(description="Batch dispatcher: run MAS agent methods across datasets and models.")
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Subset of methods to run (default: all). Choices: {list(METHOD_CONFIGS.keys())}",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Subset of datasets (default: {DEFAULT_DATASETS}). Use 'SWE-bench_Pro' for SWE-bench.",
    )
    p.add_argument(
        "--swe_max_tasks",
        type=int,
        default=SWE_MAX_TASKS,
        help=f"Max SWE-bench tasks per dataset (default: {SWE_MAX_TASKS})",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Subset of model prefixes (default: {MODEL_PREFIXES})",
    )
    p.add_argument("--output_dir", default=None, help=f"Override output dir (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--data_dir", default=None, help=f"Override data dir (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--max_llm_calls", type=int, default=None, help="Override max_llm_calls for all methods")
    p.add_argument(
        "--fault_inject",
        action="store_true",
        default=False,
        help="Enable fault injection: deterministic experiment assignment per task_id",
    )
    args = p.parse_args()

    # ── Build config from defaults + CLI overrides ──
    methods = dict(METHOD_CONFIGS)
    if args.methods:
        unknown = set(args.methods) - set(methods)
        if unknown:
            p.error(f"Unknown methods: {unknown}. Valid: {list(methods.keys())}")
        methods = {k: v for k, v in methods.items() if k in args.methods}

    # Deep copy extra_args so we can mutate per-run
    methods = {k: {**v, "extra_args": list(v["extra_args"])} for k, v in methods.items()}

    if args.max_llm_calls:
        for m_cfg in methods.values():
            ea = m_cfg["extra_args"]
            if "--max_llm_calls" in ea:
                idx = ea.index("--max_llm_calls")
                ea[idx + 1] = str(args.max_llm_calls)
            else:
                ea += ["--max_llm_calls", str(args.max_llm_calls)]

    datasets = args.datasets or DEFAULT_DATASETS
    global _swe_max_tasks
    _swe_max_tasks = args.swe_max_tasks
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    model_prefixes = args.models or MODEL_PREFIXES

    model_cfgs = _build_models(model_prefixes)
    if not model_cfgs:
        logger.error("No models configured. Set env vars like DEEPSEEK_OPENAI_MODEL, etc.")
        return

    fault_maps = {}
    if args.fault_inject:
        from util import build_fault_assignment

        for dataset in datasets:
            items = _load_dataset(data_dir, dataset)
            if items:
                task_ids = [item["task_id"] for item in items]
                fault_maps[dataset] = build_fault_assignment(task_ids, seed=42)
                # save mapping for reproducibility
                mapping_dir = os.path.join(output_dir, dataset)
                os.makedirs(mapping_dir, exist_ok=True)
                mapping_path = os.path.join(mapping_dir, "fault_assignment.json")
                with open(mapping_path, "w") as f:
                    json.dump(fault_maps[dataset], f, indent=2, ensure_ascii=False)
                logger.info(f"[fault] {dataset}: {len(task_ids)} tasks assigned, saved to {mapping_path}")

    cfg = {
        "methods": methods,
        "datasets": datasets,
        "models": model_cfgs,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "fault_map": fault_maps,
    }

    # ── Build task list grouped by model for parallel execution ──
    samples_by_model = {}
    for model_cfg in cfg["models"]:
        mn = model_cfg["name"]
        samples_by_model[mn] = {"max_workers": model_cfg.get("max_workers", 3), "samples": []}
        for method, method_cfg in cfg["methods"].items():
            for dataset in cfg["datasets"]:
                # skip non-SWE datasets for mini_se and vice versa
                is_swe = dataset in SWE_DATASETS
                is_swe_method = method_cfg.get("swe_bench", False)
                if is_swe != is_swe_method:
                    continue
                _clean_error_entries(cfg["output_dir"], dataset, mn, method)
                pending = _get_pending_tasks(cfg["data_dir"], cfg["output_dir"], dataset, mn, method)
                if pending is None:
                    print(f"  [{method}|{dataset}|{mn}] SKIP (dataset not found)")
                    continue
                if not pending:
                    print(f"  [{method}|{dataset}|{mn}] SKIP (all done)")
                    continue

                base_cmd = [sys.executable, os.path.join(SCRIPT_DIR, method_cfg["script"]), "--model", mn]
                if model_cfg.get("base_url"):
                    base_cmd += ["--base_url", model_cfg["base_url"]]
                if model_cfg.get("api_key"):
                    base_cmd += ["--api_key", model_cfg["api_key"]]
                base_cmd += method_cfg.get("extra_args", [])
                fault_map = cfg.get("fault_map", {}).get(dataset)

                print(f"  [{method}|{dataset}|{mn}] {len(pending)} pending")
                for item in pending:
                    samples_by_model[mn]["samples"].append(
                        (base_cmd, method, dataset, mn, cfg["output_dir"], fault_map, item)
                    )

    total = sum(len(v["samples"]) for v in samples_by_model.values())
    print(f"\nBatch: {len(cfg['models'])} models, {total} total samples\n")

    # ── One process per model, each with Pool(max_workers) ──
    procs = []
    for model_cfg in cfg["models"]:
        mn = model_cfg["name"]
        info = samples_by_model[mn]
        p = multiprocessing.Process(target=_run_model, args=(mn, info["max_workers"], info["samples"]))
        p.start()
        procs.append((mn, p))

    for mn, p in procs:
        p.join()
        print(f"  [{mn}] exited ({p.exitcode})")

    # ── Summary ──
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for method in cfg["methods"]:
        for model_cfg in cfg["models"]:
            for dataset in cfg["datasets"]:
                all_items = _load_dataset(cfg["data_dir"], dataset)
                pending = _get_pending_tasks(cfg["data_dir"], cfg["output_dir"], dataset, model_cfg["name"], method)
                total_count = len(all_items) if all_items else 0
                if pending is None:
                    print(f"  {method:>12} | {model_cfg['name']:>30} | {dataset:>12}: NO DATASET")
                else:
                    done = total_count - len(pending)
                    status = "✓" if not pending else f"pending: {len(pending)}"
                    print(
                        f"  {method:>12} | {model_cfg['name']:>30} | {dataset:>12}: " f"{done}/{total_count} {status}"
                    )

    print(f"\nResults: {os.path.abspath(cfg['output_dir'])}")


if __name__ == "__main__":
    batch_run()
