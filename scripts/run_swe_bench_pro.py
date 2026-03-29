# run_swe_bench_pro.py
import argparse
import json
import os
import subprocess
import sys
import logging
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("SWEBenchPro")


# Dataset loading
def load_dataset():
    """Load SWE-bench Pro from HuggingFace."""
    try:
        from datasets import load_dataset as hf_load

        ds = hf_load("ScaleAI/SWE-bench_Pro", split="test")
        logger.info(f"Loaded SWE-bench Pro: {len(ds)} tasks")
        return ds
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}. Install: pip install datasets")
        sys.exit(1)


# Repo management
def ensure_repo(repo: str, repos_dir: str) -> str:
    """Clone or update a repository. Returns path to local clone."""
    safe_name = repo.replace("/", "__")
    repo_path = os.path.join(repos_dir, safe_name)

    if os.path.exists(repo_path):
        logger.debug(f"Repo exists: {repo_path}")
        return repo_path

    logger.info(f"Cloning {repo}...")
    os.makedirs(repos_dir, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--quiet", f"https://github.com/{repo}.git", repo_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Clone failed for {repo}: {e.stderr}")
        raise
    return repo_path


def checkout_commit(repo_path: str, commit: str):
    """Ensure the repo has the required commit available."""
    r = subprocess.run(["git", "-C", repo_path, "cat-file", "-t", commit], capture_output=True, text=True)
    if r.returncode == 0:
        return
    # Try depth-1 fetch first, then full unshallow
    logger.info(f"Fetching commit {commit[:12]}...")
    subprocess.run(
        ["git", "-C", repo_path, "fetch", "--quiet", "--depth=1", "origin", commit], capture_output=True, text=True
    )
    r2 = subprocess.run(["git", "-C", repo_path, "cat-file", "-t", commit], capture_output=True, text=True)
    if r2.returncode != 0:
        logger.info(f"Depth fetch failed, trying full fetch for {commit[:12]}...")
        subprocess.run(["git", "-C", repo_path, "fetch", "--quiet", "--unshallow"], capture_output=True, text=True)


# Run single task
def run_task(
    task,
    method_output_dir: str,
    repos_dir: str,
    workspaces_dir: str,
    model: str,
    base_url: str,
    api_key: str,
    max_llm_calls: int,
) -> dict:
    """Run Mini-SE agent on a single SWE-bench Pro task."""
    instance_id = task["instance_id"]
    repo = task["repo"]
    base_commit = task["base_commit"]
    problem = task["problem_statement"]

    task_dir = os.path.join(method_output_dir, instance_id.replace("/", "__"))

    # Skip if already completed
    output_file = os.path.join(task_dir, "output.json")
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                obj = json.load(f)
            if "error" not in obj:
                logger.info(f"[SKIP] {instance_id} — already done")
                return {"instance_id": instance_id, "patch": obj.get("patch", ""), "status": "cached"}
        except Exception:
            pass

    try:
        # Ensure repo is available
        repo_path = ensure_repo(repo, repos_dir)
        checkout_commit(repo_path, base_commit)

        # Run the agent
        cmd = [
            sys.executable,
            "run_mini_se.py",
            "--task_id",
            instance_id,
            "--query",
            problem,
            "--repo_dir",
            repo_path,
            "--base_commit",
            base_commit,
            "--output_dir",
            method_output_dir,
            "--max_llm_calls",
            str(max_llm_calls),
            "--workspaces_dir",
            workspaces_dir,
        ]
        if model:
            cmd += ["--model", model]
        if base_url:
            cmd += ["--base_url", base_url]
        if api_key:
            cmd += ["--api_key", api_key]

        logger.info(f"[RUN] {instance_id}")
        start = time.time()
        # Stream both stdout and stderr in real-time
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        try:
            proc.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        elapsed = time.time() - start

        if proc.returncode != 0:
            logger.error(f"[FAIL] {instance_id} ({elapsed:.0f}s) returncode={proc.returncode}")
            os.makedirs(task_dir, exist_ok=True)
            return {
                "instance_id": instance_id,
                "patch": "",
                "status": "error",
                "error": f"exit code {proc.returncode}",
                "elapsed": elapsed,
            }

        # Read patch from output.json
        patch = ""
        if os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    patch = json.load(f).get("patch", "")
            except Exception:
                pass

        logger.info(f"[OK] {instance_id} ({elapsed:.0f}s) patch_len={len(patch)}")
        return {"instance_id": instance_id, "patch": patch, "status": "ok", "elapsed": elapsed}

    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] {instance_id}")
        return {"instance_id": instance_id, "patch": "", "status": "timeout"}
    except Exception as e:
        logger.error(f"[ERROR] {instance_id}: {e}")
        return {"instance_id": instance_id, "patch": "", "status": "error", "error": str(e)}


# Collect results into SWE-bench format
def collect_predictions(method_output_dir: str, output_dir: str, dataset) -> str:
    """Collect all patches into a single predictions JSONL file for evaluation."""
    pred_file = os.path.join(output_dir, "predictions.jsonl")
    count = 0
    with open(pred_file, "w") as f:
        for task in dataset:
            instance_id = task["instance_id"]
            task_dir = os.path.join(method_output_dir, instance_id.replace("/", "__"))
            output_file = os.path.join(task_dir, "output.json")

            patch = ""
            if os.path.exists(output_file):
                try:
                    with open(output_file) as pf:
                        patch = json.load(pf).get("patch", "")
                except Exception:
                    pass

            pred = {
                "instance_id": instance_id,
                "model_name_or_path": "mini-se-adk",
                "model_patch": patch,
            }
            f.write(json.dumps(pred) + "\n")
            if patch.strip():
                count += 1

    logger.info(f"Predictions: {count}/{len(dataset)} non-empty patches -> {pred_file}")
    return pred_file


# Evaluation
def evaluate(pred_file: str, output_dir: str):
    """Run SWE-bench evaluation harness."""
    logger.info("Running SWE-bench evaluation...")

    try:
        # Try swebench harness
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--predictions_path",
            pred_file,
            "--swe_bench_tasks",
            "ScaleAI/SWE-bench_Pro",
            "--log_dir",
            os.path.join(output_dir, "eval_logs"),
            "--testbed",
            os.path.join(output_dir, "eval_testbed"),
            "--skip_existing",
            "--timeout",
            "900",
            "--verbose",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode == 0:
            logger.info("Evaluation completed successfully.")
            logger.info(result.stdout[-2000:])
        else:
            logger.error(f"Evaluation failed: {result.stderr[-1000:]}")
            _fallback_evaluate(pred_file, output_dir)

    except FileNotFoundError:
        logger.warning("swebench harness not found. Using fallback evaluation.")
        _fallback_evaluate(pred_file, output_dir)
    except subprocess.TimeoutExpired:
        logger.error("Evaluation timed out (2h).")
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        _fallback_evaluate(pred_file, output_dir)


def _fallback_evaluate(pred_file: str, output_dir: str):
    """Fallback: print basic statistics from predictions."""
    total = 0
    non_empty = 0
    with open(pred_file) as f:
        for line in f:
            pred = json.loads(line)
            total += 1
            if pred["model_patch"].strip():
                non_empty += 1

    rate = round(non_empty / total * 100, 2) if total > 0 else 0
    logger.info(f"Fallback eval: total={total} patches={non_empty} rate={rate:.1f}%")
    logger.info(
        f"For full eval: pip install swebench && python -m swebench.harness.run_evaluation --predictions_path {pred_file} --swe_bench_tasks ScaleAI/SWE-bench_Pro --log_dir {output_dir}/eval_logs"
    )


# Main
def main():
    p = argparse.ArgumentParser(description="Run Mini-SE on SWE-bench Pro")
    p.add_argument("--output_dir", default="./swe_results", help="Output directory")
    p.add_argument("--repos_dir", default=None, help="Where to clone repos (default: output_dir/_repos)")
    p.add_argument("--instance_id", default=None, help="Run only this instance")
    p.add_argument("--max_tasks", type=int, default=1, help="Limit number of tasks (default: 1)")
    p.add_argument("--max_llm_calls", type=int, default=100, help="Max LLM calls per task")
    p.add_argument("--model", default=None, help="Model name (or OPENAI_MODEL env)")
    p.add_argument("--base_url", default=None, help="API base URL (or OPENAI_BASE_URL env)")
    p.add_argument("--api_key", default=None, help="API key (or OPENAI_API_KEY env)")
    p.add_argument("--eval_only", action="store_true", help="Skip running, just evaluate existing patches")
    p.add_argument("--no_eval", action="store_true", help="Skip evaluation after running")
    args = p.parse_args()

    model = args.model or os.getenv("OPENAI_MODEL", "")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    repos_dir = args.repos_dir or os.path.join(args.output_dir, "_repos")
    workspaces_dir = os.path.join(args.output_dir, "_workspaces")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset()

    # Build method_output_dir: {output_dir}/SWE-bench_Pro/{model}/{mini_se}/
    method_output_dir = os.path.join(args.output_dir, "SWE-bench_Pro", model or "unknown", "mini_se")
    os.makedirs(method_output_dir, exist_ok=True)

    if args.eval_only:
        pred_file = collect_predictions(method_output_dir, args.output_dir, dataset)
        evaluate(pred_file, args.output_dir)
        return

    # Filter tasks
    tasks = list(dataset)
    if args.instance_id:
        tasks = [t for t in tasks if t["instance_id"] == args.instance_id]
        if not tasks:
            logger.error(f"Instance {args.instance_id} not found in dataset")
            sys.exit(1)
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    logger.info(f"Running {len(tasks)} tasks with model={model or '(from env)'}")

    # Run tasks sequentially (LLM is the bottleneck, not CPU)
    results = []
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] {task['instance_id']}")
        r = run_task(task, method_output_dir, repos_dir, workspaces_dir, model, base_url, api_key, args.max_llm_calls)
        results.append(r)


    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    cached = sum(1 for r in results if r["status"] == "cached")
    errors = sum(1 for r in results if r["status"] == "error")
    timeouts = sum(1 for r in results if r["status"] == "timeout")
    non_empty = sum(1 for r in results if r.get("patch", "").strip())
    logger.info(
        f"Summary: total={len(results)} ok={ok} cached={cached} errors={errors} timeouts={timeouts} patches={non_empty}"
    )

    # Collect and evaluate
    if not args.no_eval:
        pred_file = collect_predictions(method_output_dir, args.output_dir, dataset)
        evaluate(pred_file, args.output_dir)


if __name__ == "__main__":
    main()
