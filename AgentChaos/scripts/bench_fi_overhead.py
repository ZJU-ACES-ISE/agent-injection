"""Benchmark FI framework overhead: compare API latency with/without httpx patch.
Usage: uv run python bench_fi_overhead.py [--n 20] [--prefix DEEPSEEK
Outputs: ../paper_outputs/bench_fi_raw.json, ../paper_outputs/bench_fi_summary.csv
"""

import asyncio
import copy
import json
import os
import time
import statistics
import argparse
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress noisy loggers from fault_injection / util
for name in ["LiteLLM", "httpx", "openai", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)


async def single_call(client, base_url: str, api_key: str, model: str) -> float:
    """Send one chat completion request, return latency in ms."""
    import httpx

    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    t0 = time.perf_counter()
    resp = await client.post(url, json=body, headers=headers, timeout=30.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return elapsed_ms


async def run_group(name: str, n: int, base_url: str, api_key: str, model: str) -> list:
    """Run n calls and return list of latencies in ms."""
    import httpx

    latencies = []
    async with httpx.AsyncClient() as client:
        for i in range(n):
            try:
                ms = await single_call(client, base_url, api_key, model)
                latencies.append(ms)
                logger.info(f"  [{name}] {i+1}/{n}: {ms:.1f}ms")
            except Exception as e:
                logger.error(f"  [{name}] {i+1}/{n}: FAILED {e}")
    return latencies


def calc_stats(name: str, latencies: list) -> dict:
    """Compute stats with trimmed mean (drop 1 min + 1 max). Returns dict + logs."""
    if len(latencies) < 3:
        logger.info(f"{name}: too few data points ({len(latencies)})")
        return {"group": name, "n": len(latencies)}
    s = sorted(latencies)
    trimmed = s[1:-1]  # drop 1 min and 1 max
    row = {
        "group": name,
        "n": len(latencies),
        "mean": round(statistics.mean(latencies), 1),
        "trimmed_mean": round(statistics.mean(trimmed), 1),
        "std": round(statistics.stdev(latencies), 1),
        "p50": round(s[len(s) // 2], 1),
        "p99": round(s[min(int(len(s) * 0.99), len(s) - 1)], 1),
        "min": round(min(latencies), 1),
        "max": round(max(latencies), 1),
    }
    logger.info(
        f"{name:25s}: n={row['n']}  mean={row['mean']}ms  trimmed={row['trimmed_mean']}ms  "
        f"std={row['std']}ms  p50={row['p50']}ms  min={row['min']}ms  max={row['max']}ms"
    )
    return row


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of requests per group")
    parser.add_argument("--model", default=None, help="Model name (default: from env)")
    parser.add_argument("--prefix", default="DEEPSEEK", help="Env var prefix (default: DEEPSEEK)")
    args = parser.parse_args()

    # Load config from env — try DEEPSEEK first (reliable OpenAI-compatible API)
    prefix = args.prefix or "DEEPSEEK"
    model = args.model or os.getenv(f"{prefix}_OPENAI_MODEL", "deepseek-v3-2-251201")
    base_url = os.getenv(f"{prefix}_OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv(f"{prefix}_OPENAI_API_KEY", "")
    if not api_key:
        logger.error("GPT_OPENAI_API_KEY not set in env")
        return

    logger.info(f"Config: model={model}, base_url={base_url}, n={args.n}")

    from fault_injection import FaultEngine, FaultSpec, install_httpx_patch, uninstall_httpx_patch
    import httpx

    # Warmup: 5 calls to fully establish connection pool / DNS / TLS cache
    logger.info("Warmup (5 calls)...")
    await run_group("warmup", 5, base_url, api_key, model)

    # Interleaved execution: each round sends 1 call per group (baseline, patch_nf, patch_wf)
    # This ensures network conditions affect all groups equally
    baseline, patch_nofault, patch_fault = [], [], []
    logger.info(f"Running {args.n} interleaved rounds (3 calls per round)...")

    async with httpx.AsyncClient() as client:
        for i in range(args.n):
            # (a) Baseline: unbind engine so patch passes through without engine
            uninstall_httpx_patch()
            try:
                ms = await single_call(client, base_url, api_key, model)
                baseline.append(ms)
                logger.info(f"  [round {i+1}/{args.n}] baseline: {ms:.1f}ms")
            except Exception as e:
                logger.error(f"  [round {i+1}/{args.n}] baseline FAILED: {e}")

            # (b) Patch + empty engine (no faults)
            engine_nf = FaultEngine(seed=42)
            install_httpx_patch(engine_nf)
            try:
                ms = await single_call(client, base_url, api_key, model)
                patch_nofault.append(ms)
                logger.info(f"  [round {i+1}/{args.n}] patch_nf: {ms:.1f}ms")
            except Exception as e:
                logger.error(f"  [round {i+1}/{args.n}] patch_nf FAILED: {e}")
            uninstall_httpx_patch()

            # (c) Patch + single fault (fresh engine each round so it fires once)
            engine_wf = FaultEngine(seed=42)
            engine_wf.add(
                FaultSpec(
                    intercept="response",
                    action="set",
                    target_path="$.choices[0].message.content",
                    value="[INJECTED ERROR]",
                    max_count=1,
                    description="benchmark: single error injection",
                )
            )
            install_httpx_patch(engine_wf)
            try:
                ms = await single_call(client, base_url, api_key, model)
                patch_fault.append(ms)
                logger.info(f"  [round {i+1}/{args.n}] patch_wf: {ms:.1f}ms (fired={len(engine_wf.log)})")
            except Exception as e:
                logger.error(f"  [round {i+1}/{args.n}] patch_wf FAILED: {e}")
            uninstall_httpx_patch()

    # Save raw latencies to JSON
    out_dir = "../paper_outputs"
    os.makedirs(out_dir, exist_ok=True)
    raw_data = {
        "config": {"model": model, "base_url": base_url, "n": args.n},
        "baseline": baseline,
        "patch_no_fault": patch_nofault,
        "patch_with_fault": patch_fault,
    }
    raw_path = f"{out_dir}/bench_fi_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    logger.info(f"Saved raw latencies: {raw_path}")

    # Compute stats (trimmed mean: drop 1 min + 1 max)
    logger.info("=" * 60)
    rows = [
        calc_stats("Baseline (no patch)", baseline),
        calc_stats("Patch (no fault)", patch_nofault),
        calc_stats("Patch (with fault)", patch_fault),
    ]

    # Overhead using trimmed mean
    if rows[0].get("trimmed_mean") and rows[1].get("trimmed_mean"):
        oh = rows[1]["trimmed_mean"] - rows[0]["trimmed_mean"]
        pct = oh / rows[0]["trimmed_mean"] * 100
        logger.info(f"Patch overhead (no fault, trimmed): {oh:+.1f}ms ({pct:+.2f}%)")
    if rows[0].get("trimmed_mean") and rows[2].get("trimmed_mean"):
        oh = rows[2]["trimmed_mean"] - rows[0]["trimmed_mean"]
        pct = oh / rows[0]["trimmed_mean"] * 100
        logger.info(f"Patch overhead (with fault, trimmed): {oh:+.1f}ms ({pct:+.2f}%)")

    # Save summary CSV
    import csv

    csv_path = f"{out_dir}/bench_fi_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
