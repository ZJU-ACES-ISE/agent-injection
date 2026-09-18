"""Analyze FI overhead from bench_fi_raw.json.
Paired comparison: same-round diff eliminates network variance.
Usage: uv run python bench_fi_analyze.py
"""
import json
import statistics
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = "../paper_outputs/bench_fi_raw.json"
OUT_CSV = "../paper_outputs/bench_fi_summary.csv"

with open(RAW_PATH) as f:
    raw = json.load(f)
logger.info(f"Loaded {RAW_PATH}: config={raw['config']}")

bl = raw["baseline"]
nf = raw["patch_no_fault"]
wf = raw["patch_with_fault"]
n = min(len(bl), len(nf), len(wf))
logger.info(f"Rounds: {n}")

# Paired diff: patch_latency - baseline_latency for each round
# Positive = patch is slower (overhead), negative = patch is faster (noise)
diff_nf = [nf[i] - bl[i] for i in range(n)]  # patch_no_fault - baseline
diff_wf = [wf[i] - bl[i] for i in range(n)]  # patch_with_fault - baseline

# Log per-round data
logger.info(f"{'Round':>5s}  {'BL':>8s}  {'NF':>8s}  {'WF':>8s}  {'Δ_NF':>8s}  {'Δ_WF':>8s}")
for i in range(n):
    logger.info(f"{i+1:5d}  {bl[i]:8.1f}  {nf[i]:8.1f}  {wf[i]:8.1f}  {diff_nf[i]:+8.1f}  {diff_wf[i]:+8.1f}")

def summarize(name: str, diffs: list) -> dict:
    """Stats on paired differences."""
    s = sorted(diffs)
    row = {
        "group": name,
        "n": len(s),
        "median_diff_ms": round(statistics.median(s), 1),
        "mean_diff_ms": round(statistics.mean(s), 1),
        "std_diff_ms": round(statistics.stdev(s), 1) if len(s) > 1 else 0,
        "min_diff_ms": round(min(s), 1),
        "max_diff_ms": round(max(s), 1),
    }
    # Median % overhead relative to median baseline
    bl_median = statistics.median(bl[:n])
    row["bl_median_ms"] = round(bl_median, 1)
    row["overhead_pct"] = round(row["median_diff_ms"] / bl_median * 100, 2) if bl_median > 0 else 0
    logger.info(
        f"{name:25s}: median_diff={row['median_diff_ms']:+.1f}ms  mean_diff={row['mean_diff_ms']:+.1f}ms  "
        f"std={row['std_diff_ms']:.1f}ms  overhead={row['overhead_pct']:+.2f}% (vs bl_median={row['bl_median_ms']:.0f}ms)"
    )
    return row

rows = [
    summarize("Patch (no fault)", diff_nf),
    summarize("Patch (with fault)", diff_wf),
]

# Save CSV
cols = ["group", "n", "bl_median_ms", "median_diff_ms", "mean_diff_ms", "std_diff_ms", "min_diff_ms", "max_diff_ms", "overhead_pct"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
logger.info(f"Saved: {OUT_CSV}")