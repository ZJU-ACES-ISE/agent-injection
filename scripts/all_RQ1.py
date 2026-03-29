"""RQ1: Overall Robustness. Outputs: rq1_raw.csv, rq1.csv, rq1.pdf/png"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.dpi": 600,
    }
)

SYSTEMS = ["autogen", "mad", "mapcoder", "evomac", "mini_se"]
SYS_LABEL = {"autogen": "AutoGen", "mad": "MAD", "mapcoder": "MapCoder", "evomac": "EvoMAC", "mini_se": "Mini-SE"}
SYS_DS = {  # valid (system, dataset) pairs
    "autogen": ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MATH", "MMLU-Pro"],
    "mad": ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MATH", "MMLU-Pro"],
    "mapcoder": ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MATH", "MMLU-Pro"],
    "evomac": ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MATH", "MMLU-Pro"],
    "mini_se": ["SWE-bench_Pro"],
}
MODELS = ["gpt-5.2-chat", "claude-sonnet-4.5", "deepseek-v3-2-251201", "doubao-seed-1-8-251228"]
MODEL_LABEL = {
    "gpt-5.2-chat": "GPT-5.2",
    "claude-sonnet-4.5": "Claude-Sonnet-4.5",
    "deepseek-v3-2-251201": "DeepSeek-V3.2",
    "doubao-seed-1-8-251228": "Seed-1.8",
}
TABLE_DS = ["HumanEval", "HumanEval+", "MBPP", "MMLU-Pro", "MATH", "SWE-bench_Pro"]

EVAL_CSV = "../results_paper/eval_detail.csv"
RAW_CSV = "../paper_outputs/raw_results.csv"
OUT = "../paper_outputs/rq1"


def safe_pass(x):
    try:
        v = float(x)
        return int(v) if v in (0, 1) else 0
    except (ValueError, TypeError):
        return 0


def rpd_metrics(g):
    """Δpass@1 (percentage points), transitions, recovery."""
    bl, ft = g["pass_nf"].mean(), g["pass_ft"].mean()
    rpd = (bl - ft) * 100  # Δpass@1 = pass@1_wo - pass@1_w
    p2f = int(((g["pass_nf"] == 1) & (g["pass_ft"] == 0)).sum())
    f2p = int(((g["pass_nf"] == 0) & (g["pass_ft"] == 1)).sum())
    return {
        "n": len(g),
        "baseline": round(bl, 4),
        "fault": round(ft, 4),
        "rpd": round(rpd, 2),
        "p2f": p2f,
        "f2p": f2p,
        "recovery": round(f2p / p2f * 100, 2) if p2f > 0 else 0.0,
    }


# --- Load eval_detail.csv: pass@1 + trigger for all datasets ---
logger.info(f"Loading {EVAL_CSV}")
df = pd.read_csv(EVAL_CSV, dtype=str).fillna("")
df = df.rename(
    columns={"method": "system", "_fault_name": "fault_name", "_fault_fired": "fault_fired", "eval_score": "pass_at_1"}
)
df["experiment"] = df["fault_type"].str.replace("results_", "", regex=False)
df = df[df["system"].isin(SYSTEMS) & df["model"].isin(MODELS)].copy()
df = df[df.apply(lambda r: r["dataset"] in SYS_DS.get(r["system"], ()), axis=1)].copy()
logger.info(f"Filtered: {len(df)} rows, {sorted(df['dataset'].unique())}")

nf = df[df["experiment"] == "nofault"].copy()
ft = df[(df["experiment"] == "fault") & (df["fault_name"] != "")].copy()  # drop empty fault_name
nf["pass_nf"] = nf["pass_at_1"].apply(safe_pass)
ft["pass_ft"] = ft["pass_at_1"].apply(safe_pass)
ft["fired_n"] = pd.to_numeric(ft["fault_fired"], errors="coerce").fillna(0)
ft["triggered"] = ft["fired_n"] > 0

ft_trig = ft[ft["triggered"]].copy()  # only triggered for Δpass@1
logger.info(f"nofault={len(nf)}, fault={len(ft)}, triggered={len(ft_trig)}")

# Match triggered fault rows with nofault baseline
keys = ["model", "dataset", "system", "task_id"]
pairs = ft_trig.merge(nf[keys + ["pass_nf"]], on=keys, how="inner")
logger.info(f"Matched pairs: {len(pairs)}")

# --- Output 1: rq1_raw.csv (all matched pairs, traceable) ---
raw_cols = ["model", "dataset", "system", "task_id", "fault_name", "fault_fired", "pass_nf", "pass_ft"]
raw_out = pairs[raw_cols].copy()
raw_out["transition"] = np.where(
    (raw_out["pass_nf"] == 1) & (raw_out["pass_ft"] == 0),
    "p2f",
    np.where((raw_out["pass_nf"] == 0) & (raw_out["pass_ft"] == 1), "f2p", "same"),
)
raw_out.sort_values(keys + ["fault_name"]).to_csv(f"{OUT}_raw.csv", index=False)
logger.info(f"Saved {OUT}_raw.csv: {len(raw_out)} rows")

# --- Compute metrics at all aggregation levels ---
rows = []
for (m, s, d), g in pairs.groupby(["model", "system", "dataset"]):
    rows.append({"level": "msd", "model": m, "system": s, "dataset": d, **rpd_metrics(g)})
for (s, d), g in pairs.groupby(["system", "dataset"]):
    rows.append({"level": "sd", "model": "ALL", "system": s, "dataset": d, **rpd_metrics(g)})
for s, g in pairs.groupby("system"):
    rows.append({"level": "sys", "model": "ALL", "system": s, "dataset": "ALL", **rpd_metrics(g)})
for m, g in pairs.groupby("model"):
    rows.append({"level": "model", "model": m, "system": "ALL", "dataset": "ALL", **rpd_metrics(g)})
rows.append({"level": "grand", "model": "ALL", "system": "ALL", "dataset": "ALL", **rpd_metrics(pairs)})

# Trigger rate per (model, system, dataset)
trig = (
    ft.groupby(["model", "system", "dataset"])
    .agg(total=("triggered", "count"), trig_count=("triggered", "sum"))
    .reset_index()
)
trig["trig_rate"] = (trig["trig_count"] / trig["total"] * 100).round(2)

# Overhead from raw_results.csv (only triggered fault cases, claude model only)
logger.info(f"Loading {RAW_CSV} for overhead (model=claude only)")
raw = pd.read_csv(RAW_CSV, dtype=str).fillna("")
raw = raw[(raw["system"].isin(SYSTEMS)) & (raw["model"] == "claude-sonnet-4.5")].copy()
for c in ["llm_step_count", "total_tool_calls", "fault_fired"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
oh_rows = []
for sys in SYSTEMS:
    nf_s = raw[(raw["system"] == sys) & (raw["experiment"] == "nofault")]
    # only count fault cases where fault actually triggered
    ft_s = raw[(raw["system"] == sys) & (raw["experiment"] == "fault") & (raw["fault_fired"] > 0)]
    ln, lf = nf_s["llm_step_count"].mean(), ft_s["llm_step_count"].mean()
    tn, tf = nf_s["total_tool_calls"].mean(), ft_s["total_tool_calls"].mean()
    oh_rows.append(
        {
            "system": sys,
            "llm_wo": round(ln, 2) if pd.notna(ln) else None,
            "llm_w": round(lf, 2) if pd.notna(lf) else None,
            "llm_ratio": round(lf / ln, 2) if pd.notna(ln) and ln > 0 and pd.notna(lf) else None,
            "tool_wo": round(tn, 2) if pd.notna(tn) else None,
            "tool_w": round(tf, 2) if pd.notna(tf) else None,
            "tool_ratio": round(tf / tn, 2) if pd.notna(tn) and tn > 0 and pd.notna(tf) else None,
        }
    )

# --- Output 2: rq1.csv (all processed data in one file) ---
metrics = pd.DataFrame(rows).sort_values(["level", "model", "system", "dataset"]).reset_index(drop=True)
with pd.ExcelWriter(f"{OUT}.xlsx", engine="openpyxl") if False else open(f"{OUT}.csv", "w") as _:
    pass  # placeholder, write below

# Combine metrics + trigger rates + overhead into one CSV with sheet-like sections
metrics.to_csv(f"{OUT}.csv", index=False)
# Append trigger rates and overhead as additional sections
with open(f"{OUT}.csv", "a") as f:
    f.write("\n# Trigger rates per (model, system, dataset)\n")
    trig.to_csv(f, index=False)
    f.write("\n# Overhead (LLM/tool calls w/o vs w/ fault)\n")
    pd.DataFrame(oh_rows).to_csv(f, index=False)

logger.info(f"Saved {OUT}.csv")

# Log grand summary
grand = metrics[metrics["level"] == "grand"].iloc[0]
logger.info(
    f"Grand: n={grand['n']}, bl={grand['baseline']:.4f}, ft={grand['fault']:.4f}, "
    f"Δpass@1={grand['rpd']:+.2f}, recovery={grand['recovery']:.1f}%"
)
for _, r in metrics[metrics["level"] == "sys"].sort_values("rpd", ascending=False).iterrows():
    logger.info(f"  {r['system']:12s} n={r['n']:5.0f} Δpass@1={r['rpd']:+.1f} recovery={r['recovery']:.1f}%")
logger.info(f"Overhead: {pd.DataFrame(oh_rows).to_string(index=False)}")

# --- Figure: Δpass@1 bar chart (4 systems × 5 datasets) ---
FDS = ["HumanEval", "HumanEval+", "MBPP", "MMLU-Pro", "MATH"]
FSYS = ["autogen", "mad", "mapcoder", "evomac"]
DSC = ["#D8E8F8", "#D5E8D4", "#FFF2CC", "#F5D5C8", "#E8D4F8"]
FS = 12
FD = 8

sd = metrics[metrics["level"] == "sd"]
pivot = sd.pivot(index="system", columns="dataset", values="rpd").reindex(index=FSYS, columns=FDS).fillna(0)
fig, ax = plt.subplots(figsize=(10, 3.5))
bw = 0.15
for j, ds in enumerate(FDS):
    xs = np.arange(len(FSYS)) + j * bw
    ax.bar(xs, pivot[ds].values, bw, color=DSC[j], edgecolor="black", linewidth=0.5, label=ds, zorder=3)
    for x, v in zip(xs, pivot[ds].values):
        if v > 0:
            ax.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=FD, fontweight="bold")

sys_avg = metrics[metrics["level"] == "sys"].set_index("system").reindex(FSYS)
for i, s in enumerate(FSYS):
    avg = sys_avg.loc[s, "rpd"]
    cx = i + (len(FDS) - 1) * bw / 2
    ax.plot(cx, avg, marker="D", color="#C0392B", markersize=6, zorder=5, markeredgecolor="black", markeredgewidth=0.4)
    ax.text(cx, avg + 1.2, f"{avg:.1f}", ha="center", va="bottom", fontsize=FD, fontweight="bold", color="#C0392B")

ax.set_xticks([i + (len(FDS) - 1) * bw / 2 for i in range(len(FSYS))])
ax.set_xticklabels([SYS_LABEL[s] for s in FSYS], fontsize=FS, fontweight="bold")
ax.set_ylabel("Δpass@1", fontsize=FS, fontweight="bold")
ax.set_ylim(0, pivot.values.max() * 1.3)
ax.tick_params(axis="y", labelsize=FS - 2)
ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.4, alpha=0.3)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_linewidth(0.6)
handles = [Patch(facecolor=DSC[j], edgecolor="black", linewidth=0.6, label=FDS[j]) for j in range(len(FDS))]
handles.append(
    plt.Line2D(
        [0],
        [0],
        marker="D",
        color="#C0392B",
        linestyle="None",
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.4,
        label="Sys Avg",
    )
)
ax.legend(handles=handles, fontsize=FS - 3, frameon=True, edgecolor="black", loc="upper left", prop={"size": FS - 3})
plt.tight_layout()
fig.savefig(f"{OUT}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}.png", format="png", bbox_inches="tight", pad_inches=0.02)
logger.info(f"Saved {OUT}.pdf/.png")
