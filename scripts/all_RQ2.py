"""RQ2: Impact of Fault Configurations (Type, Target, Strategy, Position, Compound).
Outputs: rq2_pairs.csv (all matched nofault↔fault pairs), rq2_table.csv (tab:config),
         rq2.pdf/png (fault type bar chart), rq2_position.pdf/png (position line chart)
"""

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
        "hatch.linewidth": 0.5,
        "hatch.color": "#444444",
    }
)

# Config
MODELS = ["claude-sonnet-4.5"]  # RQ2 uses claude only
SYSTEMS = ["autogen", "mad", "mapcoder", "evomac","mini_se"]
SYS_LABELS = {"autogen": "AutoGen", "mad": "MAD", "mapcoder": "MapCoder", "evomac": "EvoMAC", "mini_se":"Mini-SE"}
FTYPES = ["error", "timeout", "empty", "truncate", "corrupt", "schema"]
FTYPE_LABELS = ["Error", "Timeout", "Empty", "Truncate", "Corrupt", "Schema"]
FTYPE_COLORS = ["#D8E8F8", "#D5E8D4", "#FFF2CC", "#F5D5C8", "#E8D4F8", "#E8E0D0"]
STRATEGIES = ["single", "persistent", "intermittent", "burst"]
TARGETS = ["llm", "tool"]
TARGET_LABELS = {"llm": "content", "tool": "tool"}
POSITIONS = ["pos_early", "pos_mid", "pos_late"]
POS_LABELS = {"pos_early": "1st call", "pos_mid": "2nd call", "pos_late": "3rd call"}
COMPOUND_SCENARIOS = [
    "compound_api_degradation",
    "compound_content_filter",
    "compound_max_tokens",
    "compound_proxy_html",
    "compound_slow_response",
    "compound_stale_cache",
    "compound_stale_data",
    "compound_wrong_entity",
]
EVAL_CSV = "../results_paper/eval_detail.csv"
OUT = "../paper_outputs"
KEYS = ["model", "dataset", "system", "task_id"]
SYS_DS = {  # valid (system, dataset) pairs
    "autogen": ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MATH", "MMLU-Pro"],
    "mad": ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MATH", "MMLU-Pro"],
    "mapcoder": ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MATH", "MMLU-Pro"],
    "evomac": ["HumanEval", "HumanEval+", "MBPP", "MBPP+", "MATH", "MMLU-Pro"],
    "mini_se": ["SWE-bench_Pro"],
}


# Helpers
def safe_pass(x):
    """Convert pass@1 string to 0/1 int."""
    try:
        v = float(x)
        return int(v) if v in (0, 1) else 0
    except (ValueError, TypeError):
        return 0


def fault_class(ftype):
    """Map fault type to fault class (crash/omission/value)."""
    return {
        "error": "crash",
        "timeout": "crash",
        "empty": "omission",
        "truncate": "value",
        "corrupt": "value",
        "schema": "value",
    }.get(ftype, "other")


def rpd(g):
    """Compute Δpass@1 from matched pairs."""
    bl = g["pass_nf"].mean()
    return round((bl - g["pass_ft"].mean()) * 100, 2)  # Δpass@1 = pass@1_wo - pass@1_w


def rpd_row(g):
    """Full metrics dict for a group of matched pairs."""
    bl, ft = g["pass_nf"].mean(), g["pass_ft"].mean()
    r = (bl - ft) * 100  # Δpass@1 = pass@1_wo - pass@1_w
    return {
        "n": len(g),
        "baseline": round(bl, 4),
        "fault": round(ft, 4),
        "rpd": round(r, 2),
        "p2f": int(((g["pass_nf"] == 1) & (g["pass_ft"] == 0)).sum()),
        "f2p": int(((g["pass_nf"] == 0) & (g["pass_ft"] == 1)).sum()),
    }


def fmt_pct(v, decimals=2):
    """Format float as percentage string (e.g. 0.5132 -> '51.32%')."""
    return f"{v * 100:.{decimals}f}".rstrip("0").rstrip(".") + "%"


# Load data from eval_detail.csv (has correct pass@1 for all datasets including SWE-bench)
logger.info(f"Loading {EVAL_CSV}")
df = pd.read_csv(EVAL_CSV, dtype=str).fillna("")
df = df.rename(columns={"method": "system", "_fault_name": "fault_name", "_fault_fired": "fault_fired", "eval_score": "pass_at_1"})
df["experiment"] = df["fault_type"].str.replace("results_", "", regex=False)
df = df[df["system"].isin(SYSTEMS) & df["model"].isin(MODELS)].copy()
df = df[df.apply(lambda r: r["dataset"] in SYS_DS.get(r["system"], ()), axis=1)].copy()

nf_all = df[df["experiment"] == "nofault"].copy()
ft_all = df[(df["experiment"] == "fault") & (df["fault_name"] != "")].copy()
nf_all["pass_nf"] = nf_all["pass_at_1"].apply(safe_pass)
ft_all["pass_ft"] = ft_all["pass_at_1"].apply(safe_pass)
ft_all["fired_n"] = pd.to_numeric(ft_all["fault_fired"], errors="coerce").fillna(0)

# Only triggered faults (fault_fired > 0)
ft_trig = ft_all[ft_all["fired_n"] > 0].copy()
logger.info(f"nofault={len(nf_all)}, fault_all={len(ft_all)}, triggered={len(ft_trig)}")


# Parse fault_name into target/ftype/strategy for non-compound faults
ft_nc = ft_trig[~ft_trig["fault_name"].str.startswith("compound_")].copy()
parts = ft_nc["fault_name"].str.split("_", n=2, expand=True)
ft_nc["target"] = parts[0]
ft_nc["ftype"] = parts[1]
ft_nc["strat_raw"] = parts[2] if 2 in parts.columns else ""
ft_nc = ft_nc[ft_nc["ftype"].isin(FTYPES) & ft_nc["target"].isin(TARGETS)].copy()
ft_nc["fault_class"] = ft_nc["ftype"].apply(fault_class)

# Classify: basic strategy / position / other
ft_nc["is_pos"] = ft_nc["strat_raw"].isin(POSITIONS)
ft_nc["is_strat"] = ft_nc["strat_raw"].isin(STRATEGIES)

# Parse compound faults
ft_cp = ft_trig[ft_trig["fault_name"].isin(COMPOUND_SCENARIOS)].copy()
ft_cp["scenario"] = ft_cp["fault_name"].str.replace("compound_", "", regex=False)

# Build all matched pairs and merge into one DataFrame
# (a) Basic strategy pairs — used for type, target, and strategy analysis
pairs_basic = ft_nc[ft_nc["is_strat"]].merge(nf_all[KEYS + ["pass_nf"]], on=KEYS, how="inner")
pairs_basic["category"] = "basic"  # type/target/strategy analysis
pairs_basic["pos_label"] = ""
pairs_basic["scenario"] = ""

# (b) Position pairs
pairs_pos = ft_nc[ft_nc["is_pos"]].merge(nf_all[KEYS + ["pass_nf"]], on=KEYS, how="inner")
pairs_pos["category"] = "position"
pairs_pos["pos_label"] = pairs_pos["strat_raw"].map(POS_LABELS)
pairs_pos["scenario"] = ""

# (c) Compound pairs
pairs_comp = ft_cp.merge(nf_all[KEYS + ["pass_nf"]], on=KEYS, how="inner")
pairs_comp["category"] = "compound"
pairs_comp["target"] = ""
pairs_comp["ftype"] = ""
pairs_comp["fault_class"] = ""
pairs_comp["strat_raw"] = ""
pairs_comp["pos_label"] = ""

# Combine all pairs into one CSV
shared_cols = KEYS + [
    "fault_name",
    "category",
    "target",
    "ftype",
    "fault_class",
    "strat_raw",
    "pos_label",
    "scenario",
    "pass_nf",
    "pass_ft",
]
all_pairs = pd.concat([pairs_basic[shared_cols], pairs_pos[shared_cols], pairs_comp[shared_cols]], ignore_index=True)
all_pairs["transition"] = np.where(
    (all_pairs["pass_nf"] == 1) & (all_pairs["pass_ft"] == 0),
    "p2f",
    np.where((all_pairs["pass_nf"] == 0) & (all_pairs["pass_ft"] == 1), "f2p", "same"),
)
all_pairs.sort_values(KEYS + ["category", "fault_name"]).to_csv(f"{OUT}/rq2_pairs.csv", index=False)
logger.info(
    f"Saved rq2_pairs.csv: {len(all_pairs)} pairs (basic={len(pairs_basic)}, pos={len(pairs_pos)}, compound={len(pairs_comp)})"
)


# Build rq2_table.csv — one row per line in LaTeX tab:config
# Helper to compute Δpass@1 for a subset
def _rpd_for(pairs_df, system="ALL", **col_filters):
    """Get Δpass@1 for a filtered subset; returns float or None if empty."""
    g = pairs_df if system == "ALL" else pairs_df[pairs_df["system"] == system]
    for k, v in col_filters.items():
        g = g[g[k] == v]
    return round(rpd(g), 2) if len(g) > 0 else None


table_rows = []

# (a) Fault Type × Target — 9 rows
type_target_configs = [
    ("Crash", "error", "llm"),
    ("Crash", "error", "tool"),
    ("Crash", "timeout", "llm"),
    ("Omission", "empty", "llm"),
    ("Value", "truncate", "llm"),
    ("Value", "truncate", "tool"),
    ("Value", "corrupt", "llm"),
    ("Value", "schema", "llm"),
    ("Value", "schema", "tool"),
]
for fc, ft_name, tgt in type_target_configs:
    row = {"section": "type", "fault_class": fc, "config": ft_name, "target": TARGET_LABELS[tgt]}
    for sys in SYSTEMS:
        row[SYS_LABELS[sys]] = _rpd_for(pairs_basic, sys, ftype=ft_name, target=tgt)
    row["All"] = _rpd_for(pairs_basic, ftype=ft_name, target=tgt)
    table_rows.append(row)

# (b) Injection Strategy — 4 rows
for strat in STRATEGIES:
    row = {"section": "strategy", "fault_class": "", "config": strat, "target": "--"}
    for sys in SYSTEMS:
        row[SYS_LABELS[sys]] = _rpd_for(pairs_basic, sys, strat_raw=strat)
    row["All"] = _rpd_for(pairs_basic, strat_raw=strat)
    table_rows.append(row)

# (c) Injection Position — 3 rows
for pos in POSITIONS:
    pl = POS_LABELS[pos]
    row = {"section": "position", "fault_class": "", "config": pl, "target": "--"}
    for sys in SYSTEMS:
        row[SYS_LABELS[sys]] = _rpd_for(pairs_pos, sys, pos_label=pl)
    row["All"] = _rpd_for(pairs_pos, pos_label=pl)
    table_rows.append(row)

# (d) Compound — single-fault avg + 8 scenarios = 9 rows
row_avg = {"section": "compound", "fault_class": "", "config": "single-fault avg", "target": "--"}
for sys in SYSTEMS:
    row_avg[SYS_LABELS[sys]] = _rpd_for(pairs_basic, sys)
row_avg["All"] = _rpd_for(pairs_basic)
table_rows.append(row_avg)

for sc_full in COMPOUND_SCENARIOS:
    sc = sc_full.replace("compound_", "")
    row = {"section": "compound", "fault_class": "", "config": sc.replace("_", " "), "target": "--"}
    for sys in SYSTEMS:
        row[SYS_LABELS[sys]] = _rpd_for(pairs_comp, sys, scenario=sc)
    row["All"] = _rpd_for(pairs_comp, scenario=sc)
    table_rows.append(row)

table_df = pd.DataFrame(table_rows)
table_df.to_csv(f"{OUT}/rq2_table.csv", index=False)
logger.info(f"Saved rq2_table.csv: {len(table_df)} rows")

# Log the full table
for _, r in table_df.iterrows():
    vals = " ".join(f"{r.get(SYS_LABELS[s], ''):>8}" for s in SYSTEMS)
    logger.info(f"  [{r['section']:10s}] {r['config']:20s} {r['target']:8s} {vals} All={r.get('All','')}")


# Figure 1: Fault type bar chart — baseline vs fault pass@1 per system × fault type
# Compute system×ftype metrics from basic pairs
sf_data = []
for sys in SYSTEMS:
    for ft_name in FTYPES:
        g = pairs_basic[(pairs_basic["system"] == sys) & (pairs_basic["ftype"] == ft_name)]
        if not g.empty:
            sf_data.append({"system": sys, "ftype": ft_name, **rpd_row(g)})
sf = pd.DataFrame(sf_data)

n_ft = len(FTYPES)
bar_w, group_gap = 3 / n_ft, 0.6
group_w, group_step = n_ft * bar_w, n_ft * (3 / n_ft) + 0.6

fig, ax = plt.subplots(figsize=(12, 3))
for ci, sys in enumerate(SYSTEMS):
    sys_tbl = sf[sf["system"] == sys].set_index("ftype")
    for j, ft_name in enumerate(FTYPES):
        if ft_name not in sys_tbl.index:
            continue
        r = sys_tbl.loc[ft_name]
        x = ci * group_step + j * bar_w
        # Baseline (hatch) behind, fault (solid) in front
        ax.bar(
            x,
            r["baseline"],
            bar_w,
            color=FTYPE_COLORS[j],
            edgecolor="black",
            linewidth=0.6,
            hatch="//",
            zorder=2,
            align="edge",
        )
        ax.bar(x, r["fault"], bar_w, color=FTYPE_COLORS[j], edgecolor="black", linewidth=0.6, zorder=3, align="edge")
        xc, hi = x + bar_w / 2, max(r["baseline"], r["fault"])
        # Δpass@1 label
        if abs(r["rpd"]) >= 0.01:
            color = "#C0392B" if r["rpd"] > 0 else "#27AE60"
            txt = f"↓{fmt_pct(r['rpd']/100)}" if r["rpd"] > 0 else f"↑{fmt_pct(abs(r['rpd'])/100)}"
        else:
            color, txt = "#888", "0%"
        ax.text(xc, hi + 0.07, txt, ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

ax.set_xticks([ci * group_step + group_w / 2 for ci in range(len(SYSTEMS))])
ax.set_xticklabels([SYS_LABELS[s] for s in SYSTEMS], fontsize=15, fontweight="bold")
ax.set_xlim(-0.15, len(SYSTEMS) * group_step - group_gap + 0.15)
ax.set_ylim(0, 0.8)
ax.set_ylabel("pass@1", fontsize=17, fontweight="bold")
ax.tick_params(axis="y", labelsize=13)
ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_linewidth(0.8)
handles = [
    Patch(facecolor=FTYPE_COLORS[j], edgecolor="black", linewidth=0.8, label=FTYPE_LABELS[j]) for j in range(n_ft)
] + [
    Patch(facecolor="#CCC", edgecolor="black", linewidth=0.8, hatch="//", label="w/o FI"),
    Patch(facecolor="#CCC", edgecolor="black", linewidth=0.8, label="w/ FI"),
]
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.51, 1.04),
    ncol=n_ft + 2,
    fontsize=12,
    frameon=True,
    edgecolor="black",
    prop={"weight": "bold", "size": 12},
)
plt.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.08)
fig.savefig(f"{OUT}/rq2.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}/rq2.png", format="png", bbox_inches="tight", pad_inches=0.02)
logger.info("Saved rq2.pdf/.png")


# Figure 2: Position line chart — Δpass@1 by call index per system × fault type
POS_FTYPES = ["error", "timeout", "schema"]
FT_CLR = {"error": "#5B8DB8", "timeout": "#6BAA6B", "schema": "#C8A820"}
FT_MRK = {"error": "o", "timeout": "s", "schema": "^"}
pos_x, pos_order = [1, 2, 3], ["1st call", "2nd call", "3rd call"]

plot_sys = [s for s in SYSTEMS if not pairs_pos[pairs_pos["system"] == s].empty]
fig2, axes = plt.subplots(1, len(plot_sys), figsize=(3.2 * len(plot_sys), 3), sharey=True)
if len(plot_sys) == 1:
    axes = [axes]

for ax_i, sys in enumerate(plot_sys):
    ax = axes[ax_i]
    for ft_name in POS_FTYPES:
        vals = []
        for pl in pos_order:
            g = pairs_pos[
                (pairs_pos["system"] == sys) & (pairs_pos["ftype"] == ft_name) & (pairs_pos["pos_label"] == pl)
            ]
            bl = g["pass_nf"].mean() if not g.empty else 0
            ft_v = g["pass_ft"].mean() if not g.empty else 0
            vals.append((bl - ft_v) * 100)  # Δpass@1
        ax.plot(
            pos_x,
            vals,
            color=FT_CLR[ft_name],
            marker=FT_MRK[ft_name],
            markersize=6,
            linewidth=1.8,
            label=ft_name.capitalize(),
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=3,
        )
        for xi, vi in zip(pos_x, vals):
            if abs(vi) > 0.5:
                ax.text(
                    xi,
                    vi + 2.5,
                    f"{vi:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=FT_CLR[ft_name],
                )
    ax.set_title(SYS_LABELS[sys], fontsize=13, fontweight="bold")
    ax.set_xticks(pos_x)
    ax.set_xticklabels(pos_order, fontsize=11)
    ax.set_xlim(0.5, 3.5)
    ax.axhline(y=0, color="#999", linewidth=0.8, linestyle="--", zorder=1)
    ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=11)

axes[0].set_ylabel("Δpass@1", fontsize=13, fontweight="bold")
handles = [
    plt.Line2D(
        [0],
        [0],
        color=FT_CLR[f],
        marker=FT_MRK[f],
        markersize=6,
        linewidth=1.8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=f.capitalize(),
    )
    for f in POS_FTYPES
]
fig2.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=3,
    fontsize=11,
    frameon=True,
    edgecolor="black",
    prop={"weight": "bold", "size": 11},
)
plt.tight_layout()
fig2.savefig(f"{OUT}/rq2_position.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
fig2.savefig(f"{OUT}/rq2_position.png", format="png", bbox_inches="tight", pad_inches=0.02)
logger.info("Saved rq2_position.pdf/.png")

logger.info("RQ2 complete.")
