"""RQ3: Fault Attribution — rule-based vs LLM-based on Mini-SE (SWE-bench Pro).
Outputs: rq3_pairs.csv (all detection results), rq3_table.csv (tab:attribution)
"""

import json
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Config
BACKBONE_MODELS = ["gpt-5.2-chat", "claude-sonnet-4.5", "deepseek-v3-2-251201", "doubao-seed-1-8-251228"]
JUDGE_MODELS = ["gpt-5.2-chat", "claude-sonnet-4.5", "deepseek-v3-2-251201", "doubao-seed-1-8-251228"]
JUDGE_LABELS = {
    "gpt-5.2-chat": "GPT-5.2",
    "claude-sonnet-4.5": "Claude-Sonnet-4.5",
    "deepseek-v3-2-251201": "DeepSeek-V3.2",
    "doubao-seed-1-8-251228": "Seed-1.8",
}
# Only evaluate these 6 single-fault types (exclude compound)
FAULT_TYPES = ["error", "timeout", "empty", "truncate", "corrupt", "schema"]
BASE_DIR = "../results_all/results_fault/SWE-bench_Pro"
OUT = "../paper_outputs"


# Collect all fault_detected results (excluding kimi backbone)
rows = []
for backbone in BACKBONE_MODELS:
    pattern = os.path.join(BASE_DIR, backbone, "mini_se", "*", "fault_detected", "rule.json")
    for rule_path in glob.glob(pattern):
        det_dir = os.path.dirname(rule_path)
        try:
            rule_data = json.load(open(rule_path))
        except Exception as e:
            logger.error(f"Failed to read {rule_path}: {e}")
            continue

        gt_type = rule_data.get("gt_fault_type", "")
        gt_step = rule_data.get("gt_fault_step", -1)

        # Skip compound faults — RQ3 only evaluates single-fault types
        if gt_type not in FAULT_TYPES:
            continue

        row = {
            "backbone": backbone,
            "task_id": rule_data.get("task_id", ""),
            "fault_name": rule_data.get("fault_name", ""),
            "gt_type": gt_type,
            "gt_step": int(gt_step) if gt_step is not None else -1,
            # Rule-based predictions
            "rule_type": rule_data.get("rule", {}).get("fault_type", ""),
            "rule_step": rule_data.get("rule", {}).get("fault_step", -1),
        }

        # LLM-as-Judge predictions for each judge model
        for judge in JUDGE_MODELS:
            llm_path = os.path.join(det_dir, f"all-at-once-{judge}.json")
            if os.path.exists(llm_path):
                try:
                    llm_data = json.load(open(llm_path))
                    pred = llm_data.get("llm", {}).get(judge, {})
                    row[f"llm_{judge}_type"] = pred.get("fault_type", "")
                    row[f"llm_{judge}_step"] = pred.get("fault_step", -1)
                except Exception as e:
                    logger.error(f"Failed to read {llm_path}: {e}")
                    row[f"llm_{judge}_type"] = ""
                    row[f"llm_{judge}_step"] = -1
            else:
                row[f"llm_{judge}_type"] = ""
                row[f"llm_{judge}_step"] = -1

        rows.append(row)

df = pd.DataFrame(rows)
logger.info(f"Collected {len(df)} cases (backbone models: {sorted(df['backbone'].unique())})")
logger.info(f"Fault type distribution: {df['gt_type'].value_counts().to_dict()}")

# Save raw pairs
df.to_csv(f"{OUT}/rq3_pairs.csv", index=False)
logger.info(f"Saved rq3_pairs.csv: {len(df)} rows")


# Compute accuracy for a method's predictions vs ground truth
def _accuracy(gt_series, pred_series):
    """Compute accuracy (%) between two series, ignoring empty predictions."""
    mask = pred_series != ""
    if mask.sum() == 0:
        return None
    return round((gt_series[mask] == pred_series[mask]).mean() * 100, 2)


def _step_accuracy(gt_steps, pred_steps):
    """Compute step accuracy (%) — exact match on step index."""
    mask = pred_steps != -1
    if mask.sum() == 0:
        return None
    return round((gt_steps[mask] == pred_steps[mask]).mean() * 100, 2)


# Build table: one row per method, columns = Overall T/S + per-fault-type T/S
table_rows = []

# Rule-based row
rule_row = {"method": "Rule-based"}
rule_row["Overall_T"] = _accuracy(df["gt_type"], df["rule_type"])
rule_row["Overall_S"] = _step_accuracy(df["gt_step"], df["rule_step"])
for ft in FAULT_TYPES:
    mask = df["gt_type"] == ft
    sub = df[mask]
    rule_row[f"{ft}_T"] = _accuracy(sub["gt_type"], sub["rule_type"]) if len(sub) > 0 else None
    rule_row[f"{ft}_S"] = _step_accuracy(sub["gt_step"], sub["rule_step"]) if len(sub) > 0 else None
table_rows.append(rule_row)

# LLM-based rows (one per judge model)
llm_type_preds = {}  # store for kappa computation later
for judge in JUDGE_MODELS:
    jlabel = JUDGE_LABELS[judge]
    type_col = f"llm_{judge}_type"
    step_col = f"llm_{judge}_step"

    jrow = {"method": jlabel}
    jrow["Overall_T"] = _accuracy(df["gt_type"], df[type_col])
    jrow["Overall_S"] = _step_accuracy(df["gt_step"], df[step_col])
    for ft in FAULT_TYPES:
        mask = df["gt_type"] == ft
        sub = df[mask]
        jrow[f"{ft}_T"] = _accuracy(sub["gt_type"], sub[type_col]) if len(sub) > 0 else None
        jrow[f"{ft}_S"] = _step_accuracy(sub["gt_step"], sub[step_col]) if len(sub) > 0 else None
    table_rows.append(jrow)
    llm_type_preds[judge] = df[type_col]

# LLM-based Median row (robust to outlier judges)
med_row = {"method": "LLM-based Med."}
for col_suffix in ["Overall_T", "Overall_S"] + [f"{ft}_{m}" for ft in FAULT_TYPES for m in ["T", "S"]]:
    vals = [r[col_suffix] for r in table_rows[1:] if r.get(col_suffix) is not None]  # skip rule row
    med_row[col_suffix] = round(np.median(vals), 2) if vals else None
table_rows.append(med_row)

# Cohen's Kappa — Rule-based vs Claude-Sonnet-4.5 (type + step)
KAPPA_JUDGE = "claude-sonnet-4.5"
kappa_row = {"method": f"κ (Rule vs Claude)"}

# Log coverage: how many cases each method has predictions
type_col = f"llm_{KAPPA_JUDGE}_type"
step_col = f"llm_{KAPPA_JUDGE}_step"
rule_type_n = (df["rule_type"] != "").sum()
rule_step_n = (df["rule_step"] != -1).sum()
claude_type_n = (df[type_col] != "").sum()
claude_step_n = (df[step_col] != -1).sum()
logger.info(
    f"Kappa coverage: rule type={rule_type_n}/{len(df)}, rule step={rule_step_n}/{len(df)}, claude type={claude_type_n}/{len(df)}, claude step={claude_step_n}/{len(df)}"
)

# Overall type kappa
mask_t = (df["rule_type"] != "") & (df[type_col] != "")
kappa_row["Overall_T"] = (
    round(cohen_kappa_score(df.loc[mask_t, "rule_type"], df.loc[mask_t, type_col]), 3) if mask_t.sum() > 1 else None
)

# Overall step kappa
step_col = f"llm_{KAPPA_JUDGE}_step"
mask_s = (df["rule_step"] != -1) & (df[step_col] != -1)
kappa_row["Overall_S"] = (
    round(cohen_kappa_score(df.loc[mask_s, "rule_step"].astype(str), df.loc[mask_s, step_col].astype(str)), 3)
    if mask_s.sum() > 1
    else None
)

# Per fault type kappa
for ft in FAULT_TYPES:
    mask_ft = df["gt_type"] == ft
    # Type kappa
    mask = mask_ft & (df["rule_type"] != "") & (df[type_col] != "")
    try:
        kappa_row[f"{ft}_T"] = (
            round(cohen_kappa_score(df.loc[mask, "rule_type"], df.loc[mask, type_col]), 3) if mask.sum() > 1 else None
        )
    except Exception as e:
        logger.warning(f"Kappa type failed for {ft}: {e}")
        kappa_row[f"{ft}_T"] = None
    # Step kappa
    mask = mask_ft & (df["rule_step"] != -1) & (df[step_col] != -1)
    try:
        kappa_row[f"{ft}_S"] = (
            round(cohen_kappa_score(df.loc[mask, "rule_step"].astype(str), df.loc[mask, step_col].astype(str)), 3)
            if mask.sum() > 1
            else None
        )
    except Exception as e:
        logger.warning(f"Kappa step failed for {ft}: {e}")
        kappa_row[f"{ft}_S"] = None
table_rows.append(kappa_row)

# Build and save table
table_df = pd.DataFrame(table_rows)
table_df.to_csv(f"{OUT}/rq3_table.csv", index=False)
logger.info(f"Saved rq3_table.csv: {len(table_df)} rows")

# Log table
logger.info(f"Total cases: {len(df)} (excluding kimi backbone, excluding compound)")
header = f"{'Method':20s} {'Overall':>14s}"
for ft in FAULT_TYPES:
    header += f" {ft:>14s}"
logger.info(header)
logger.info("-" * len(header))
for _, r in table_df.iterrows():
    line = f"{r['method']:20s}"
    for col_base in ["Overall"] + FAULT_TYPES:
        t = r.get(f"{col_base}_T", "")
        s = r.get(f"{col_base}_S", "")
        t_str = f"{t}" if t is not None and t != "" else "-"
        s_str = f"{s}" if s is not None and s != "" else "-"
        line += f" {t_str:>6s}/{s_str:<6s}"
    logger.info(line)

logger.info("RQ3 complete.")
