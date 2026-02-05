#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

DIGITS = ["1", "2", "3", "4", "5"]
DIGITS_INT = np.array([1, 2, 3, 4, 5], dtype=float)

DIMS = ["coherence", "consistency", "fluency", "relevance"]
TIERS = ["pure", "digit_plus", "digit_word"]

def parse_probs_cell(x):
    """Cell is a JSON string like {"1":0.1,...} or "{}". Return (5,) array or None."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict) or len(obj) == 0:
        return None

    arr = np.zeros(5, dtype=float)
    ok = False
    for i, d in enumerate(DIGITS):
        if d in obj:
            try:
                arr[i] = float(obj[d])
                ok = True
            except Exception:
                arr[i] = np.nan
    if not ok:
        return None

    if np.all(np.isfinite(arr)) and arr.sum() > 0:
        arr = arr / arr.sum()
    return arr

def probs_to_pred(probs_1to5, mode="expected"):
    """expected: sum(k*p_k); argmax: argmax_k p_k (1..5)."""
    if probs_1to5 is None:
        return np.nan
    if mode == "expected":
        return float((DIGITS_INT * probs_1to5).sum())
    if mode == "argmax":
        return float(np.argmax(probs_1to5) + 1)
    raise ValueError(f"Unknown mode: {mode}")

def compute_metrics(gold, pred):
    """
    gold, pred: 1d arrays (may contain nan)
    Returns: N, MSE, MSE_std, MAE, MAE_std, Spearman, Kendall
    (NO std for correlations)
    """
    mask = np.isfinite(gold) & np.isfinite(pred)
    g = gold[mask].astype(float)
    p = pred[mask].astype(float)
    n = len(g)

    if n == 0:
        return {
            "N": 0,
            "MSE": np.nan, "MSE_std": np.nan,
            "MAE": np.nan, "MAE_std": np.nan,
            "Spearman": np.nan,
            "Kendall": np.nan,
        }

    se = (p - g) ** 2
    ae = np.abs(p - g)

    out = {
        "N": int(n),
        "MSE": float(np.mean(se)),
        "MSE_std": float(np.std(se, ddof=1)) if n > 1 else 0.0,
        "MAE": float(np.mean(ae)),
        "MAE_std": float(np.std(ae, ddof=1)) if n > 1 else 0.0,
    }

    if n < 3:
        out["Spearman"] = np.nan
        out["Kendall"] = np.nan
        return out

    sp, _ = spearmanr(g, p)
    kd, _ = kendalltau(g, p)
    out["Spearman"] = float(sp) if np.isfinite(sp) else np.nan
    out["Kendall"] = float(kd) if np.isfinite(kd) else np.nan
    return out

def load_dim_csv(csv_dir, csv_stem, dim):
    path = os.path.join(csv_dir, f"{csv_stem}_{dim}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV for dim='{dim}': {path}")
    return path, pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default="results", help="Directory containing *_<dim>.csv files")
    parser.add_argument("--csv_stem", default="summeval_results_3tiers", help="Stem before _<dim>.csv")
    parser.add_argument("--pred_mode", default="expected", choices=["expected", "argmax"])
    parser.add_argument("--out_csv", default="results/metrics_table.csv", help="Optional: save metrics table to csv")
    args = parser.parse_args()

    rows = []

    for dim in DIMS:
        path, df = load_dim_csv(args.csv_dir, args.csv_stem, dim)

        if "gold" not in df.columns:
            raise ValueError(f"'gold' column missing in {path}")

        gold = df["gold"].to_numpy(dtype=float)

        for tier in TIERS:
            c_found = f"{tier}_found_rating_token"
            c_rating_probs = f"{tier}_rating_probs_json"

            if c_rating_probs not in df.columns:
                raise ValueError(f"Missing column '{c_rating_probs}' in {path}")

            rating_probs = df[c_rating_probs].apply(parse_probs_cell).tolist()
            rating_pred = np.array([probs_to_pred(p, mode=args.pred_mode) for p in rating_probs], dtype=float)

            if c_found in df.columns:
                found = df[c_found].to_numpy(dtype=float)
                found_mask = found == 1
            else:
                found_mask = np.isfinite(rating_pred)

            rating_pred2 = rating_pred.copy()
            rating_pred2[~found_mask] = np.nan

            cov = float(np.mean(found_mask)) if len(found_mask) > 0 else np.nan
            m = compute_metrics(gold, rating_pred2)

            rows.append({
                "dim": dim,
                "tier": tier,
                "method": f"rating_token_{args.pred_mode}",
                "coverage": cov,
                **m
            })

    out = pd.DataFrame(rows)

    out = out[[
        "dim", "tier", "method", "coverage", "N",
        "MSE", "MSE_std",
        "MAE", "MAE_std",
        "Spearman",
        "Kendall",
    ]]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))

    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print("Saved metrics table:", args.out_csv)

if __name__ == "__main__":
    main()
