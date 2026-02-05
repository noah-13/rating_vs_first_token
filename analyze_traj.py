#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# I/O
# ============================================================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ============================================================
# rating utils
# ============================================================
RATING_KEYS = ["1", "2", "3", "4", "5"]


def expected_score_from_probs(probs_dict):
    return sum(float(k) * float(probs_dict.get(k, 0.0)) for k in RATING_KEYS)


def argmax_rating_from_probs(probs_dict):
    best_k, best_p = 3, -1.0
    for k in RATING_KEYS:
        p = float(probs_dict.get(k, 0.0))
        if p > best_p:
            best_p, best_k = p, int(k)
    return best_k


def rating_mass_from_vocab_probs_single(vocab_probs_single):
    return sum(float(vocab_probs_single.get(k, 0.0)) for k in RATING_KEYS)


# ============================================================
# stats helpers
# ============================================================
def compute_cutoff_D(rating_gen_steps, cutoff_ratio):
    if not rating_gen_steps:
        return 0
    cnt = Counter(rating_gen_steps)
    total = sum(cnt.values())
    cum = 0
    for D in sorted(cnt.keys()):
        cum += cnt[D]
        if cum / total >= cutoff_ratio:
            return D
    return max(cnt.keys())


# ============================================================
# per-dim analysis
# ============================================================
def analyze_one_dim(records, cutoff_ratio, min_count, use_squared_error):
    valid = []
    for r in records:
        if int(r.get("found_pure_rating_token", 0)) != 1:
            continue
        if not r.get("pure_traj"):
            continue
        if r.get("pure_rating_gen_step") is None:
            continue
        if r.get("gold") is None:
            continue
        valid.append(r)

    if not valid:
        return None

    rating_gen_steps = [int(r["pure_rating_gen_step"]) for r in valid]
    D_cut = compute_cutoff_D(rating_gen_steps, cutoff_ratio)

    err_by_dist = defaultdict(list)
    maxerr_by_dist = defaultdict(list)
    mass_by_dist = defaultdict(list)
    count_by_dist = Counter()

    first_step_ratio_numer = Counter()
    first_step_ratio_denom = len(valid)

    for r in valid:
        gold = float(r["gold"])
        g = int(r["pure_rating_gen_step"])

        if g <= D_cut:
            first_step_ratio_numer[g] += 1

        for node in r["pure_traj"]:
            step = int(node["step"])
            dist = g - step
            if dist < 0 or dist > D_cut:
                continue

            probs = node.get("probs", {})
            e = expected_score_from_probs(probs)
            r_hat = argmax_rating_from_probs(probs)

            if use_squared_error:
                err_by_dist[dist].append((e - gold) ** 2)
                maxerr_by_dist[dist].append((r_hat - gold) ** 2)
            else:
                err_by_dist[dist].append(abs(e - gold))
                maxerr_by_dist[dist].append(abs(r_hat - gold))

            mass_by_dist[dist].append(
                rating_mass_from_vocab_probs_single(node.get("vocab_probs_single", {}))
            )
            count_by_dist[dist] += 1

    dists = sorted(d for d in range(D_cut + 1) if count_by_dist[d] >= min_count)
    if not dists:
        return None

    return {
        "D_cut": D_cut,
        "dists": np.asarray(dists),
        "mean_err": np.asarray([np.mean(err_by_dist[d]) for d in dists]),
        "std_err": np.asarray([np.std(err_by_dist[d]) for d in dists]),
        "mean_maxerr": np.asarray([np.mean(maxerr_by_dist[d]) for d in dists]),
        "std_maxerr": np.asarray([np.std(maxerr_by_dist[d]) for d in dists]),
        "first_step_ratio": np.asarray(
            [first_step_ratio_numer[d] / first_step_ratio_denom for d in dists]
        ),
        "mean_mass": np.asarray([np.mean(mass_by_dist[d]) for d in dists]),
        "n_valid": len(valid),
    }


# ============================================================
# plotting helpers
# ============================================================
def plot_error_plus_firststep(ax, dim, res, use_squared_error):
    h1 = ax.errorbar(
        res["dists"], res["mean_err"], yerr=res["std_err"],
        fmt="-o", linewidth=2, capsize=3,
        label="Expected-based error (mean±std)"
    )
    h2 = ax.errorbar(
        res["dists"], res["mean_maxerr"], yerr=res["std_maxerr"],
        fmt="--^", linewidth=2, capsize=3,
        label="Argmax-based error (mean±std)"
    )

    ax.set_title(dim)
    ax.set_xlabel("Distance (steps away from rating generation step)")
    ax.set_ylabel("Mean error" + (" (squared)" if use_squared_error else " (absolute)"))

    axb = ax.twinx()
    h3 = axb.bar(
        res["dists"], res["first_step_ratio"],
        width=0.65, alpha=0.25,
        label="First step ratio (bar)"
    )
    axb.set_ylabel("First step ratio (bar)")

    ax.set_zorder(2)
    ax.patch.set_visible(False)

    return [h1.lines[0], h2.lines[0], h3.patches[0]], [
        "Expected-based error (mean±std)",
        "Argmax-based error (mean±std)",
        "First step ratio (bar)",
    ]


def plot_mass_only(ax, dim, res):
    h = ax.plot(
        res["dists"], res["mean_mass"],
        "-o", linewidth=2,
        label="Mean rating-token mass"
    )[0]

    ax.set_title(dim)
    ax.set_xlabel("Distance (steps away from rating generation step)")
    ax.set_ylabel("Mean rating-token mass")

    return [h], ["Mean rating-token mass"]


def plot_4subplots(results_by_dim, dims, out_path, title, kind, use_squared_error):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    legend_handles, legend_labels = None, None

    for i, dim in enumerate(dims):
        ax = axes[i]
        res = results_by_dim.get(dim)
        if res is None:
            ax.set_title(dim)
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.axis("off")
            continue

        if kind == "error":
            hs, ls = plot_error_plus_firststep(ax, dim, res, use_squared_error)
        else:
            hs, ls = plot_mass_only(ax, dim, res)

        if legend_handles is None:
            legend_handles, legend_labels = hs, ls


    fig.legend(
        legend_handles, legend_labels,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        frameon=False,
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 0.84, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="results")
    ap.add_argument("--out_prefix", type=str, default="results/pure_traj_4dims")
    ap.add_argument("--cutoff_ratio", type=float, default=0.90)
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--use_squared_error", action="store_true")
    args = ap.parse_args()

    dims = ["coherence", "consistency", "fluency", "relevance"]
    results = {}

    for dim in dims:
        path = os.path.join(args.input_dir, f"pure_rating_traj_{dim}.jsonl")
        if not os.path.exists(path):
            print(f"[WARN] missing {path}")
            results[dim] = None
            continue

        recs = load_jsonl(path)
        results[dim] = analyze_one_dim(
            recs,
            cutoff_ratio=args.cutoff_ratio,
            min_count=args.min_count,
            use_squared_error=args.use_squared_error,
        )

    plot_4subplots(
        results, dims,
        f"{args.out_prefix}_fig1_error_firststep.png",
        "Error vs. distance + P(rating_gen_step == distance)",
        kind="error",
        use_squared_error=args.use_squared_error,
    )

    plot_4subplots(
        results, dims,
        f"{args.out_prefix}_fig2_mass_only.png",
        "Mean rating-token mass vs. distance",
        kind="mass",
        use_squared_error=args.use_squared_error,
    )

    print("Done.")


if __name__ == "__main__":
    main()
