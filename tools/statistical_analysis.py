# ponytail: Statistical comparison of 4 sleep staging architectures across 10 folds.
# Friedman test → pairwise Wilcoxon signed-rank with Holm correction.
import os
import json
import argparse
import numpy as np
from itertools import combinations

def load_fold_metrics(results_dir):
    """Load per-fold metrics for each architecture. Returns dict[arch] -> dict[metric] -> list[float]."""
    architectures = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])
    data = {}
    for arch in architectures:
        arch_dir = os.path.join(results_dir, arch)
        acc, mf1, kappa = [], [], []
        for fold in range(10):
            path = os.path.join(arch_dir, f"fold_{fold}", "metrics.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                m = json.load(f)
            acc.append(m["test_acc"])
            mf1.append(m["test_macro_f1"])
            kappa.append(m["test_kappa"])
        if acc:
            data[arch] = {"ACC": np.array(acc), "MF1": np.array(mf1), "Kappa": np.array(kappa)}
    return data


def friedman_test(groups):
    """Friedman chi-squared test for k related samples.
    groups: list of arrays, each of length n (same subjects/folds).
    Returns (chi2, p_value).
    """
    from scipy.stats import friedmanchisquare
    stat, p = friedmanchisquare(*groups)
    return stat, p


def wilcoxon_test(x, y):
    """Two-sided Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(x, y, alternative="two-sided")
    return stat, p


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (m - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


def run_analysis(results_dir):
    data = load_fold_metrics(results_dir)
    architectures = sorted(data.keys())
    metrics = ["ACC", "MF1", "Kappa"]

    if len(architectures) < 2:
        print("Need at least 2 architectures to compare.")
        return

    # --- Descriptive stats ---
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS (Mean ± Std over 10 folds)")
    print("=" * 70)
    header = f"{'Model':<22}" + "".join(f"{m:>16}" for m in metrics)
    print(header)
    print("-" * 70)
    for arch in architectures:
        row = f"{arch:<22}"
        for m in metrics:
            vals = data[arch][m]
            row += f"{np.mean(vals):.4f} ± {np.std(vals):.4f}  "
        print(row)
    print()

    # --- Friedman test per metric ---
    print("=" * 70)
    print("FRIEDMAN TEST (H0: all models perform equally)")
    print("=" * 70)
    friedman_results = {}
    for m in metrics:
        groups = [data[arch][m] for arch in architectures]
        chi2, p = friedman_test(groups)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        friedman_results[m] = (chi2, p, sig)
        print(f"  {m:>6}: chi2 = {chi2:.4f}, p = {p:.6f}  {sig}")
    print()

    # --- Pairwise Wilcoxon with Holm correction + effect sizes ---
    print("=" * 70)
    print("PAIRWISE WILCOXON SIGNED-RANK TESTS (Holm-corrected)")
    print("=" * 70)
    pairs = list(combinations(architectures, 2))

    all_rows = []
    for m in metrics:
        raw_ps = []
        pair_stats = []
        for a, b in pairs:
            stat, p = wilcoxon_test(data[a][m], data[b][m])
            raw_ps.append(p)
            # Effect sizes: median diff (pp) and rank-biserial r_rb
            diffs = data[a][m] - data[b][m]
            median_diff_pp = np.median(diffs) * 100  # percentage points
            # Matched-pairs rank-biserial: r_rb = 1 - (2*T) / (n*(n+1)/2)
            # where T is the smaller of W+ and W- (= stat from scipy)
            n = len(diffs[diffs != 0])
            if n > 0:
                r_rb = 1.0 - (2.0 * stat) / (n * (n + 1) / 2.0)
            else:
                r_rb = 0.0
            pair_stats.append((a, b, stat, p, median_diff_pp, r_rb))

        adj_ps = holm_correction(raw_ps)

        print(f"\n  Metric: {m}")
        print(f"  {'Pair':<40} {'W-stat':>8} {'p-raw':>10} {'p-adj':>10} {'Med.Diff':>10} {'r_rb':>7} {'Sig':>5}")
        print("  " + "-" * 92)
        for i, (a, b, stat, p_raw, med_diff, r_rb) in enumerate(pair_stats):
            p_adj = adj_ps[i]
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
            pair_label = f"{a} vs {b}"
            print(f"  {pair_label:<40} {stat:>8.1f} {p_raw:>10.6f} {p_adj:>10.6f} {med_diff:>+9.2f}pp {r_rb:>7.3f} {sig:>5}")
            all_rows.append({
                "metric": m, "model_a": a, "model_b": b,
                "w_statistic": stat, "p_raw": p_raw, "p_adjusted": p_adj,
                "median_diff_pp": round(med_diff, 2), "rank_biserial_r": round(r_rb, 3),
                "significant": sig
            })

    # --- Save CSV ---
    import csv
    csv_path = os.path.join(results_dir, "statistical_tests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "metric", "model_a", "model_b", "w_statistic", "p_raw", "p_adjusted",
            "median_diff_pp", "rank_biserial_r", "significant"
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of sleep staging models")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Path to results directory")
    args = parser.parse_args()
    run_analysis(args.results_dir)


if __name__ == "__main__":
    main()
