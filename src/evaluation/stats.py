import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def holm_bonferroni(pvals):
    """
    Applies Holm-Bonferroni step-down correction to an array of p-values.
    """
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj

def bootstrap_ci(diff, n_bootstraps=2000, ci=95, seed=42):
    """
    Computes bootstrap confidence interval for the mean difference.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    if len(diff) == 0:
        return np.nan, np.nan
    boot_means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_bootstraps)]
    alpha = (100 - ci) / 2.0
    low = np.percentile(boot_means, alpha)
    high = np.percentile(boot_means, 100 - alpha)
    return float(low), float(high)

def compare_models_paired(scores_a, scores_b, metric_name="dice"):
    """
    Performs direct paired Wilcoxon signed-rank test between Model A and Model B.
    Returns dictionary with statistical metrics.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b

    if np.allclose(diff, 0):
        stat, p = np.nan, 1.0
    else:
        try:
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except Exception:
            stat, p = np.nan, 1.0

    ci_low, ci_high = bootstrap_ci(diff)

    return {
        "n_samples": len(diff),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "delta_mean": float(np.mean(diff)),
        "delta_median": float(np.median(diff)),
        "won_pct": float(100.0 * np.mean(diff > 0)),
        "p_value": float(p),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }
