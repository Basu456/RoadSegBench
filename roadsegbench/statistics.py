import numpy as np
import pandas as pd
from scipy.stats import norm, friedmanchisquare

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0,1]")
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted

def paired_normal_comparisons(wide, model_columns, alpha=0.05):
    """
    wide: one row per paired seed/scene unit; one column per model.
    Uses paired differences and a normal approximation. This is not an
    exact permutation test. Zero differences are retained in n and variance.
    """
    rows = []
    for i, a in enumerate(model_columns):
        for b in model_columns[i+1:]:
            d = (wide[a] - wide[b]).dropna().to_numpy(dtype=float)
            n = len(d)
            if n < 2:
                z, p = np.nan, np.nan
            else:
                sd = d.std(ddof=1)
                if sd == 0:
                    z, p = (0.0, 1.0) if np.mean(d) == 0 else (np.inf, 0.0)
                else:
                    z = float(np.mean(d) / (sd / np.sqrt(n)))
                    p = float(2 * norm.sf(abs(z)))
            rows.append({"model_a": a, "model_b": b, "n_pairs": n,
                         "mean_difference": float(np.mean(d)) if n else np.nan,
                         "z_statistic": z, "p_raw_normal_approx": p})
    out = pd.DataFrame(rows)
    valid = out["p_raw_normal_approx"].notna()
    out["p_holm"] = np.nan
    out.loc[valid, "p_holm"] = holm_adjust(out.loc[valid, "p_raw_normal_approx"])
    out["significant_holm"] = out["p_holm"] < alpha
    return out

def nemenyi_cd_decisions(mean_ranks, critical_difference):
    names = list(mean_ranks)
    rows = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            diff = abs(float(mean_ranks[a]) - float(mean_ranks[b]))
            rows.append({"model_a": a, "model_b": b, "rank_difference": diff,
                         "critical_difference": critical_difference,
                         "significant": bool(diff > critical_difference)})
    return pd.DataFrame(rows)

def friedman_test(matrix):
    """matrix: DataFrame with paired blocks as rows and models as columns."""
    if matrix.isna().any().any():
        raise ValueError("Friedman input contains missing values; resolve pairing first.")
    return friedmanchisquare(*[matrix[c].to_numpy() for c in matrix.columns])
