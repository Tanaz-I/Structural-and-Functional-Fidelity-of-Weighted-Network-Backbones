"""Summaries of the sweeps: area under the curve (Eq. 16) and the optimal DBS parameter (Eq. 17, Table 3)."""
import numpy as np
import pandas as pd

from .backbones import TOL, adjusted_size
from .sparsify import METHODS, ranking
from .spread import spreading_fidelity
from .sweep import eigenvector_centrality, rho_C

SCENARIOS = {  # Table 3: (w_C, w_t, w_xi, w_X)
    "Balanced": (.25, .25, .25, .25),
    "Balanced Except xi": (1 / 3, 1 / 3, 0, 1 / 3),
    "Balanced Except rho_t": (1 / 3, 0, 1 / 3, 1 / 3),
    "Balanced Except rho_C": (0, 1 / 3, 1 / 3, 1 / 3),
    "Balanced Except X": (1 / 3, 1 / 3, 1 / 3, 0),
    "Slight Preference xi": (.2, .2, .4, .2),
    "Slight Preference rho_t": (.2, .4, .2, .2),
    "Slight Preference rho_C": (.4, .2, .2, .2),
    "Slight Preference X": (.2, .2, .2, .4),
    "Strong Preference xi": (.1, .1, .7, .1),
    "Strong Preference rho_t": (.1, .7, .1, .1),
    "Strong Preference rho_C": (.7, .1, .1, .1),
    "Strong Preference X": (.1, .1, .1, .7),
    "Mainly xi": (.02, .02, .94, .02),
    "Mainly rho_t": (.02, .94, .02, .02),
    "Mainly rho_C": (.94, .02, .02, .02),
    "Mainly X": (.02, .02, .02, .94),
}


def metric_backbone_X(table, n):
    return adjusted_size(n, len(table), int((table["distortion"] <= 1 + TOL).sum()))


def x_tilde(x_conn, x_metric):
    """Lower integration limit: the method's connectivity limit (from sweep.connectivity_limits), but never below the
    metric-backbone size (paper Sec. 5.3.3: for DBS the limit would be ~0, so |B_m| is used; the same floor is
    applied to all methods)."""
    return max(x_conn, x_metric)


def auc(X, y, lo):
    """(1 / (1 - lo)) * integral_lo^1 y dX by the trapezoid rule, linearly interpolating the value at `lo`."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    o = np.argsort(X)
    X, y = X[o], y[o]
    if lo >= 1:
        return float("nan")
    xs = np.concatenate([[lo], X[X > lo]])
    ys = np.concatenate([[np.interp(lo, X, y)], y[X > lo]])
    return float(np.trapezoid(ys, xs) / (1 - lo))


def auc_table(rho_df, si_df, x_metric, limits=None, fixed_limit=None):
    """AUC of rho_C, rho_t and xi^-1_0.5 per method. With `limits` (method -> connectivity limit) each method uses
    its own X-tilde; with `fixed_limit` every method uses that same limit."""
    rows = []
    for m in METHODS:
        lo = fixed_limit if fixed_limit is not None else x_tilde(limits[m], x_metric)
        c, s = rho_df[rho_df.method == m], si_df[si_df.method == m]
        rows.append((m, lo, auc(c.X, c.rho_C, lo), auc(s.X, s.rho_t, lo), auc(s.X, s.xi, lo)))
    return pd.DataFrame(rows, columns=["method", "X_tilde", "AUC_rho_C", "AUC_rho_t", "AUC_xi"])


def lambda_grid(n_points=48):
    return sorted(set(np.geomspace(0.1, 20, n_points)) | {0.3, 1 / 3, 1.0})


def lambda_sweep(net, seeds, beta, ref, lams, runs, seed=0):
    """Metrics of the DBS backbone B_lam for each lam: X, rho_C, rho_t, xi^-1_0.5 (+ the MST as lam = inf)."""
    order = ranking(net.table, "DBS", seed)
    lam_ij = net.table["lam"].to_numpy()
    c_full = eigenvector_centrality(net.adjacency())
    rows = []
    for lam in list(lams) + [np.inf]:
        k = int((lam_ij >= lam * (1 - TOL)).sum())
        keep = order[:k]
        rho, xi = spreading_fidelity(net.adjacency(keep), seeds, beta, ref, runs, rng_seed=777_000 + k)
        rows.append((lam, k, adjusted_size(net.n, net.m, k),
                     rho_C(c_full, eigenvector_centrality(net.adjacency(keep))), rho.mean(), xi.mean()))
    return pd.DataFrame(rows, columns=["lam", "n_edges", "X", "rho_C", "rho_t", "xi"])


def optimal_lambda(lam_df, scenarios=SCENARIOS):
    """For each scenario, the lam maximising w_C rho_C + w_t rho_t + w_xi xi + w_X (1 - X) (Eq. 17)."""
    rows = []
    for name, (wc, wt, wx, wX) in scenarios.items():
        obj = wc * lam_df.rho_C + wt * lam_df.rho_t + wx * lam_df.xi + wX * (1 - lam_df.X)
        b = lam_df.loc[obj.idxmax()]
        rows.append((name, wc, wt, wx, wX, b.lam, b.n_edges, b.X, b.rho_C, b.rho_t, b.xi, obj.max()))
    return pd.DataFrame(rows, columns=["scenario", "w_C", "w_t", "w_xi", "w_X", "lam_opt", "n_edges", "X",
                                       "rho_C", "rho_t", "xi", "objective"])


def cross_network_tests(auc_all):
    """DBS against each other method across networks, for each AUC measure.
    auc_all has columns network, method, AUC_rho_C, AUC_rho_t, AUC_xi.
    Mann-Whitney U (one-sided, DBS larger) is what the paper uses; the paired Wilcoxon signed-rank test
    (one-sided) uses the fact that the same networks are compared."""
    from scipy.stats import mannwhitneyu, wilcoxon

    rows = []
    for col in ["AUC_rho_C", "AUC_rho_t", "AUC_xi"]:
        wide = auc_all.pivot(index="network", columns="method", values=col)
        for m in METHODS:
            if m == "DBS":
                continue
            d = wide["DBS"] - wide[m]
            mw = mannwhitneyu(wide["DBS"], wide[m], alternative="greater").pvalue
            try:
                wx = wilcoxon(wide["DBS"], wide[m], alternative="greater").pvalue
            except ValueError:  # all differences zero
                wx = float("nan")
            rows.append((col, m, len(wide), wide["DBS"].median(), wide[m].median(),
                         float(d.mean()), int((d > 0).sum()), mw, wx))
    return pd.DataFrame(rows, columns=["measure", "versus", "n_networks", "median_DBS", "median_other",
                                       "mean_diff", "networks_DBS_better", "p_mannwhitney", "p_wilcoxon"])
