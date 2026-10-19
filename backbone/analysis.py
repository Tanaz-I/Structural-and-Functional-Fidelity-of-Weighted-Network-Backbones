"""Result tables, bootstrap, statistics and the E1/E3/E4 analyses (agreement, mechanisms, contrasts)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, kendalltau, wilcoxon

# ---------------------------------------------------------------- tables
FUNCS = ["spread", "comm", "path", "rank"]  # functional fidelities; "struct" is the structural one
RESULTS = Path(__file__).resolve().parents[1] / "results" / "eval"
NEAR_CEILING = 0.05
EXTRA = ("connected", "n_components", "bridge_ratio", "hub_concentration", "tie_share",
         "median_stretch", "disconnected_pair_share")


def load_results(names=None, tag: str = "", results_dir: Path = RESULTS) -> pd.DataFrame:
    if names:
        files = [results_dir / f"{n}{tag}.parquet" for n in names]
    else:
        files = sorted(results_dir.glob(f"*{tag}.parquet"))
        if not tag:
            files = [f for f in files if not f.stem.endswith("_quick") and "smoke" not in f.stem]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (network, kind, name, draw, q) with a column per function."""
    keys = ["network", "kind", "name", "draw", "q"]
    base = df.pivot_table(index=keys, columns="function", values="score").reset_index()
    meta = df.drop_duplicates(keys)[keys + [c for c in EXTRA if c in df.columns]]
    return base.merge(meta, on=keys)


def method_table(w: pd.DataFrame) -> pd.DataFrame:
    return w[w.kind == "method"].drop(columns=["kind", "draw"]).rename(columns={"name": "method"}).reset_index(drop=True)


def null_means(w: pd.DataFrame) -> pd.DataFrame:
    nl = w[w.kind == "null"]
    g = nl.groupby(["network", "name", "q"])[["struct"] + FUNCS].mean().reset_index()
    return g.rename(columns={"name": "null"})


def skill_table(w: pd.DataFrame) -> pd.DataFrame:
    """Skill of each method versus each null, per function: (F - F_rand) / (1 - F_rand).

    Where 1 - F_rand < 0.05 the plain difference is reported and near_ceiling is set.
    """
    mm = method_table(w).merge(null_means(w), on=["network", "q"], suffixes=("", "_null"))
    rows = []
    for f in ["struct"] + FUNCS:
        den = 1.0 - mm[f + "_null"]
        ok = den >= NEAR_CEILING
        skill = np.where(ok, (mm[f] - mm[f + "_null"]) / den.where(ok, 1.0), mm[f] - mm[f + "_null"])
        rows.append(mm[["network", "method", "q", "null"]].assign(
            function=f, score=mm[f], null_score=mm[f + "_null"], skill=skill, near_ceiling=~ok))
    return pd.concat(rows, ignore_index=True)


def gap_table(m: pd.DataFrame, tau_s: float = 0.8, tau_f: float = 0.6) -> pd.DataFrame:
    """gap = F_struct - F_g; failure = structure passes (>= tau_s) while function fails (< tau_f)."""
    rows = []
    for f in FUNCS:
        d = m[["network", "method", "q", "struct", f, "bridge_ratio", "hub_concentration", "connected"]].copy()
        d = d.rename(columns={f: "F_g"})
        d["function"] = f
        d["gap"] = d["struct"] - d["F_g"]
        d["failure"] = (d["struct"] >= tau_s) & (d["F_g"] < tau_f)
        d["reverse_failure"] = (d["struct"] < tau_f) & (d["F_g"] >= tau_s)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- bootstrap
def percentile_ci(samples, level: float = 0.95):
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def cluster_bootstrap(groups: dict, stat, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """groups: {cluster id -> 1-D array of values}; stat(list of arrays) -> float.

    Returns (point estimate, lo, hi, n_clusters). Clusters are resampled with replacement.
    """
    keys = list(groups)
    rng = np.random.default_rng(seed)
    point = stat([groups[k] for k in keys])
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        boots[b] = stat([groups[keys[i]] for i in pick])
    lo, hi = percentile_ci(boots, level)
    return float(point), lo, hi, len(keys)


def mean_of_concat(arrs) -> float:
    return float(np.mean(np.concatenate(arrs)))


# ---------------------------------------------------------------- stats
def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def fit_mixed(formula: str, data: pd.DataFrame, group: str = "network"):
    """Random-intercept model by maximum likelihood (so likelihood-ratio tests are valid)."""
    last = None
    for method in ("lbfgs", "powell", "nm", "bfgs"):  # a boundary variance can make one optimiser's Hessian singular
        try:
            return smf.mixedlm(formula, data, groups=data[group]).fit(reml=False, method=method)
        except np.linalg.LinAlgError as err:
            last = err
    raise last


def lr_test(full, reduced) -> tuple[float, float]:
    """Likelihood-ratio statistic and p-value (chi-square, df = difference in fixed parameters)."""
    stat = max(0.0, 2.0 * (full.llf - reduced.llf))
    df = len(full.fe_params) - len(reduced.fe_params)
    return float(stat), float(chi2.sf(stat, df)) if df > 0 else float("nan")


# ---------------------------------------------------------------- agreement
def kendall_per_cell(m: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau-b between F_struct and F_g across methods, for each (network, q, function)."""
    rows = []
    for (net, q), g in m.groupby(["network", "q"]):
        for f in FUNCS:
            if g["struct"].nunique() < 2 or g[f].nunique() < 2:
                tau = np.nan  # a constant ranking has no defined agreement
            else:
                tau = kendalltau(g["struct"], g[f], variant="b").statistic
            rows.append({"network": net, "q": q, "function": f, "tau": tau})
    return pd.DataFrame(rows)


def agreement_summary(m: pd.DataFrame, n_boot: int = 1000, seed: int = 0, threshold: float = 0.6,
                      level: float = 0.95) -> pd.DataFrame:
    """Mean tau per function with cluster-bootstrap CI over networks; RQ1 flag per function."""
    cells = kendall_per_cell(m).dropna(subset=["tau"])
    rows = []
    for f, g in cells.groupby("function"):
        groups = {n: d["tau"].to_numpy() for n, d in g.groupby("network")}
        mean, lo, hi, k = cluster_bootstrap(groups, mean_of_concat, n_boot, seed, level)
        rows.append({"function": f, "mean_tau": mean, "ci_lo": lo, "ci_hi": hi, "n_networks": k,
                     "n_cells": len(g), "below_threshold": hi < threshold})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- mechanisms
def bridge_regression(gap: pd.DataFrame, functions=("spread", "comm")) -> pd.DataFrame:
    """H3 (associational, real networks): gap ~ bridge_ratio + hub_concentration + q + method + (1|network).

    Predicted sign: negative coefficient on bridge_ratio (fewer bridges kept relative to intra edges ->
    larger gap). The likelihood-ratio test compares against the model with method and budget only.
    Holm correction runs across the functions tested.
    """
    rows = []
    for f in functions:
        d = gap[(gap.function == f)].dropna(subset=["gap", "bridge_ratio", "hub_concentration"]).rename(
            columns={"q": "budget"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = fit_mixed("gap ~ bridge_ratio + hub_concentration + budget + C(method)", d)
            red = fit_mixed("gap ~ budget + C(method)", d)
        stat, p_lr = lr_test(full, red)
        ci = full.conf_int().loc["bridge_ratio"]
        rows.append({"function": f, "coef_bridge_ratio": float(full.params["bridge_ratio"]),
                     "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p_coef": float(full.pvalues["bridge_ratio"]),
                     "coef_hub": float(full.params["hub_concentration"]), "lr_stat": stat, "p_lr": p_lr,
                     "n_networks": d.network.nunique(), "n_obs": len(d)})
    out = pd.DataFrame(rows)
    out["p_lr_holm"] = holm(out.p_lr)
    out["predicted_sign"] = out.coef_bridge_ratio < 0
    out["supported"] = out.predicted_sign & (out.p_lr_holm < 0.05)
    return out


def weight_heterogeneity_model(gap: pd.DataFrame, meta: pd.DataFrame, function: str = "spread",
                               weight_based=("W", "DF"), reference=("SMDS",)) -> pd.DataFrame:
    """RQ3 (causal, synthetic factorial): do weight-based filters degrade faster with weight spread/coupling?

    gap ~ wb * (sigma + coupling + mixing) + q + (1|network), on weight-based and reference
    methods only. `meta` has one row per network with mixing, sigma, coupling. A positive interaction
    wb:sigma or wb:coupling means weight-based filters lose more fidelity than the reference as that
    factor grows.
    """
    d = gap[(gap.function == function) & gap.method.isin(list(weight_based) + list(reference))].merge(
        meta, on="network").rename(columns={"q": "budget"})
    d["wb"] = d.method.isin(weight_based).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_mixed("gap ~ wb * (sigma + coupling + mixing) + budget", d)
    ci = fit.conf_int()
    rows = []
    for term in ("wb:sigma", "wb:coupling", "wb:mixing"):
        rows.append({"function": function, "term": term, "coef": float(fit.params[term]),
                     "ci_lo": float(ci.loc[term, 0]), "ci_hi": float(ci.loc[term, 1]),
                     "p": float(fit.pvalues[term]), "n_networks": d.network.nunique()})
    out = pd.DataFrame(rows)
    out["positive_excludes_zero"] = out.ci_lo > 0
    return out


# ---------------------------------------------------------------- contrasts
def winners(m: pd.DataFrame) -> pd.DataFrame:
    """Best method per (network, q, function); ties go to the first method alphabetically (reported via n_tied)."""
    rows = []
    for (net, q), g in m.groupby(["network", "q"]):
        for f in FUNCS:
            top = g[f].max()
            best = sorted(g.loc[g[f] >= top - 1e-12, "method"])
            rows.append({"network": net, "q": q, "function": f, "winner": best[0], "n_tied": len(best)})
    return pd.DataFrame(rows)


def function_dependence(win: pd.DataFrame, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """RQ4: share of cells where >= 2 functions have different best methods; cluster-bootstrap over networks."""
    cell = win.groupby(["network", "q"]).winner.nunique().reset_index(name="n_winners")
    cell["differs"] = (cell.n_winners >= 2).astype(float)
    groups = {n: d["differs"].to_numpy() for n, d in cell.groupby("network")}
    share, lo, hi, k = cluster_bootstrap(groups, mean_of_concat, n_boot, seed, level)
    return {"share": share, "ci_lo": lo, "ci_hi": hi, "n_networks": k, "n_cells": len(cell),
            "supported": lo > 0.5}


def leaderboard(skill: pd.DataFrame, null: str = "R-W") -> pd.DataFrame:
    """Mean skill (versus the chosen null) per method and function, with cluster-bootstrap CI over networks."""
    s = skill[(skill.null == null) & (skill.function != "struct")]
    rows = []
    for (meth, f), g in s.groupby(["method", "function"]):
        groups = {n: d["skill"].to_numpy() for n, d in g.groupby("network")}
        mean, lo, hi, k = cluster_bootstrap(groups, mean_of_concat, 500, 0)
        rows.append({"method": meth, "function": f, "mean_skill": mean, "ci_lo": lo, "ci_hi": hi, "n_networks": k})
    return pd.DataFrame(rows)


def pathbased_vs_weight_filters(m: pd.DataFrame, skill: pd.DataFrame, challengers=("SMDS", "HSS"), baselines=("W", "DF"),
                                functions=("spread",), null: str = "R-W") -> pd.DataFrame:
    """Paper H1: at the same budget, each shortest-path-based method has a smaller structure-to-function gap and a
    higher skill score than each weight-based baseline. Paired one-sided Wilcoxon over (network, q) cells;
    Holm across all tests. Pass contact networks only to test the hypothesis as stated."""
    rows = []
    for f in functions:
        piv = m.pivot_table(index=["network", "q"], columns="method", values=["struct", f])
        sk = skill[(skill.null == null) & (skill.function == f)].pivot_table(
            index=["network", "q"], columns="method", values="skill")
        for chal in challengers:
            for base in baselines:
                gap_c, gap_b = piv["struct"][chal] - piv[f][chal], piv["struct"][base] - piv[f][base]
                for name, diff in (("gap", gap_b - gap_c), ("skill", sk[chal] - sk[base])):
                    diff = diff.dropna()
                    p = wilcoxon(diff, alternative="greater").pvalue if len(diff) >= 6 and np.any(diff != 0) else float("nan")
                    rows.append({"challenger": chal, "baseline": base, "function": f, "measure": name,
                                 "median_diff": float(diff.median()), "n_cells": len(diff),
                                 "n_networks": diff.index.get_level_values(0).nunique(), "p": p})
    out = pd.DataFrame(rows)
    ok = out.p.notna()
    out["p_holm"] = np.nan
    out.loc[ok, "p_holm"] = holm(out.loc[ok, "p"])
    out["supported"] = out.p_holm < 0.05
    return out


def metric_backbone_communities(m: pd.DataFrame, metric_share: dict, tau_f: float, n_boot: int = 1000, seed: int = 0,
                                level: float = 0.95) -> dict:
    """Paper H2: the metric backbone preserves community structure. For each network take SMDS at the smallest
    budget on the grid that holds the whole metric backbone and read its F_comm; cluster-bootstrap over networks."""
    groups = {}
    for net, g in m[m.method == "SMDS"].groupby("network"):
        q = g.q[g.q >= metric_share.get(net, 1.0) - 1e-9]
        if len(q):
            groups[net] = g.loc[g.q == q.min(), "comm"].to_numpy()
    if not groups:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_networks": 0, "threshold": tau_f}
    mean, lo, hi, k = cluster_bootstrap(groups, mean_of_concat, n_boot, seed, level)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_networks": k, "threshold": tau_f}
