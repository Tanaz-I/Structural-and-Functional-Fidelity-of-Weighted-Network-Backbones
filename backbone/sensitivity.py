"""Sensitivity of the AUC ranking to choices the paper leaves open.

Settings, all compared with the base run (beta target 20, 100 SI runs):
    beta_40, beta_80    SI timescale: half-infection time of the full graph set to 40 / 80 steps (beta capped at its
                        largest valid value where the target is unreachable, as in the base run)
    runs_300            three times as many SI runs (lower noise floor)
    dispfilt_norm       DispFilt with per-node normalised weights (Serrano et al.) instead of the paper's formula
Each row is the AUC of rho_C, rho_t and xi^-1 for one method under one setting, with each method's own X-tilde
(`limit == "own"`) and with the metric-backbone size for every method (`limit == "metric"`).
rho_C does not depend on beta or the SI runs, so the base rho_C sweep is reused for those settings.
(SmD tie-breaking seeds were also checked: above the metric-backbone size all tied edges are kept whatever the order,
so they cannot change the AUC.)
"""
from pathlib import Path

import pandas as pd

from .sparsify import METHODS
from .spread import si_setup, si_sweep, x_grid
from .summary import auc_table, metric_backbone_X
from .sweep import Network, connectivity_limits, sweep

BETA_TARGETS = {"beta_40": 40.0, "beta_80": 80.0}
SETTINGS = ["base", *BETA_TARGETS, "runs_300", "dispfilt_norm"]
MEASURES = ["AUC_rho_C", "AUC_rho_t", "AUC_xi"]


def _both_limits(rho_df, si_df, x_metric, limits, methods):
    """AUC table with each method's own X-tilde and with the common metric-backbone limit."""
    out = []
    for label, kw in [("own", {"limits": limits}), ("metric", {"fixed_limit": x_metric})]:
        out.append(auc_table(rho_df, si_df, x_metric, methods=methods, **kw).assign(limit=label))
    return pd.concat(out, ignore_index=True)


def run_setting(net, setting, runs=100, base_rho=None):
    """AUC rows (method, X_tilde, AUC_*, limit) for one setting."""
    k_metric = int((net.table.distortion <= 1 + 1e-9).sum())
    x_metric = metric_backbone_X(net.table, net.n)
    methods, target, n_runs = list(METHODS), 20.0, runs
    rho_df = base_rho if base_rho is not None else sweep(net)
    if setting in BETA_TARGETS:
        target = BETA_TARGETS[setting]
    elif setting == "runs_300":
        n_runs = 3 * runs
    elif setting == "dispfilt_norm":
        methods = ["DispFiltNorm"]
        rho_df = sweep(net, methods=methods)
    elif setting != "base":
        raise ValueError(f"unknown setting {setting!r}")
    limits = connectivity_limits(net, methods=methods)
    setup = si_setup(net, target_t_half=target, runs=n_runs, seed=0)
    si_df = si_sweep(net, methods=methods, ks=x_grid(net, 21), extra_ks=[k_metric], runs=n_runs, setup=setup)
    res = _both_limits(rho_df, si_df, x_metric, limits, methods)
    res.insert(0, "setting", setting)
    res["beta"] = setup[1]
    return res


def run(net_dir, out_dir=None, runs=100, log=print):
    """All settings for one network (`net_dir` = results/<network>); each setting is cached as its own CSV so an
    interrupted run can resume. Returns the combined table and writes it to out_dir/sensitivity.csv."""
    net_dir = Path(net_dir)
    out = Path(out_dir or net_dir / "sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    net = Network(pd.read_csv(net_dir / "edge_scores.csv"))
    base_rho = pd.read_csv(net_dir / "sweep_rhoC.csv")
    parts = []
    for s in SETTINGS:
        f = out / f"{s}.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
            continue
        df = run_setting(net, s, runs=runs, base_rho=base_rho)
        df.to_csv(f, index=False)
        parts.append(df)
        log(f"  sensitivity: {s} done")
    allr = pd.concat(parts, ignore_index=True)
    allr.to_csv(out / "sensitivity.csv", index=False)
    return allr


def ranking_table(df, limit="metric"):
    """Method order (best first) per setting and measure, so changes in the ranking are visible at a glance."""
    rows = []
    for (setting, lim), g in df.groupby(["setting", "limit"]):
        if lim != limit:
            continue
        for m in MEASURES:
            rows.append((setting, m, " > ".join(g.sort_values(m, ascending=False).method)))
    return pd.DataFrame(rows, columns=["setting", "measure", "order"])


def dispfilt_variants(df, limit="metric"):
    """DispFilt (paper formula) against the per-node normalised variant, base beta and SI runs."""
    d = df[(df.limit == limit) & (df.setting.isin(["base", "dispfilt_norm"])) &
           (df.method.isin(["DispFilt", "DispFiltNorm"]))]
    return d[["method", *MEASURES]].reset_index(drop=True)


def spread_over_settings(df, limit="metric"):
    """Largest minus smallest AUC of each method across the beta and run-count settings (a robustness band)."""
    d = df[(df.limit == limit) & (df.setting.isin(["base", *BETA_TARGETS, "runs_300"]))]
    g = d.groupby("method")[MEASURES]
    return (g.max() - g.min()).reset_index()
