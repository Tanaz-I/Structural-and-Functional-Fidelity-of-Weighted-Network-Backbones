"""Extension: why do W and DispFilt lose spreading fidelity, and what does a distance backbone add?

Diagnosis (backbone/diagnose.py) found that W and DispFilt fragment the graph and that the number of components tracks their
loss of spreading fidelity, which rho_C does not see. Three tests, all on the same budgets X, seeds, beta and reference runs
as the base replication, so the numbers are comparable with DBS, SmD and wEffRes:

1. Connectivity fix. M = method as in the paper; M+T = a spanning tree built along M's own ranking first, then M's top
   edges up to the budget; M+R = the same with a random spanning tree (control: connectivity or the choice of tree?).
2. Shortest paths. M+B forces the metric backbone (connected, every shortest path preserved) first, then M's order; M+X forces
   the same number of random edges instead (control).
3. Bridges (causal). Keep the metric backbone and fill the rest of the budget from M's ranking restricted to between-community
   edges first, within-community edges first, or random edges. Louvain partitions of the original graph define "between".

Headline number, per measure: the share of the gap between M and DBS that a variant closes,
    (AUC(variant) - AUC(M)) / (AUC(DBS) - AUC(M)),   AUC over [X0, 1], X0 = metric-backbone size ('metric') or 0 ('zero').
Verdict rules fixed before the runs (NOISE = 0.02 is the largest spread of the random-tree controls on xi^-1):
    fix     'supported' if M+T closes >= 50% of the gap on rho_t and xi^-1 and is within NOISE of (or above) M+R
    paths   'supported' if M+B closes >= 50% of the remaining xi^-1 gap between M+T and DBS and beats M+X by more than NOISE
    bridges 'supported' if the between-community fill closes >= 50% of the xi^-1 gap between M+B and DBS and beats the
            within-community and random fills by more than NOISE
    otherwise 'partial' if the variant gains more than NOISE over its starting point, else 'not supported'.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnose import louvain_labels
from .sparsify import ranking, spanning_tree_edges
from .spread import si_setup, si_sweep, x_grid
from .summary import auc_table, metric_backbone_X
from .sweep import Network, sweep

BASE = ["W", "DispFilt"]
FIXED = ["W+T", "DispFilt+T"]
CONTROLS = ["W+R", "DispFilt+R"]
PATHS = ["W+B", "DispFilt+B"]
PATH_CONTROLS = ["W+X", "DispFilt+X"]
POOLS = ["between", "within", "random"]
RANDOM_SEEDS = [0, 1, 2]
N_PARTITIONS = 3
MEASURES = ["AUC_rho_C", "AUC_rho_t", "AUC_xi"]
NOISE = 0.02


def run_curves(net, methods, setup, seed=0, orders=None):
    """rho_C and SI curves for `methods` on the shared X grid (plus the metric-backbone size)."""
    k_metric = int((net.table.distortion <= 1 + 1e-9).sum())
    rho_df = sweep(net, methods=methods, seed=seed, orders=orders)
    si_df = si_sweep(net, methods=methods, ks=x_grid(net, 21), extra_ks=[k_metric], runs=setup_runs(setup), seed=seed,
                     setup=setup, orders=orders)
    return rho_df, si_df


def setup_runs(setup):
    """Number of SI runs per seed used by the shared setup (rows of the reference runs)."""
    return next(iter(setup[2].values())).shape[0]


def auc_both_limits(rho_df, si_df, x_metric, methods):
    """AUC rows with the lower limit at the metric-backbone size ('metric') and at the spanning tree ('zero')."""
    out = []
    for label, lo in [("metric", x_metric), ("zero", 0.0)]:
        out.append(auc_table(rho_df, si_df, x_metric, fixed_limit=lo, methods=methods).assign(limit=label))
    return pd.concat(out, ignore_index=True)


def gap_closed(auc, base, fixed, reference="DBS", limit="metric"):
    """Share of the gap AUC(reference) - AUC(base) closed by `fixed`, per measure. `auc` has a `limit` column."""
    a = auc[auc.limit == limit].set_index("method")
    rows = []
    for m in ["AUC_rho_t", "AUC_xi", "AUC_rho_C"]:
        gap = a.loc[reference, m] - a.loc[base, m]
        gain = a.loc[fixed, m] - a.loc[base, m]
        rows.append((base, fixed, m, a.loc[base, m], a.loc[fixed, m], a.loc[reference, m], gain,
                     gain / gap if abs(gap) > NOISE else np.nan))
    return pd.DataFrame(rows, columns=["base", "fixed", "measure", "AUC_base", "AUC_fixed", "AUC_reference",
                                       "gain", "gap_closed"])


def verdict(gaps, control_gain):
    """Connectivity fix: 'supported' if M+T closes >= 50% of the gap on rho_t and xi^-1 and is within NOISE of the
    random-tree control (`control_gain`: measure -> gain of M+R); 'partial' if it gains more than NOISE on both."""
    g = gaps.set_index("measure")
    closes = all(g.loc[m, "gap_closed"] >= 0.5 for m in ["AUC_rho_t", "AUC_xi"])
    helps = all(g.loc[m, "gain"] > NOISE for m in ["AUC_rho_t", "AUC_xi"])
    beats_control = all(g.loc[m, "gain"] >= control_gain[m] - NOISE for m in ["AUC_rho_t", "AUC_xi"])
    return "supported" if closes and beats_control else "partial" if helps else "not supported"


def path_verdict(a, base, limit="metric", reference="DBS"):
    """Shortest-path test. `a` is the AUC table indexed by (method, limit)."""
    m = "AUC_xi"
    t, b = a.loc[(f"{base}+T", limit), m], a.loc[(f"{base}+B", limit), m]
    x = a.loc[(f"{base}+X", limit), m]
    gap = a.loc[(reference, limit), m] - t
    closed = (b - t) / gap if abs(gap) > NOISE else float("nan")
    return {"test": "paths", "base": base, "limit": limit, "xi_start": t, "xi_variant": b, "xi_control": x,
            "xi_DBS": a.loc[(reference, limit), m], "gap_closed": closed,
            "verdict": "supported" if closed >= 0.5 and b - x > NOISE else "partial" if b - t > NOISE else "not supported"}


def fill_order(net, base_order, labels, pool, rng=None):
    """Metric backbone first (spanning tree inside it, then the other metric edges), then the remaining edges in
    `base_order` but with the `pool` edges first: 'between' = edges joining two communities, 'within' = edges inside one,
    'random' = a random order. The budget is filled from the pool, then from the rest."""
    t = net.table
    is_metric = (t["distortion"] <= 1 + 1e-9).to_numpy()
    tree = spanning_tree_edges(t, base_order[is_metric[base_order]])
    in_tree = set(tree)
    forced = tree + [int(e) for e in base_order if is_metric[e] and e not in in_tree]
    taken = np.zeros(net.m, dtype=bool)
    taken[forced] = True
    rest = base_order[~taken[base_order]]
    if pool == "random":
        rest = rng.permutation(rest)
    else:
        between = (labels[net.i] != labels[net.j])[rest]
        first = between if pool == "between" else ~between
        rest = np.concatenate([rest[first], rest[~first]])
    return np.concatenate([np.array(forced, dtype=int), rest])


def bridge_verdict(a, base, limit="metric", reference="DBS"):
    """Causal bridge test. `a` has rows M+B, M|between, M|within, M|random and the reference, indexed by (method, limit)."""
    m = "AUC_xi"
    b, bt = a.loc[(f"{base}+B", limit), m], a.loc[(f"{base}|between", limit), m]
    wi, ra = a.loc[(f"{base}|within", limit), m], a.loc[(f"{base}|random", limit), m]
    gap = a.loc[(reference, limit), m] - b
    closed = (bt - b) / gap if abs(gap) > NOISE else float("nan")
    ok = closed >= 0.5 and bt - max(wi, ra) > NOISE
    return {"test": "bridges", "base": base, "limit": limit, "xi_start": b, "xi_variant": bt,
            "xi_control": max(wi, ra), "xi_DBS": a.loc[(reference, limit), m], "gap_closed": closed,
            "verdict": "supported" if ok else "partial" if bt - b > NOISE else "not supported"}


def _mean_std(rows):
    allr = pd.concat(rows, ignore_index=True)
    mean = allr.groupby(["method", "limit"])[["X_tilde", *MEASURES]].mean().reset_index()
    std = allr.groupby(["method", "limit"])[MEASURES].std().add_suffix("_std").reset_index()
    return mean.merge(std, on=["method", "limit"], how="left")


def run(net_dir, out_dir=None, runs=100, log=print):
    """Run the three tests for one network. `net_dir` = results/<network> from the base replication (edge_scores.csv,
    sweep_rhoC.csv, sweep_si.csv); outputs go to out_dir (default net_dir/extension)."""
    net_dir = Path(net_dir)
    out = Path(out_dir or net_dir / "extension")
    out.mkdir(parents=True, exist_ok=True)
    net = Network(pd.read_csv(net_dir / "edge_scores.csv"))
    x_metric = metric_backbone_X(net.table, net.n)
    setup = si_setup(net, target_t_half=20.0, runs=runs, seed=0)

    # reference rows from the base run
    ref = auc_both_limits(pd.read_csv(net_dir / "sweep_rhoC.csv"), pd.read_csv(net_dir / "sweep_si.csv"), x_metric,
                          ["DBS", "SmD", "wEffRes"])

    # 1. connectivity fix
    rho, si = run_curves(net, BASE + FIXED, setup)
    curves = [(rho, si)]
    rows = [auc_both_limits(rho, si, x_metric, BASE + FIXED)]
    rows_ctrl = []
    for s in RANDOM_SEEDS:
        r, sd = run_curves(net, CONTROLS, setup, seed=s)
        rows_ctrl.append(auc_both_limits(r, sd, x_metric, CONTROLS).assign(tree_seed=s))
    ctrl = _mean_std(rows_ctrl)
    log("  extension 1/3 connectivity fix done")

    # 2. shortest paths
    rho, si = run_curves(net, PATHS, setup)
    curves.append((rho, si))
    pd.concat([c[0] for c in curves], ignore_index=True).to_csv(out / "curves_rhoC.csv", index=False)  # for the dashboard
    pd.concat([c[1] for c in curves], ignore_index=True).to_csv(out / "curves_si.csv", index=False)
    rows.append(auc_both_limits(rho, si, x_metric, PATHS))
    rows_x = []
    for s in RANDOM_SEEDS:
        r, sd = run_curves(net, PATH_CONTROLS, setup, seed=s)
        rows_x.append(auc_both_limits(r, sd, x_metric, PATH_CONTROLS).assign(tree_seed=s))
    ctrl_x = _mean_std(rows_x)
    log("  extension 2/3 shortest-path test done")

    # 3. bridges (causal)
    labels = louvain_labels(net, N_PARTITIONS)
    rows_b = []
    for b in BASE:
        base_order = ranking(net.table, b)
        for pool in POOLS:
            for s in range(N_PARTITIONS):
                name = f"{b}|{pool}|{s}"
                orders = {name: fill_order(net, base_order, labels[s], pool, np.random.default_rng(100 + s))}
                r, sd = run_curves(net, [name], setup, orders=orders)
                t = auc_both_limits(r, sd, x_metric, [name])
                t["method"] = f"{b}|{pool}"
                rows_b.append(t.assign(seed=s))
    bridges = _mean_std(rows_b)
    log("  extension 3/3 bridge test done")

    main = pd.concat(rows, ignore_index=True)
    auc = pd.concat([ref, main, ctrl, ctrl_x, bridges], ignore_index=True)
    auc.to_csv(out / "auc.csv", index=False)

    gaps, verdicts = [], []
    ctable = auc[["method", "limit", *MEASURES]].set_index(["method", "limit"])
    for limit in ["metric", "zero"]:
        for b, f, c in zip(BASE, FIXED, CONTROLS):
            g = gap_closed(auc, b, f, limit=limit).assign(limit=limit)
            gc = gap_closed(auc, b, c, limit=limit).set_index("measure").gain.to_dict()
            gaps.append(g)
            row = g.set_index("measure")
            verdicts.append({"test": "fix", "base": b, "limit": limit,
                             "xi_start": row.loc["AUC_xi", "AUC_base"], "xi_variant": row.loc["AUC_xi", "AUC_fixed"],
                             "xi_control": ctable.loc[(c, limit), "AUC_xi"], "xi_DBS": row.loc["AUC_xi", "AUC_reference"],
                             "gap_closed": row.loc["AUC_xi", "gap_closed"], "verdict": verdict(g, gc)})
            verdicts.append(path_verdict(ctable, b, limit))
            verdicts.append(bridge_verdict(ctable, b, limit))
    pd.concat(gaps, ignore_index=True).to_csv(out / "gap_closed.csv", index=False)
    res = pd.DataFrame(verdicts)
    res.to_csv(out / "verdicts.csv", index=False)
    return auc, res
