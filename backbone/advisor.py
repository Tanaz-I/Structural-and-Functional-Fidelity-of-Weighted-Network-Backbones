"""Backbone Advisor, evidence mode: recommend a method for a network, goal weights and budget."""
from __future__ import annotations


import numpy as np
import pandas as pd
from . import analysis
from .analysis import FUNCS, NEAR_CEILING
from .pipeline import EvalConfig, evaluate_network

# ---------------------------------------------------------------- evidence
N_BOOT = 400
CLEAR_P = 0.8
PRACTICAL_TIE = 0.02


def bootstrap_means(samples: dict, methods: list[str], q: float, point: pd.DataFrame, n_boot: int = N_BOOT,
                    seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of F_{m,g}: array [method, function, n_boot].

    Functions with per-unit samples (spread, comm, path) are resampled with indices shared across
    methods. Rank is deterministic given tie-breaking, so its bootstrap is its point value.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((len(methods), len(FUNCS), n_boot))
    for gi, f in enumerate(FUNCS):
        key0 = f"{methods[0]}|{q:.2f}|{f}"
        if key0 in samples:
            n_units = len(samples[key0])
            idx = rng.integers(0, n_units, (n_boot, n_units))
            for mi, m in enumerate(methods):
                arr = np.asarray(samples[f"{m}|{q:.2f}|{f}"], dtype=float)
                out[mi, gi] = arr[idx].mean(1)
        else:
            for mi, m in enumerate(methods):
                out[mi, gi] = float(point.loc[m, f])
    return out


def pareto_front(F: np.ndarray) -> list[int]:
    """Indices of methods not dominated on all functions (higher is better)."""
    keep = []
    for i in range(len(F)):
        dominated = any(np.all(F[j] >= F[i]) and np.any(F[j] > F[i]) for j in range(len(F)) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def recommend(methods_q: pd.DataFrame, nulls_q: pd.DataFrame, samples: dict, q: float, weights: dict,
              require_connected: bool = False, neighbours: dict | None = None, seed: int = 0) -> dict:
    """methods_q: one row per method at budget q with columns struct, spread, comm, path, rank, connected,
    bridge_ratio, hub_concentration, n_components. nulls_q: rows indexed by null name (R-U, R-W) with the same
    score columns (means over draws). neighbours: {q: recommendation method} for the sensitivity check."""
    mq = methods_q.set_index("method")
    u = np.array([weights.get(f, 0.0) for f in FUNCS], dtype=float)
    if u.sum() <= 0:
        raise ValueError("objective weights must have a positive sum")
    u = u / u.sum()
    methods = [m for m in mq.index if (mq.connected[m] or not require_connected)]
    excluded = [m for m in mq.index if m not in methods]
    if not methods:
        raise ValueError("no method gives a connected backbone at this budget")

    boot = bootstrap_means(samples, methods, q, mq, seed=seed)            # [m, g, B]
    util_b = np.einsum("g,mgb->mb", u, boot)
    F = mq.loc[methods, FUNCS].to_numpy(float)
    util = F @ u
    lo, hi = np.quantile(util_b, [0.025, 0.975], axis=1)
    best_b = util_b.argmax(0)
    p_best = np.bincount(best_b, minlength=len(methods)) / util_b.shape[1]
    order = np.argsort(-util)
    top, second = order[0], (order[1] if len(order) > 1 else order[0])
    overlap = len(order) > 1 and hi[second] >= lo[top] and lo[top] <= hi[second] and lo[second] <= hi[top]
    clear = p_best[top] >= CLEAR_P
    tied = len(order) > 1 and util[top] - util[second] < PRACTICAL_TIE  # practically tied even if noise is small
    label = "close call" if tied else ("clear" if clear else ("close call" if overlap else "likely"))

    # trade-offs: what the recommended method gives up per function against the best method for it
    best_per_f = F.max(0)
    giveups = {f: {"best_method": methods[int(F[:, gi].argmax())], "best": float(best_per_f[gi]),
                   "recommended": float(F[top, gi]), "give_up": float(best_per_f[gi] - F[top, gi])}
               for gi, f in enumerate(FUNCS)}
    pareto = [methods[i] for i in pareto_front(F)]

    # evidence per method and function: score, CI, skill vs both nulls, gap, failure warning
    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975], axis=2)               # [m, g]
    evidence = {}
    for mi, m in enumerate(methods):
        per = {}
        for gi, f in enumerate(FUNCS):
            sk = {}
            for nl in nulls_q.index:
                den = 1.0 - nulls_q.loc[nl, f]
                sk[nl] = float((mq.loc[m, f] - nulls_q.loc[nl, f]) / den) if den >= NEAR_CEILING else float(mq.loc[m, f] - nulls_q.loc[nl, f])
            gap = float(mq.struct[m] - mq.loc[m, f])
            per[f] = {"score": float(mq.loc[m, f]), "ci": [float(ci_lo[mi, gi]), float(ci_hi[mi, gi])], "skill": sk,
                      "gap": gap, "failure_warning": bool(mq.struct[m] >= 0.8 and mq.loc[m, f] < 0.6)}
        evidence[m] = {"functions": per, "struct": float(mq.struct[m]), "connected": bool(mq.connected[m]),
                       "bridge_ratio": float(mq.bridge_ratio[m]), "hub_concentration": float(mq.hub_concentration[m]),
                       "utility": float(util[mi]), "utility_ci": [float(lo[mi]), float(hi[mi])], "p_best": float(p_best[mi])}
    warnings = []
    rec = methods[top]
    for f, e in evidence[rec]["functions"].items():
        if e["failure_warning"]:
            warnings.append(f"{rec} looks structurally fine (F_struct {evidence[rec]['struct']:.2f}) but {f} fidelity is only {e['score']:.2f}.")
        if e["skill"].get("R-W", 1.0) <= 0 and u[FUNCS.index(f)] > 0:
            warnings.append(f"{rec} is no better than weight-proportional random removal on {f}.")
    if excluded:
        warnings.append(f"Excluded because their backbone is disconnected at q={q:.2f}: {', '.join(excluded)}.")
    sens = None
    if neighbours:
        sens = {"neighbour_recommendations": neighbours, "stable": all(v == rec for v in neighbours.values())}
    return {"q": q, "weights": dict(zip(FUNCS, map(float, u))), "recommendation": rec, "label": label,
            "p_best": {m: float(p) for m, p in zip(methods, p_best)}, "ranking": [methods[i] for i in order],
            "pareto": pareto, "tradeoffs": giveups, "evidence": evidence, "warnings": warnings, "sensitivity": sens,
            "excluded": excluded, "require_connected": require_connected}


def recommend_for_network(name: str, q: float, weights: dict, require_connected: bool = False,
                          eval_dir=None, samples_dir=None, seed: int = 0) -> dict:
    """Preset networks: read cached method scores + samples (results/samples) and null means (results/eval)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "results"
    eval_dir = Path(eval_dir) if eval_dir else root / "eval"
    samples_dir = Path(samples_dir) if samples_dir else root / "samples"
    z = np.load(samples_dir / f"{name}.npz")
    samples = {k.replace("__", "|"): z[k] for k in z.files}
    meth = analysis.method_table(analysis.wide(pd.read_parquet(samples_dir / f"{name}_methods.parquet")))
    nul = analysis.null_means(analysis.wide(pd.read_parquet(eval_dir / f"{name}.parquet")))
    grid = sorted(meth.q.unique())
    q = min(grid, key=lambda g: abs(g - q))
    mq = meth[np.isclose(meth.q, q)]
    nq = nul[np.isclose(nul.q, q)].set_index("null")
    neigh = {}
    i = grid.index(q)
    for j in (i - 1, i + 1):
        if 0 <= j < len(grid):
            r = recommend(meth[np.isclose(meth.q, grid[j])], nul[np.isclose(nul.q, grid[j])].set_index("null"),
                          samples, grid[j], weights, require_connected, seed=seed)
            neigh[f"{grid[j]:.2f}"] = r["recommendation"]
    return recommend(mq, nq, samples, q, weights, require_connected, neigh, seed)


def recommend_for_live_network(net, q: float, weights: dict, require_connected: bool = False, seed: int = 0,
                               n_null_draws: int = 5) -> dict:
    """Uploaded network: run all six methods at q with reduced seeds/runs (the protocol allows this for live use)."""
    cfg = EvalConfig(budgets=(round(q, 2),), seed_frac=0.02, n_runs=30, n_runs_null=10, n_null_draws=n_null_draws,
                     louvain_seeds=10, seed=seed)
    df = evaluate_network(net, cfg, collect=True)
    w = analysis.wide(df)
    mq, nq = analysis.method_table(w), analysis.null_means(w).set_index("null")
    return recommend(mq, nq, df.attrs["samples"], round(q, 2), weights, require_connected, None, seed)


def to_markdown(rec: dict, network: str) -> str:
    L = [f"# Backbone Advisor report: {network}", "",
         f"- Budget q = {rec['q']:.2f}; objective weights: " + ", ".join(f"{f} {w:.2f}" for f, w in rec["weights"].items()),
         f"- **Recommendation: {rec['recommendation']}** ({rec['label']}; probability of being best "
         f"{rec['p_best'][rec['recommendation']]:.0%})", "", "## Utility by method", "| Method | Utility | 95% CI | P(best) |", "|---|---|---|---|"]
    for m in rec["ranking"]:
        e = rec["evidence"][m]
        L.append(f"| {m} | {e['utility']:.3f} | [{e['utility_ci'][0]:.3f}, {e['utility_ci'][1]:.3f}] | {e['p_best']:.0%} |")
    L += ["", f"Non-dominated methods (Pareto set): {', '.join(rec['pareto'])}", "", "## What the recommendation gives up"]
    for f, t in rec["tradeoffs"].items():
        L.append(f"- {f}: {rec['recommendation']} {t['recommended']:.2f} versus best {t['best_method']} {t['best']:.2f} (gives up {t['give_up']:.2f})")
    if rec["warnings"]:
        L += ["", "## Warnings", *[f"- {w}" for w in rec["warnings"]]]
    if rec["sensitivity"]:
        s = rec["sensitivity"]
        L += ["", f"## Sensitivity: neighbouring budgets recommend {s['neighbour_recommendations']} "
                  f"({'stable' if s['stable'] else 'changes'})"]
    return "\n".join(L) + "\n"
