"""Export cached results to JSON for the web UI (data contract in docs/PLAN.md)."""
from __future__ import annotations

from pathlib import Path
import base64
import json

import networkx as nx
import numpy as np
import pandas as pd
from . import analysis, metrics
from .advisor import bootstrap_means
from .filters import BUDGETS, FILTERS, edge_order, k_of, metric_backbone_mask, tie_permutation, top_k_mask
from .network import load_network

# ---------------------------------------------------------------- ui export
ROOT = Path(__file__).resolve().parents[1]
FUNCS5 = ["struct", "spread", "comm", "path", "rank"]
B_EXPORT = 200


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), default=_json_default), encoding="utf-8")


def _bits(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask).tobytes()).decode()


def layout(net, seed: int = 0) -> np.ndarray:
    g = nx.Graph()
    g.add_nodes_from(range(net.n))
    g.add_weighted_edges_from(zip(net.u.tolist(), net.v.tolist(), np.log1p(net.w).tolist()))
    pos = nx.spring_layout(g, weight="weight", seed=seed, iterations=200)
    xy = np.array([pos[i] for i in range(net.n)])
    xy -= xy.min(0)
    return xy / max(xy.max(), 1e-9)


def export_network(name: str, out_dir: Path, results_dir: Path | None = None, samples_dir: Path | None = None,
                   with_spread: bool = True) -> dict:
    results_dir = Path(results_dir) if results_dir else ROOT / "results" / "eval"
    samples_dir = Path(samples_dir) if samples_dir else ROOT / "results" / "samples"
    net = load_network(name)
    d = Path(out_dir) / name
    xy = layout(net)
    PG = metrics.louvain_partitions(net, None, range(1))
    comm = net.communities if net.communities is not None else PG[0]
    s = net.strength()
    _write(d / "graph.json", {
        "name": name, "n": net.n, "m": net.m, "domain": net.meta.get("domain", ""),
        "nodes": {"strength": np.round(s, 3), "community": comm, "x": np.round(xy[:, 0], 4), "y": np.round(xy[:, 1], 4)},
        "edges": {"u": net.u, "v": net.v, "w": np.round(net.w, 4)}})

    perm = tie_permutation(net.m, 0)
    scores = {m: (FILTERS[m](net) if m in ("W", "DF", "NC") else FILTERS[m](net, "jaccard")) for m in FILTERS}
    orders = {m: edge_order(sc, net.w, perm) for m, sc in scores.items()}
    inter = comm[net.u] != comm[net.v]
    hubs = s >= np.quantile(s, 0.9)
    metric = metric_backbone_mask(net)
    ranks = {}
    for m, o in orders.items():
        r = np.empty(net.m, dtype=np.int32)
        r[o] = np.arange(1, net.m + 1)
        ranks[m] = r
    _write(d / "edges.json", {"methods": list(FILTERS), "role": {"bridge": inter.astype(int), "hub": (hubs[net.u] | hubs[net.v]).astype(int),
                                                                   "metric": metric.astype(int)},
                              "rank": ranks, "score": {m: np.round(sc, 5) for m, sc in scores.items()}})
    _write(d / "backbones.json", {"m": net.m, "q": list(BUDGETS), "bits": {
        m: {f"{q:.2f}": _bits(top_k_mask(orders[m], k_of(q, net.m))) for q in BUDGETS} for m in FILTERS}})

    ev = analysis.wide(pd.read_parquet(results_dir / f"{name}.parquet"))
    mt, nm = analysis.method_table(ev), analysis.null_means(ev)
    sk = analysis.skill_table(ev)
    mx = {"methods": list(FILTERS), "q": list(BUDGETS), "functions": FUNCS5, "scores": {}, "ci": {}, "flags": {},
               "nulls": {}, "skill": {}}
    sp = samples_dir / f"{name}.npz"
    samples, mq_s = None, None
    if sp.exists():
        z = np.load(sp)
        samples = {k.replace("__", "|"): z[k] for k in z.files}
        mq_s = analysis.method_table(analysis.wide(pd.read_parquet(samples_dir / f"{name}_methods.parquet")))
    boot_out = {"B": B_EXPORT, "methods": list(FILTERS), "q": list(BUDGETS), "functions": ["spread", "comm", "path", "rank"], "data": {}}
    for m in FILTERS:
        mx["scores"][m], mx["ci"][m], mx["flags"][m], boot_out["data"][m] = {}, {}, {}, {}
        for q in BUDGETS:
            r = mt[(mt.method == m) & np.isclose(mt.q, q)].iloc[0]
            k = f"{q:.2f}"
            mx["scores"][m][k] = {f: round(float(r[f]), 4) for f in FUNCS5}
            mx["flags"][m][k] = {c: (bool(r[c]) if c == "connected" else round(float(r[c]), 4)) for c in
                                      ("connected", "n_components", "bridge_ratio", "hub_concentration", "tie_share")}
    for nl in ("R-U", "R-W"):
        mx["nulls"][nl] = {f"{q:.2f}": {f: round(float(nm[(nm.null == nl) & np.isclose(nm.q, q)][f].iloc[0]), 4) for f in FUNCS5}
                                for q in BUDGETS}
    for nl in ("R-U", "R-W"):
        mx["skill"][nl] = {m: {f"{q:.2f}": {f: round(float(sk[(sk.method == m) & (sk.null == nl) & (sk.function == f) &
                                                                      np.isclose(sk.q, q)].skill.iloc[0]), 4) for f in FUNCS5}
                                    for q in BUDGETS} for m in FILTERS}
    if samples is not None:
        for q in BUDGETS:
            point = mq_s[np.isclose(mq_s.q, q)].set_index("method")
            bm = bootstrap_means(samples, list(FILTERS), q, point, n_boot=B_EXPORT, seed=0)  # [m, g, B]
            lo, hi = np.quantile(bm, [0.025, 0.975], axis=2)
            for mi, m in enumerate(FILTERS):
                k = f"{q:.2f}"
                mx["ci"][m][k] = {f: [round(float(lo[mi, gi]), 4), round(float(hi[mi, gi]), 4)] for gi, f in
                                       enumerate(["spread", "comm", "path", "rank"])}
                boot_out["data"][m][k] = {f: (np.round(bm[mi, gi], 3) if f != "rank" else round(float(bm[mi, gi, 0]), 4))
                                          for gi, f in enumerate(["spread", "comm", "path", "rank"])}
    mx["has_ci"] = samples is not None
    _write(d / "metrics.json", mx)
    _write(d / "bootstrap.json", boot_out)

    if with_spread:
        _write(d / "spread.json", spread_payload(net, orders, results_dir, name))
    return {"name": name, "n": net.n, "m": net.m, "domain": net.meta.get("domain", ""), "has_ci": samples is not None}


def spread_payload(net, orders, results_dir, name, n_runs: int = 40, steps: int = 200, n_curve: int = 80) -> dict:
    """Mean infection time per node and infected-fraction curves for G and each backbone (3 seed tiers)."""
    beta = float(pd.read_parquet(Path(results_dir) / f"{name}.parquet", columns=["beta"]).beta.iloc[0])
    seeds_all = metrics.choose_seeds(net, "jaccard", 0.05)
    k = len(seeds_all) // 3
    seeds = {"high": int(seeds_all[0]), "median": int(seeds_all[k]), "low": int(seeds_all[-1])}
    ts = np.unique(np.linspace(0, steps, n_curve).astype(int))

    def summarise(mask):
        out = {}
        for tier, sd in seeds.items():
            T = metrics.simulate(net, mask, [sd], beta, n_runs, steps, 0)[0]          # runs x n
            frac = (T[:, :, None] <= ts[None, None, :]).mean(1).mean(0) if False else \
                   np.array([(T <= t).mean() for t in ts])
            out[tier] = {"mean_time": np.round(T.mean(0), 1), "curve": np.round(frac, 4)}
        return out

    data = {"G": summarise(None)}
    for m, o in orders.items():
        data[m] = {f"{q:.2f}": summarise(top_k_mask(o, k_of(q, net.m))) for q in BUDGETS}
    return {"beta": beta, "t": ts, "seeds": seeds, "n_runs": n_runs, "steps": steps, "data": data}


def export_all(names, out_dir: Path | None = None, atlas_label: str = "real_dev", with_spread: bool = True) -> list:
    """Export networks plus the shared files: networks.json, atlas.json, conclusions.json."""
    out_dir = Path(out_dir) if out_dir else ROOT / "ui" / "data"
    index = [export_network(n, out_dir, with_spread=with_spread) for n in names]
    _write(out_dir / "networks.json", {"networks": index, "methods": list(FILTERS), "q": list(BUDGETS)})
    atlas = ROOT / "results" / "analysis" / atlas_label / "e2_atlas.csv"
    if atlas.exists():
        a = pd.read_csv(atlas)
        a = a[["network", "method", "q", "function", "struct", "F_g", "gap", "failure", "reverse_failure", "connected", "bridge_ratio", "hub_concentration"]]
        _write(out_dir / "atlas.json", {"columns": list(a.columns), "rows": a.round(4).values.tolist()})
    conc = ROOT / "results" / "conclusions.json"
    if conc.exists():
        (out_dir / "conclusions.json").write_text(conc.read_text(encoding="utf-8"), encoding="utf-8")
    return index
