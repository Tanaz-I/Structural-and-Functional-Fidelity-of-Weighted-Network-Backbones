"""Export the stored results to ui/data.js (a plain script, so ui/index.html also works when opened from disk).
`python run.py` calls this at the end of a full run.

Per network: node layout, edges with their scores, each method's edge ranking (most to least important),
the sweeps, AUC tables, lambda study, the calibrated SI infection rate beta, and the results of the sensitivity,
diagnosis and extension stages. The dashboard recomputes sparsified graphs and SI epidemics in the browser from the
edges and rankings.
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .sparsify import METHODS, ranking

VARIANTS = ["W+T", "DispFilt+T", "W+B", "DispFilt+B"]  # connected variants shown in the lens and the epidemic tab
from .summary import SCENARIOS
from .sweep import Network, eigenvector_centrality

RESULTS = Path("results")
OUT = Path("ui/data.js")


def clean(df, nd=5):
    """DataFrame -> dict of column lists; inf becomes the string 'inf', NaN becomes null (valid JSON)."""
    out = {}
    for c in df.columns:
        col = df[c]
        if col.dtype.kind == "f":
            out[c] = [None if v != v else ("inf" if np.isinf(v) else round(v, nd)) for v in col.tolist()]
        else:
            out[c] = [None if (isinstance(v, float) and v != v) else v for v in col.tolist()]
    return out


def export_network(name, root=RESULTS):
    d = Path(root) / name
    table = pd.read_csv(d / "edge_scores.csv")
    net = Network(table)
    info = json.loads((d / "si_setup.json").read_text())
    beta, seeds = info["beta"], [net.nodes.index(x) for x in info["seed_nodes"]]
    G = nx.Graph()
    G.add_nodes_from(range(net.n))
    G.add_weighted_edges_from(zip(net.i, net.j, net.p))
    pos = nx.spring_layout(G, weight="weight", seed=1, iterations=300)
    cen = eigenvector_centrality(net.adjacency())
    meta = json.loads((d / "meta.json").read_text())
    lam = pd.read_csv(d / "lambda_sweep.csv")
    ret_cols = ["method", "n_edges", "X", "rho_C", "rho_t", "xi", "components", "bridge_retention", "strength_rho"]
    return {
        "name": name,
        "meta": meta,
        "beta": beta,
        "seeds": [int(s) for s in seeds],
        "nodes": [int(x) for x in net.nodes],
        "pos": [[round(float(x), 4), round(float(y), 4)] for x, y in (pos[k] for k in range(net.n))],
        "centrality": [round(float(c), 4) for c in cen],
        "edges": {"u": net.i.tolist(), "v": net.j.tolist(), "p": clean(table[["proximity"]], 6)["proximity"],
                  "s": clean(table[["distortion"]], 4)["distortion"], "lam": clean(table[["lam"]], 4)["lam"]},
        "rank": {m: [int(k) for k in ranking(table, m)] for m in METHODS + VARIANTS},
        "rho": {**{m: clean(g[["n_edges", "X", "rho_C", "components"]]) for m, g in
                   pd.read_csv(d / "sweep_rhoC.csv").groupby("method")},
                **{m: clean(g[["n_edges", "X", "rho_C", "components"]]) for m, g in
                   pd.read_csv(d / "extension" / "curves_rhoC.csv").groupby("method") if m in VARIANTS}},
        "si": {**{m: clean(g[["n_edges", "X", "rho_t", "rho_t_std", "xi", "xi_std"]]) for m, g in
                  pd.read_csv(d / "sweep_si.csv").groupby("method")},
               **{m: clean(g[["n_edges", "X", "rho_t", "rho_t_std", "xi", "xi_std"]]) for m, g in
                  pd.read_csv(d / "extension" / "curves_si.csv").groupby("method") if m in VARIANTS}},
        "auc_own": clean(pd.read_csv(d / "auc_xtilde.csv")),
        "auc_fixed": clean(pd.read_csv(d / "auc_fixed_limit.csv")),
        "lam_sweep": clean(lam),
        "opt": clean(pd.read_csv(d / "optimal_lambda.csv")),
        "sens": clean(pd.read_csv(d / "sensitivity" / "sensitivity.csv")),
        "diag": {"dissociation": clean(pd.read_csv(d / "diagnose" / "dissociation.csv")),
                 "retention": clean(pd.read_csv(d / "diagnose" / "retention.csv")[ret_cols]),
                 "matched": clean(pd.read_csv(d / "diagnose" / "matched_x_correlations.csv"))},
        "ext": {"auc": clean(pd.read_csv(d / "extension" / "auc.csv")),
                "verdicts": clean(pd.read_csv(d / "extension" / "verdicts.csv"))},
    }


def main(names=None, root=RESULTS, out=OUT):
    root, out = Path(root), Path(out)
    names = names or sorted(p.parent.name for p in root.glob("*/meta.json"))
    data = {"methods": METHODS, "scenarios": {k: list(v) for k, v in SCENARIOS.items()},
            "networks": {n: export_network(n, root) for n in names}}
    summ = root / "summary" / "auc_xtilde_all.csv"
    if summ.exists():
        data["summary_own"] = clean(pd.read_csv(summ))
        data["summary_fixed"] = clean(pd.read_csv(root / "summary" / "auc_fixed_all.csv"))
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    return out
