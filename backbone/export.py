"""Export the stored results to ui/data.js (a plain script, so ui/index.html also works when opened from disk).

    python -m backbone.export                  all networks that have results/<name>/meta.json
    python -m backbone.export hospital         only these

Per network: node layout, edges with their scores, each method's edge ranking (most to least important),
the sweeps, AUC tables, lambda study and the calibrated SI infection rate beta. The dashboard recomputes
sparsified graphs and SI epidemics in the browser from the edges and rankings.
"""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .sparsify import METHODS, ranking
from .spread import si_setup
from .summary import SCENARIOS
from .sweep import Network, eigenvector_centrality

RESULTS = Path("results")
OUT = Path("ui/data.js")


def clean(df, nd=5):
    """DataFrame -> dict of column lists; inf becomes the string 'inf', NaN becomes null."""
    out = {}
    for c in df.columns:
        col = df[c]
        if col.dtype.kind == "f":
            col = col.round(nd).astype(object).where(col.notna(), None)
            col = col.map(lambda v: "inf" if v is not None and np.isinf(v) else v)
        out[c] = col.tolist()
    return out


def export_network(name):
    d = RESULTS / name
    table = pd.read_csv(d / "edge_scores.csv")
    net = Network(table)
    seeds, beta, _ = si_setup(net)
    G = nx.Graph()
    G.add_nodes_from(range(net.n))
    G.add_weighted_edges_from(zip(net.i, net.j, net.p))
    pos = nx.spring_layout(G, weight="weight", seed=1, iterations=300)
    cen = eigenvector_centrality(net.adjacency())
    meta = json.loads((d / "meta.json").read_text())
    lam = pd.read_csv(d / "lambda_sweep.csv")
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
        "rank": {m: [int(k) for k in ranking(table, m)] for m in METHODS},
        "rho": {m: clean(g[["n_edges", "X", "rho_C", "components"]]) for m, g in
                pd.read_csv(d / "sweep_rhoC.csv").groupby("method")},
        "si": {m: clean(g[["n_edges", "X", "rho_t", "rho_t_std", "xi", "xi_std"]]) for m, g in
               pd.read_csv(d / "sweep_si.csv").groupby("method")},
        "auc_own": clean(pd.read_csv(d / "auc_xtilde.csv")),
        "auc_fixed": clean(pd.read_csv(d / "auc_fixed_limit.csv")),
        "lam_sweep": clean(lam),
        "opt": clean(pd.read_csv(d / "optimal_lambda.csv")),
    }


def main(names):
    names = names or sorted(p.parent.name for p in RESULTS.glob("*/meta.json"))
    data = {"methods": METHODS, "scenarios": {k: list(v) for k, v in SCENARIOS.items()},
            "networks": {n: export_network(n) for n in names}}
    summ = RESULTS / "summary" / "auc_xtilde_all.csv"
    if summ.exists():
        data["summary_own"] = clean(pd.read_csv(summ))
        data["summary_fixed"] = clean(pd.read_csv(RESULTS / "summary" / "auc_fixed_all.csv"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB) for {names}")


if __name__ == "__main__":
    main(sys.argv[1:])
