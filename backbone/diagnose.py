"""Diagnosis of where backbones fail (extension, step 3). Uses the stored sweeps, no new simulations.

  dissociation_summary   budgets where rho_C is high but rho_t is low (and the reverse), per method
  sparse_regime          rho_C, rho_t and xi^-1 at the sparsest budgets (below the metric-backbone size)
  retention_table        share of between-community edges ("bridges") kept, components and strength rank preserved
  matched_x_correlations structural variables against rho_t / rho_C across methods at equal size
Structural fidelity here means measures computed from the static graph (rho_C, connectivity, bridge retention,
strength ranks); functional fidelity means outcomes of the SI process (rho_t, xi^-1). This split is ours, the base
paper does not make it. Communities are Louvain partitions of the ORIGINAL proximity graph (several seeds), and a
bridge is an edge between two communities.
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .sparsify import METHODS, ranking
from .sweep import Network, sweep

N_PARTITIONS = 10


def merged_curves(rho_df, si_df):
    """rho_C (every k) restricted to the budgets where the SI sweep was run: one row per (method, n_edges)."""
    return si_df.merge(rho_df[["method", "n_edges", "rho_C", "components"]], on=["method", "n_edges"])


def dissociation_summary(merged, x_min=0.05, high=0.9, low=0.8):
    """Per method: share of budgets (X > x_min) with rho_C >= high but rho_t < low (`structure_ok_function_bad`) and
    the reverse, plus the Spearman correlation between rho_C and rho_t across budgets."""
    rows = []
    for m, g in merged[merged.X > x_min].groupby("method", sort=False):
        rho = spearmanr(g.rho_C, g.rho_t)[0] if g.rho_C.nunique() > 1 and g.rho_t.nunique() > 1 else np.nan
        rows.append((m, len(g), ((g.rho_C >= high) & (g.rho_t < low)).mean(),
                     ((g.rho_C < low) & (g.rho_t >= high)).mean(), rho))
    return pd.DataFrame(rows, columns=["method", "n_budgets", "structure_ok_function_bad",
                                       "function_ok_structure_bad", "spearman_rhoC_rhot"])


def sparse_regime(merged, x_max):
    """Fidelity of every method at the budgets with X <= x_max (the regime below the metric-backbone size)."""
    cols = ["method", "n_edges", "X", "rho_C", "rho_t", "xi", "components"]
    return merged[merged.X <= x_max + 1e-12][cols].sort_values(["method", "X"]).reset_index(drop=True)


def louvain_labels(net, n_partitions=N_PARTITIONS):
    """Community label (array over net.nodes) of each of several seeded Louvain partitions of the full graph."""
    G = nx.Graph()
    G.add_nodes_from(range(net.n))
    G.add_weighted_edges_from(zip(net.i.tolist(), net.j.tolist(), net.p.tolist()))
    out = []
    for seed in range(n_partitions):
        lab = np.zeros(net.n, dtype=int)
        for c, nodes in enumerate(nx.community.louvain_communities(G, weight="weight", seed=seed)):
            lab[list(nodes)] = c
        out.append(lab)
    return out


def bridge_retention(net, labels, keep):
    """Share of the between-community edges of the full graph that are in the subgraph `keep` (edge positions)."""
    between = labels[net.i] != labels[net.j]
    kept = np.zeros(net.m, dtype=bool)
    kept[keep] = True
    return float((between & kept).sum() / between.sum()) if between.any() else float("nan")


def _strength_rho(net, keep, s_full):
    s = net.adjacency(keep).sum(1)
    return 0.0 if np.ptp(s) == 0 else float(spearmanr(s_full, s)[0])


def retention_table(net, ks, methods=METHODS, labels=None, seed=0):
    """Per method and edge count: mean/std over partitions of bridge retention, plus components and the Spearman
    correlation of node strengths with the full graph."""
    labels = louvain_labels(net) if labels is None else labels
    s_full = net.adjacency().sum(1)
    rows = []
    for m in methods:
        order = ranking(net.table, m, seed)
        for k in ks:
            keep = order[:k]
            r = [bridge_retention(net, lab, keep) for lab in labels]
            rows.append((m, k, np.nanmean(r), np.nanstd(r), int(net.n_components(keep)), _strength_rho(net, keep, s_full)))
    return pd.DataFrame(rows, columns=["method", "n_edges", "bridge_retention", "bridge_retention_std",
                                       "components", "strength_rho"])


def matched_x_correlations(df, x_lo=0.05, x_hi=0.9):
    """Mean over budgets of the Spearman correlation, across the methods at the same X, between a structural variable
    (components, bridge retention, strength-rank preservation) and rho_t / rho_C. Comparing methods at equal size removes
    the common trend with X. `df` is the merged table with one row per (method, budget)."""
    x = df[(df.X > x_lo) & (df.X < x_hi)]
    rows = []
    for col in ["components", "bridge_retention", "strength_rho"]:
        for y in ["rho_t", "rho_C"]:
            r = [spearmanr(g[col], g[y])[0] for _, g in x.groupby("X") if g[col].nunique() > 1 and g[y].nunique() > 1]
            rows.append((col, y, float(np.nanmean(r)) if r else float("nan"), len(r)))
    return pd.DataFrame(rows, columns=["variable", "versus", "mean_spearman", "n_budgets"])


def run(net_dir, out_dir=None):
    """All tables for one network from its stored base results (`net_dir` = results/<network>); written to
    out_dir (default net_dir/diagnose): dissociation.csv, sparse_regime.csv, retention.csv."""
    base = Path(net_dir)
    out = Path(out_dir or base / "diagnose")
    out.mkdir(parents=True, exist_ok=True)
    net = Network(pd.read_csv(base / "edge_scores.csv"))
    si = pd.read_csv(base / "sweep_si.csv")
    # the stored rho_C sweep is on a coarser grid than the SI budgets, so recompute rho_C exactly at the SI edge counts
    merged = merged_curves(sweep(net, ks=sorted(si.n_edges.unique())), si)
    x_metric = float(((net.table.distortion <= 1 + 1e-9).sum() - (net.n - 1)) / (net.m - (net.n - 1)))
    diss = dissociation_summary(merged)
    sparse = sparse_regime(merged, x_metric)
    ret = retention_table(net, sorted(merged.n_edges.unique()))
    both = merged.merge(ret.drop(columns="components"), on=["method", "n_edges"])
    diss.to_csv(out / "dissociation.csv", index=False)
    sparse.to_csv(out / "sparse_regime.csv", index=False)
    both.to_csv(out / "retention.csv", index=False)
    matched_x_correlations(both).to_csv(out / "matched_x_correlations.csv", index=False)
    return diss, sparse, both
