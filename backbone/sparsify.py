"""Edge-importance scores for the five sparsification methods of the paper (Table 2) and the
progressive sparsification that keeps the top edges for a target adjusted size X.

Every method is turned into an `importance` (higher = keep longer), so removal is always
"lowest importance first":
    W        importance = p_ij                          (remove increasing p_ij)
    DispFilt importance = -alpha_ij                     (remove decreasing alpha_ij)
    wEffRes  importance = p_ij * R_ij                   (remove increasing)
    SmD      importance = -s_ij                         (remove decreasing semi-metric distortion)
    DBS      importance = lambda_ij                     (remove increasing)
Ties (SmD has many s_ij = 1 edges) are broken by a seeded random order, so the result is reproducible and
carries no bias toward strong or weak edges.
"""
import networkx as nx
import numpy as np
import pandas as pd

from .backbones import adjusted_size

METHODS = ["W", "DispFilt", "wEffRes", "SmD", "DBS"]


def disparity_alpha(table):
    """alpha_ij = 1 - min(F_ki(p_ij), F_kj(p_ij)), F_k(p) = 1 - (1-p)^(k-1) (Sec. 5.2).
    p_ij is the proximity, as in the paper. A degree-1 node has F = 1 (its only edge is maximally significant)."""
    deg = pd.concat([table["u"], table["v"]]).value_counts()
    p = table["proximity"].to_numpy()

    def F(k):
        k = table[k].map(deg).to_numpy()
        return np.where(k > 1, 1 - (1 - p) ** np.maximum(k - 1, 1), 1.0)

    return 1 - np.minimum(F("u"), F("v"))


def disparity_alpha_normalised(table):
    """Serrano et al.'s original disparity filter: each endpoint uses the edge's share of its own strength,
    p_i = w_ij / s_i (w = proximity), and alpha_ij = min over the two endpoints of (1 - p_i)^(k_i - 1).
    A degree-1 node has alpha = 0 (maximally significant). Used only as a sensitivity variant of DispFilt."""
    w = table["proximity"].to_numpy()
    strength = pd.concat([table["u"], table["v"]]).to_frame("x").assign(w=np.concatenate([w, w])).groupby("x").w.sum()
    deg = pd.concat([table["u"], table["v"]]).value_counts()

    def alpha(col):
        k = table[col].map(deg).to_numpy()
        share = w / table[col].map(strength).to_numpy()
        return np.where(k > 1, (1 - share) ** np.maximum(k - 1, 1), 0.0)

    return np.minimum(alpha("u"), alpha("v"))


def weighted_effective_resistance(table):
    """p_ij * R_ij with R_ij = (e_i - e_j)^T L^+ (e_i - e_j) and Laplacian conductances = proximity."""
    nodes = sorted(set(table["u"]) | set(table["v"]))
    idx = {n: k for k, n in enumerate(nodes)}
    i = table["u"].map(idx).to_numpy()
    j = table["v"].map(idx).to_numpy()
    p = table["proximity"].to_numpy()
    n = len(nodes)
    A = np.zeros((n, n))
    A[i, j] = A[j, i] = p
    Lp = np.linalg.pinv(np.diag(A.sum(1)) - A)
    R = Lp[i, i] + Lp[j, j] - 2 * Lp[i, j]
    return p * R


def importance(table, method):
    if method == "W":
        return table["proximity"].to_numpy()
    if method == "DispFilt":
        return -disparity_alpha(table)
    if method == "DispFiltNorm":  # sensitivity variant, not one of METHODS
        return -disparity_alpha_normalised(table)
    if method == "wEffRes":
        return weighted_effective_resistance(table)
    if method == "SmD":
        return -table["distortion"].to_numpy()
    if method == "DBS":
        return table["lam"].to_numpy()
    raise ValueError(f"unknown method {method!r}")


def spanning_tree_edges(table, order, rng=None):
    """Edge positions of a spanning tree (Kruskal with union-find), scanning the edges in `order`; with `rng` the scan
    order is a random permutation instead. Returned in the order they were added."""
    nodes = sorted(set(table["u"]) | set(table["v"]))
    idx = {n: k for k, n in enumerate(nodes)}
    u, v = table["u"].map(idx).to_numpy(), table["v"].map(idx).to_numpy()
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    scan = rng.permutation(len(table)) if rng is not None else order
    tree = []
    for e in scan:
        a, b = find(u[e]), find(v[e])
        if a != b:
            parent[a] = b
            tree.append(int(e))
            if len(tree) == len(nodes) - 1:
                break
    return tree


def ranking(table, method, seed=0):
    """Edge row positions ordered from most to least important (seeded random tie-break).

    Extension variants of a method M (a spanning tree or the metric backbone is forced in first):
      `M+T`  spanning tree built along M's own ranking, then M's order: connected at every budget >= n-1
      `M+R`  the same with a random spanning tree (control)
      `M+B`  spanning tree inside the metric backbone, then the other metric-backbone edges, then M's order: connected
             at every budget >= n-1 and, from the metric-backbone size on, every shortest path is preserved
      `M+X`  guided tree, then random edges up to the metric-backbone size, then M's order (control for `+B`: same
             number of forced edges, not chosen for being metric)"""
    if method.endswith(("+T", "+R", "+B", "+X")):
        kind, order = method[-1], ranking(table, method[:-2], seed)
        rng = np.random.default_rng(seed + 12345)
        is_metric = (table["distortion"] <= 1 + 1e-9).to_numpy()
        if kind == "T":
            tree = spanning_tree_edges(table, order)
            forced = tree
        elif kind == "R":
            tree = spanning_tree_edges(table, order, rng)
            forced = tree
        elif kind == "B":
            tree = spanning_tree_edges(table, order[is_metric[order]])
            in_tree = set(tree)
            forced = tree + [int(e) for e in order if is_metric[e] and e not in in_tree]
        else:
            tree = spanning_tree_edges(table, order)
            n_extra = int(is_metric.sum()) - len(tree)
            rest = rng.permutation(np.setdiff1d(np.arange(len(table)), tree))
            forced = tree + [int(e) for e in rest[:max(n_extra, 0)]]
        taken = np.zeros(len(table), dtype=bool)
        taken[forced] = True
        return np.concatenate([np.array(forced, dtype=int), order[~taken[order]]])
    imp = importance(table, method)
    tie = np.random.default_rng(seed).permutation(len(imp))
    return np.lexsort((tie, -imp))  # primary: importance descending, secondary: random


def edges_to_keep(n_nodes, n_edges, X):
    """Number of edges of the subgraph with adjusted size X (Eq. 13)."""
    return int(round((n_nodes - 1) + X * (n_edges - (n_nodes - 1))))


def sparsify(table, method, X, seed=0):
    """Edge table of the subgraph with adjusted size X, keeping the most important edges."""
    n = len(set(table["u"]) | set(table["v"]))
    keep = ranking(table, method, seed)[: edges_to_keep(n, len(table), X)]
    return table.iloc[np.sort(keep)]


def to_graph(sub, nodes):
    """networkx graph on ALL original nodes with proximity weights (removed edges simply absent)."""
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for r in sub.itertuples(index=False):
        H.add_edge(r.u, r.v, proximity=r.proximity, distance=r.distance)
    return H


def is_connected(sub, nodes):
    return nx.is_connected(to_graph(sub, nodes))


def realised_X(table, sub):
    n = len(set(table["u"]) | set(table["v"]))
    return adjusted_size(n, len(table), len(sub))
