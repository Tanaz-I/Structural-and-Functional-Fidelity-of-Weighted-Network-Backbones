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
    if method == "wEffRes":
        return weighted_effective_resistance(table)
    if method == "SmD":
        return -table["distortion"].to_numpy()
    if method == "DBS":
        return table["lam"].to_numpy()
    raise ValueError(f"unknown method {method!r}")


def ranking(table, method, seed=0):
    """Edge row positions ordered from most to least important (seeded random tie-break)."""
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
