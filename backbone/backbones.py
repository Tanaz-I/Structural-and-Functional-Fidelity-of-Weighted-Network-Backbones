"""Distance backbones of a weighted graph (Pereira, Costa & Rocha 2026, Sec. 2-3.1).

An edge (i, j) belongs to the distance backbone B_g if d_ij <= g(path) for every indirect path,
i.e. d_ij equals the shortest-path distance d^{T,g}_ij under the path-length measure g:
    g = sum  -> metric backbone        (semi-metric distortion s_ij = d_ij / d^{T,sum}_ij)
    g = max  -> ultrametric backbone   (union of minimum spanning trees)
    g = Dombi g_lam(x, y) = (x^lam + y^lam)^(1/lam)   -> family used by Distance Backbone Synthesis (DBS)
The Dombi family is the l_lam norm, so a path's g_lam-length is (sum d^lam)^(1/lam) and the g_lam shortest path
is an ordinary shortest path on edge weights d^lam. Backbones are nested: B_lam shrinks as lam grows
(lam=1 metric, lam -> inf ultrametric, lam -> 0 the full graph).

Per-edge importance lambda_ij = largest lam for which the edge is still in B_lam (inf for ultrametric edges).
DBS removes edges in increasing lambda_ij; B_lam = {edges with lambda_ij >= lam}.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

TOL = 1e-9
LAMBDA_MIN, LAMBDA_MAX = 1e-3, 200.0  # search range for lambda_ij of non-ultrametric edges
EXP_CLIP = 690.0                      # keep exp() inside float64 range
BISECTION_STEPS = 30                  # log-lam resolution (log 200/1e-3)/2^30 ~ 1e-8


def edge_arrays(G):
    """Sorted node list, node->index map, and edge endpoints (i<j) with their distances."""
    nodes = sorted(G)
    idx = {n: k for k, n in enumerate(nodes)}
    ends = [(idx[u], idx[v], d) for u, v, d in G.edges(data="distance")]
    i, j, d = (np.array(x) for x in zip(*ends))
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return nodes, idx, lo.astype(int), hi.astype(int), d.astype(float)


def semi_metric_distortion(G):
    """s_ij = d_ij / d^{T,sum}_ij >= 1 per edge (1 = metric edge). Returns array aligned with edge_arrays."""
    nodes, _, i, j, d = edge_arrays(G)
    n = len(nodes)
    sp = dijkstra(csr_matrix((d, (i, j)), shape=(n, n)), directed=False)
    return d / sp[i, j]


def minimax_closure(n, i, j, d):
    """Ultrametric closure d^{T,max}: smallest possible largest edge on a path (Floyd-Warshall, vectorised)."""
    M = np.full((n, n), np.inf)
    M[i, j] = M[j, i] = d
    np.fill_diagonal(M, 0.0)
    for k in range(n):
        np.minimum(M, np.maximum(M[:, k:k + 1], M[k:k + 1, :]), out=M)
    return M


def ultrametric_mask(G):
    """True for edges in the ultrametric backbone (d_ij equals the minimax path distance)."""
    nodes, _, i, j, d = edge_arrays(G)
    M = minimax_closure(len(nodes), i, j, d)
    return d <= M[i, j] * (1 + TOL)


def _in_backbone(lam, n, i, j, logd, edge_ids):
    """Membership at one lam for the given edges: edge e is in B_lam iff the g_lam-shortest path i->j has no
    indirect path shorter than the edge itself. Weights are (d_k/d_e)^lam so the edge's own length is 1."""
    out = np.empty(len(edge_ids), dtype=bool)
    for k, e in enumerate(edge_ids):
        w = np.exp(np.clip(lam * (logd - logd[e]), -EXP_CLIP, EXP_CLIP))
        sp = dijkstra(csr_matrix((w, (i, j)), shape=(n, n)), directed=False, indices=i[e])
        out[k] = sp[j[e]] > 1 - TOL
    return out


def dombi_lambda(G):
    """lambda_ij per edge (array aligned with edge_arrays). Bisection in log(lam); membership is monotone in lam."""
    nodes, _, i, j, d = edge_arrays(G)
    n, logd = len(nodes), np.log(d)
    lam = np.full(len(d), np.inf)
    todo = np.flatnonzero(~ultrametric_mask(G))
    for e in todo:
        lo, hi = np.log(LAMBDA_MIN), np.log(LAMBDA_MAX)  # lo: in backbone, hi: out (assumed)
        if _in_backbone(LAMBDA_MAX, n, i, j, logd, [e])[0]:
            lam[e] = LAMBDA_MAX  # never excluded within range (ties); cap
            continue
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _in_backbone(np.exp(mid), n, i, j, logd, [e])[0] else (lo, mid)
        lam[e] = np.exp(lo)
    return lam


def edge_table(G):
    """One row per edge: u, v, proximity, distance, distortion (sum), lam (DBS importance)."""
    nodes, _, i, j, d = edge_arrays(G)
    u = np.array(nodes)[i]
    v = np.array(nodes)[j]
    prox = np.array([G[a][b]["proximity"] for a, b in zip(u, v)])
    return pd.DataFrame({"u": u, "v": v, "proximity": prox, "distance": d,
                         "distortion": semi_metric_distortion(G), "lam": dombi_lambda(G)})


def metric_edges(table):
    return table[table["distortion"] <= 1 + TOL]


def ultrametric_edges(table):
    return table[np.isinf(table["lam"])]


def backbone_at_lambda(table, lam):
    """Edges of the Dombi distance backbone B_lam (nested: bigger lam -> smaller backbone)."""
    return table[table["lam"] >= lam * (1 - TOL)]


def adjusted_size(n_nodes, n_edges_total, n_edges_sub):
    """X = (|G'| - (n-1)) / (|G| - (n-1)): 0 for the spanning tree, 1 for the full graph (Eq. 13)."""
    return (n_edges_sub - (n_nodes - 1)) / (n_edges_total - (n_nodes - 1))
