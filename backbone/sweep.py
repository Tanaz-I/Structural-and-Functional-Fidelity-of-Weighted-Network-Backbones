"""Progressive sparsification sweep and eigenvector-centrality preservation rho_C (paper Sec. 5.3.1, Eq. 14).

For each method the edges are ranked once (backbone.sparsify.ranking); the subgraph with k edges keeps the top k.
k runs from n-1 (X = 0) to |G| (X = 1) one edge at a time, so X is swept at the finest possible resolution.
rho_C = Spearman correlation between node eigenvector centralities of the full and sparsified proximity matrix,
where removed edges get proximity 0 and kept edges keep their proximity.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from .backbones import adjusted_size
from .sparsify import METHODS, ranking

ZERO = 1e-12  # centrality entries below this are numerical noise on other components -> tied at 0


class Network:
    """Edge table plus dense index arrays, so each subgraph is a cheap slice of the edge list."""

    def __init__(self, table):
        self.table = table.reset_index(drop=True)
        self.nodes = sorted(set(table["u"]) | set(table["v"]))
        idx = {n: k for k, n in enumerate(self.nodes)}
        self.n = len(self.nodes)
        self.m = len(table)
        self.i = self.table["u"].map(idx).to_numpy()
        self.j = self.table["v"].map(idx).to_numpy()
        self.p = self.table["proximity"].to_numpy()

    def adjacency(self, keep=None):
        """Dense symmetric proximity matrix of the subgraph given by edge positions `keep` (all if None)."""
        k = slice(None) if keep is None else keep
        A = np.zeros((self.n, self.n))
        A[self.i[k], self.j[k]] = A[self.j[k], self.i[k]] = self.p[k]
        return A

    def n_components(self, keep):
        k = keep
        A = csr_matrix((np.ones(len(k)), (self.i[k], self.j[k])), shape=(self.n, self.n))
        return connected_components(A, directed=False)[0]


def eigenvector_centrality(A):
    """Leading eigenvector (Perron vector) of the symmetric proximity matrix, non-negative, noise zeroed."""
    w, V = np.linalg.eigh(A)
    c = np.abs(V[:, -1])
    c[c < ZERO] = 0.0
    return c


def rho_C(c_full, c_sub):
    """Spearman rank correlation of centralities (ties get average ranks)."""
    return spearmanr(c_full, c_sub)[0]


def rho_grid(net, n_points=200):
    """Edge counts k (X from 0 to 1) at which the sweep is evaluated; every k if the network is small enough."""
    full = range(net.n - 1, net.m + 1)
    if len(full) <= n_points:
        return list(full)
    return sorted({round((net.n - 1) + x * (net.m - (net.n - 1))) for x in np.linspace(0, 1, n_points)})


def sweep(net, methods=METHODS, seed=0, ks=None, orders=None):
    """One row per (method, number of edges k): X, rho_C, number of connected components.
    `orders` optionally maps a method name to a ready-made edge order (most important first)."""
    c_full = eigenvector_centrality(net.adjacency())
    ks = rho_grid(net) if ks is None else ks
    rows = []
    for method in methods:
        order = orders[method] if orders and method in orders else ranking(net.table, method, seed)
        for k in ks:
            keep = order[:k]
            rows.append((method, k, adjusted_size(net.n, net.m, k),
                         rho_C(c_full, eigenvector_centrality(net.adjacency(keep))),
                         net.n_components(keep)))
    return pd.DataFrame(rows, columns=["method", "n_edges", "X", "rho_C", "components"])


def connectivity_limits(net, methods=METHODS, seed=0):
    """Per method, the smallest X from which the subgraph stays connected for every larger X.
    Scans every edge count downward from the full graph (cheap: one connected-components call per step)."""
    out = {}
    for method in methods:
        order = ranking(net.table, method, seed)
        k_conn = net.m
        for k in range(net.m, net.n - 2, -1):
            if net.n_components(order[:k]) != 1:
                break
            k_conn = k
        out[method] = adjusted_size(net.n, net.m, k_conn)
    return out
