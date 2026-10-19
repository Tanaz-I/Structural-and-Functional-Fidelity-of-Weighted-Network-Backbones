"""Six edge-scoring filters, the matched-budget controller and the random null baselines."""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .network import Network

# ---------------------------------------------------------------- edge-scoring filters
def score_threshold(net: Network) -> np.ndarray:
    return net.w.copy()


def score_disparity(net: Network) -> np.ndarray:
    """Serrano et al. 2009. score = 1 - min over endpoints of (1-p)^(k-1)."""
    s, k = net.strength(), net.degree()
    p_u, p_v = net.w / s[net.u], net.w / s[net.v]
    a_u = np.power(1.0 - p_u, k[net.u] - 1.0)
    a_v = np.power(1.0 - p_v, k[net.v] - 1.0)
    return 1.0 - np.minimum(a_u, a_v)


def score_noise_corrected(net: Network, double_count: bool = True) -> np.ndarray:
    """Coscia & Neffke 2017 (undirected form, as in the authors' reference code).

    score = lift / posterior standard deviation of lift.
    """
    s = net.strength()
    # n.. : every undirected edge counted in both directions (authors' code); NetBone uses W instead
    tot = s.sum() if double_count else net.w.sum()
    ni, nj, nij = s[net.u], s[net.v], net.w
    mean_prior = (ni * nj) / tot**2
    kappa = tot / (ni * nj)
    lift = (kappa * nij - 1.0) / (kappa * nij + 1.0)
    var_prior = (ni * nj * (tot - ni) * (tot - nj)) / (tot**4 * (tot - 1.0))
    var_prior = np.maximum(var_prior, 1e-300)
    alpha_prior = (mean_prior**2 / var_prior) * (1.0 - mean_prior) - mean_prior
    beta_prior = (mean_prior / var_prior) * (1.0 - mean_prior**2) - (1.0 - mean_prior)
    alpha_post = alpha_prior + nij
    beta_post = tot - nij + beta_prior
    e_p = alpha_post / (alpha_post + beta_post)
    var_nij = e_p * (1.0 - e_p) * tot
    d = 1.0 / (ni * nj) - tot * (ni + nj) / (ni * nj) ** 2
    var_lift = var_nij * ((2.0 * (kappa + nij * d)) / (kappa * nij + 1.0) ** 2) ** 2
    sd = np.sqrt(np.maximum(var_lift, 1e-300))
    return lift / sd


def _apsp(net: Network, dist: np.ndarray) -> np.ndarray:
    return dijkstra(net.csr(dist), directed=False)


def score_smds(net: Network, proximity: str = "jaccard", tol: float = 1e-9) -> np.ndarray:
    """Semi-metric distortion (Soriano-Panos et al. 2025; Simas et al. 2021).

    s = d_ij / indirect shortest distance, clipped so metric edges have s = 1.
    Because the all-pairs distance D_ij is the direct edge exactly when the edge is
    metric, no per-edge removal is needed: s = d_ij / D_ij (D_ij <= d_ij).
    Score = -s so the metric backbone (s = 1) ranks first.
    """
    d = net.distance(proximity)
    D = _apsp(net, d)
    dij = D[net.u, net.v]
    den = np.minimum(d, dij)
    s = np.divide(d, den, out=np.ones_like(d), where=den > 0)  # zero-length edges (proximity 1) count as metric
    s = np.where(d - dij <= tol * np.maximum(d, 1.0), 1.0, s)
    return -s


def metric_backbone_mask(net: Network, proximity: str = "jaccard") -> np.ndarray:
    return score_smds(net, proximity) == -1.0


def score_hss(net: Network, proximity: str = "jaccard") -> np.ndarray:
    """High-salience skeleton (Grady et al. 2012): share of shortest-path trees using an edge."""
    d = net.distance(proximity)
    G = net.csr(d)
    _, pred = dijkstra(G, directed=False, return_predecessors=True)
    keys = net.u * net.n + net.v  # sorted because Network keeps edges lexsorted by (u, v)
    count = np.zeros(net.m)
    for src in range(net.n):
        p = pred[src]
        j = np.flatnonzero(p >= 0)
        a, b = p[j], j
        k = np.minimum(a, b) * net.n + np.maximum(a, b)
        count[np.searchsorted(keys, k)] += 1
    return count / net.n


def effective_resistance_matrix(A: np.ndarray) -> np.ndarray:
    """All-pairs effective resistance for a connected conductance matrix A (dense)."""
    n = len(A)
    L = np.diag(A.sum(1)) - A
    J = np.ones((n, n)) / n
    Lp = np.linalg.inv(L + J) - J  # pseudoinverse of a connected Laplacian
    d = np.diag(Lp)
    return d[:, None] + d[None, :] - 2.0 * Lp


def score_eff_resistance(net: Network, proximity: str = "jaccard") -> np.ndarray:
    """Weighted effective resistance: proximity x R_ij, Laplacian conductance = proximity."""
    p = net.proximity(proximity)
    R = effective_resistance_matrix(net.csr(p).toarray())
    return p * R[net.u, net.v]


FILTERS = {
    "W": score_threshold,
    "DF": score_disparity,
    "NC": score_noise_corrected,
    "SMDS": score_smds,
    "HSS": score_hss,
    "wEffRes": score_eff_resistance,
}
PROXIMITY_FILTERS = frozenset({"SMDS", "HSS", "wEffRes"})  # scores that depend on the proximity measure


# ---------------------------------------------------------------- matched-budget controller
BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70)


def tie_permutation(m: int, seed: int) -> np.ndarray:
    """Seeded random rank per edge, shared by every method."""
    return np.random.default_rng(seed).permutation(m)


def edge_order(score: np.ndarray, w: np.ndarray, tie_perm: np.ndarray) -> np.ndarray:
    """Edge indices best-first: score desc, then weight desc, then shared random rank."""
    return np.lexsort((tie_perm, -w, -score))


def k_of(q: float, m: int) -> int:
    return int(round(q * m))


def top_k_mask(order: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:k]] = True
    return mask


def tie_share(score: np.ndarray, order: np.ndarray, k: int) -> float:
    """Share of edges whose score ties with the cut-off edge (those are ranked by the tie rules)."""
    if k <= 0 or k >= len(score):
        return 0.0
    return float(np.mean(score == score[order[k - 1]]))


def backbone_flags(net: Network, mask: np.ndarray) -> dict:
    nc = net.n_components(mask)
    return {"k": int(mask.sum()), "n_components": nc, "connected": nc == 1}


def connectivity_floor(net: Network, order: np.ndarray, step: float = 0.01) -> float:
    """Smallest q (on a fine grid) at which the backbone is connected."""
    for q in np.arange(step, 1.0 + 1e-9, step):
        if net.n_components(top_k_mask(order, k_of(q, net.m))) == 1:
            return float(round(q, 4))
    return 1.0


def max_spanning_tree_mask(net: Network) -> np.ndarray:
    from scipy.sparse.csgraph import minimum_spanning_tree
    inv = 1.0 / net.w
    T = minimum_spanning_tree(net.csr(inv)).tocoo()
    keys = net.u * net.n + net.v
    a, b = np.minimum(T.row, T.col), np.maximum(T.row, T.col)
    mask = np.zeros(net.m, dtype=bool)
    mask[np.searchsorted(keys, a * net.n + b)] = True
    return mask


# ---------------------------------------------------------------- random null baselines
def random_uniform_order(m: int, rng: np.random.Generator) -> np.ndarray:
    """R-U: uniform random order."""
    return rng.permutation(m)


def random_weighted_order(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """R-W: sampling without replacement proportional to weight (Efraimidis-Spirakis keys)."""
    keys = np.log(rng.random(len(w))) / w
    return np.argsort(-keys)


NULLS = {"R-U": lambda w, rng: random_uniform_order(len(w), rng), "R-W": random_weighted_order}
