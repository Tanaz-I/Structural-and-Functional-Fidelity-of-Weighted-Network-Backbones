"""Fidelity metrics: structure, spreading (SI and SIR), communities, shortest paths, node rankings."""
from __future__ import annotations


from numba import njit
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from .network import Network

# ---------------------------------------------------------------- structure
def _connected_pairs(net: Network, mask) -> float:
    idx = np.flatnonzero(mask) if mask is not None else np.arange(net.m)
    g = sp.coo_matrix((np.ones(len(idx)), (net.u[idx], net.v[idx])), shape=(net.n, net.n))
    _, lab = sp.csgraph.connected_components(g, directed=False)
    sizes = np.bincount(lab).astype(float)
    return float((sizes * (sizes - 1)).sum() / (net.n * (net.n - 1)))


def structure_components(net: Network, mask: np.ndarray) -> dict:
    reach = _connected_pairs(net, mask) / _connected_pairs(net, None)
    s_g, s_b = net.strength(), net.strength(mask)
    noniso = float(np.mean(s_b > 0))
    hell = np.sqrt(0.5 * np.sum((np.sqrt(s_g / s_g.sum()) - np.sqrt(s_b / max(s_b.sum(), 1e-300))) ** 2))
    return {"reach": reach, "noniso": noniso, "strength": 1.0 - float(hell)}


def f_struct(net: Network, mask: np.ndarray) -> tuple[float, dict]:
    c = structure_components(net, mask)
    return float(np.mean(list(c.values()))), c


# ---------------------------------------------------------------- paths
MAX_EXACT_NODES = 400
N_SOURCES = 300


def _sources(net: Network, seed: int) -> np.ndarray:
    if net.n <= MAX_EXACT_NODES:
        return np.arange(net.n)
    return np.sort(np.random.default_rng(seed).choice(net.n, N_SOURCES, replace=False))


def path_distances(net: Network, mask, src: np.ndarray, proximity="jaccard") -> np.ndarray:
    return dijkstra(net.csr(net.distance(proximity), mask), directed=False, indices=src)


def f_path(net: Network, mask: np.ndarray, eps: float = 0.1, proximity="jaccard",
           seed: int = 0, DG: np.ndarray | None = None) -> tuple[float, dict]:
    src = _sources(net, seed)
    if DG is None:
        DG = path_distances(net, None, src, proximity)
    DB = path_distances(net, mask, src, proximity)
    valid = np.isfinite(DG) & (DG > 0)  # pairs connected in G, excluding self
    ok = valid & (DB <= (1.0 + eps) * DG + 1e-12)
    conn = valid & np.isfinite(DB)
    stretch = DB[conn] / DG[conn]
    per_source = ok.sum(1) / np.maximum(valid.sum(1), 1)  # one value per source node
    return float(ok.sum() / valid.sum()), {
        "samples": per_source,
        "median_stretch": float(np.median(stretch)) if stretch.size else float("inf"),
        "disconnected_share": float(1.0 - conn.sum() / valid.sum()),
        "n_pairs": int(valid.sum()),
    }


# ---------------------------------------------------------------- ranks
def eigenvector_centrality(net: Network, mask, proximity="jaccard") -> np.ndarray:
    A = net.csr(net.proximity(proximity), mask).astype(float)
    if A.nnz == 0:
        return np.zeros(net.n)
    if net.n <= 1500:  # dense eigh is exact; avoids iterative-solver noise in ranks
        vecs = np.linalg.eigh(A.toarray())[1]
        v = np.abs(vecs[:, -1])
    else:
        v = np.abs(sla.eigsh(A, k=1, which="LA", v0=np.ones(net.n), tol=1e-12)[1][:, 0])
    v = np.round(v / v.max(), 9)  # structurally equivalent nodes must tie exactly (float noise otherwise breaks ranks)
    return np.where(v < 1e-7, 0.0, v)  # noise outside the dominant component -> ties


def betweenness(net: Network, mask, proximity="jaccard") -> np.ndarray:
    g = net.to_networkx(mask)
    idx = np.arange(net.m) if mask is None else np.flatnonzero(mask)
    d = net.distance(proximity)
    for e in idx:
        g[int(net.u[e])][int(net.v[e])]["dist"] = float(d[e])
    bc = nx.betweenness_centrality(g, weight="dist", normalized=True)
    return np.round(np.array([bc[i] for i in range(net.n)]), 12)


def centralities(net: Network, mask, proximity="jaccard") -> dict:
    return {
        "strength": net.strength(mask),
        "eigenvector": eigenvector_centrality(net, mask, proximity),
        "betweenness": betweenness(net, mask, proximity),
    }


def _rho(a, b) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 1.0 if np.ptp(a) == 0 and np.ptp(b) == 0 else 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if np.isnan(r) else max(0.0, float(r))


def f_rank(net: Network, mask, proximity="jaccard", cG: dict | None = None) -> tuple[float, dict]:
    cG = cG or centralities(net, None, proximity)
    cB = centralities(net, mask, proximity)
    parts = {k: _rho(cG[k], cB[k]) for k in cG}
    k10 = max(1, int(round(0.1 * net.n)))
    overlap = float(np.mean([
        len(set(np.argsort(-cG[k])[:k10]) & set(np.argsort(-cB[k])[:k10])) / k10 for k in cG]))
    return float(np.mean(list(parts.values()))), {**parts, "top10_overlap": overlap}


# ---------------------------------------------------------------- communities
N_SEEDS = 20


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information (arithmetic-mean normalisation, same as scikit-learn), vectorised."""
    n = len(a)
    ka, kb = a.max() + 1, b.max() + 1
    c = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(float)
    pa, pb = c.sum(1) / n, c.sum(0) / n
    nz = c > 0
    pij = c[nz] / n
    mi = float(np.sum(pij * np.log(pij / np.outer(pa, pb)[nz])))
    ha = -float(np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = -float(np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    if ha == 0 and hb == 0:
        return 1.0
    return mi / ((ha + hb) / 2) if (ha + hb) > 0 else 0.0


def louvain_partitions(net: Network, mask, seeds=range(N_SEEDS), resolution: float = 1.0) -> list[np.ndarray]:
    g = net.to_networkx(mask)  # isolates stay in as singletons
    out = []
    for s in seeds:
        comms = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=int(s))
        lab = np.empty(net.n, dtype=np.int64)
        for c, nodes in enumerate(comms):
            lab[list(nodes)] = c
        out.append(lab)
    return out


def self_agreement(parts: list[np.ndarray]) -> float:
    vals = [nmi(a, b) for i, a in enumerate(parts) for j, b in enumerate(parts) if i != j]
    return float(np.mean(vals))


def f_comm(net: Network, mask, PG: list[np.ndarray] | None = None, ceiling: float | None = None,
           resolution: float = 1.0) -> tuple[float, dict]:
    PG = PG or louvain_partitions(net, None, resolution=resolution)
    ceiling = ceiling or self_agreement(PG)
    PB = louvain_partitions(net, mask, resolution=resolution)
    per_seed = np.array([np.mean([nmi(a, b) for b in PB]) for a in PG])  # one value per G partition
    raw = float(per_seed.mean())
    info = {"samples": np.minimum(1.0, per_seed / ceiling), "nmi_raw": raw, "ceiling": ceiling, "n_comm_B": float(np.mean([p.max() + 1 for p in PB]))}
    if net.communities is not None:
        info["ari_G_planted"] = float(np.mean([adjusted_rand_score(net.communities, p) for p in PG]))
        info["ari_B_planted"] = float(np.mean([adjusted_rand_score(net.communities, p) for p in PB]))
    return min(1.0, raw / ceiling), info


# ---------------------------------------------------------------- spreading
@njit(cache=True)
def _mix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit(cache=True)
def _uniform(seed, run, step, edge, direction):
    h = _mix(np.uint64(seed))
    h = _mix(h ^ np.uint64(run))
    h = _mix(h ^ np.uint64(step))
    h = _mix(h ^ np.uint64(edge * 2 + direction))
    return float(h >> np.uint64(11)) / 9007199254740992.0


@njit(cache=True)
def si_runs(indptr, indices, eids, dirs, probs, n, seed_node, run_ids, max_steps, rng_seed):
    """Return infection times (len(run_ids) x n); never-infected nodes get max_steps."""
    out = np.full((len(run_ids), n), float(max_steps))
    infected = np.zeros(n, dtype=np.bool_)
    newly = np.empty(n, dtype=np.int64)
    allinf = np.empty(n, dtype=np.int64)
    for ri in range(len(run_ids)):
        run = run_ids[ri]
        infected[:] = False
        infected[seed_node] = True
        out[ri, seed_node] = 0.0
        na = 1
        allinf[0] = seed_node
        for step in range(1, max_steps + 1):
            nn = 0
            for a in range(na):  # every infected node retries each step (SI)
                i = allinf[a]
                for p in range(indptr[i], indptr[i + 1]):
                    j = indices[p]
                    if infected[j]:
                        continue
                    if _uniform(rng_seed, run, step, eids[p], dirs[p]) < probs[p]:
                        infected[j] = True
                        out[ri, j] = step
                        newly[nn] = j
                        nn += 1
            for b in range(nn):
                allinf[na] = newly[b]
                na += 1
            if na == n:
                break
    return out


def build_adj(net: Network, mask, p_edge: np.ndarray, beta: float):
    """CSR adjacency over kept edges, carrying original edge ids and direction bits."""
    idx = np.arange(net.m) if mask is None else np.flatnonzero(mask)
    src = np.concatenate([net.u[idx], net.v[idx]])
    dst = np.concatenate([net.v[idx], net.u[idx]])
    eid = np.concatenate([idx, idx]).astype(np.int64)
    dr = np.concatenate([np.zeros(len(idx), np.int64), np.ones(len(idx), np.int64)])
    pr = np.minimum(1.0, beta * np.concatenate([p_edge[idx], p_edge[idx]]))
    order = np.argsort(src, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=net.n))]).astype(np.int64)
    return indptr, dst[order].astype(np.int64), eid[order], dr[order], pr[order]


def half_time(times: np.ndarray, n: int) -> np.ndarray:
    h = int(np.ceil(n / 2))
    return np.partition(times, h - 1, axis=1)[:, h - 1]


def choose_seeds(net: Network, proximity="jaccard", frac: float = 0.05) -> np.ndarray:
    """Pereira protocol: 5% most, 5% median, 5% least central by eigenvector centrality in G."""
    ec = eigenvector_centrality(net, None, proximity)
    order = np.argsort(-ec, kind="stable")
    k = max(1, int(round(frac * net.n)))
    mid = net.n // 2
    return np.concatenate([order[:k], order[mid - k // 2: mid - k // 2 + k], order[-k:]]).astype(np.int64)


def simulate(net: Network, mask, seeds, beta, n_runs, max_steps, rng_seed, proximity="jaccard"):
    """Infection times, shape (n_seeds, n_runs, n)."""
    adj = build_adj(net, mask, net.proximity(proximity), beta)
    runs = np.arange(n_runs, dtype=np.int64)
    return np.stack([si_runs(*adj, net.n, int(s), runs, max_steps, rng_seed) for s in seeds])


def calibrate_beta(net: Network, seeds, target: float = 20.0, rng_seed: int = 0, proximity="jaccard",
                   n_runs: int = 10, max_seeds: int = 12) -> float:
    """Geometric bisection so the median half-infection time on G is ~target steps.

    Uses a thin subsample of seeds and runs (calibration only); runs are capped at 10x target steps,
    which is enough because anything above the target already steers the search upwards.
    """
    sub = seeds[:: max(1, len(seeds) // max_seeds)]
    lo, hi = 1e-3, 1e4
    beta = 1.0
    for _ in range(25):
        beta = float(np.sqrt(lo * hi))
        t = simulate(net, None, sub, beta, n_runs, int(target * 10), rng_seed, proximity)
        med = np.median(half_time(t.reshape(-1, net.n), net.n))
        if abs(med - target) <= 1.0:
            break
        if med > target:
            lo = beta
        else:
            hi = beta
    return beta


def f_spread(net: Network, mask, seeds, beta, n_runs: int = 100, target: float = 20.0,
             rng_seed: int = 0, proximity="jaccard", TG: np.ndarray | None = None) -> tuple[float, dict]:
    """F_spread = mean(F_order, F_time). TG caches G's infection times (seeds x runs x n)."""
    max_steps = int(10 * target)
    if TG is None:
        TG = simulate(net, None, seeds, beta, n_runs, max_steps, rng_seed, proximity)
    TB = simulate(net, mask, seeds, beta, n_runs, max_steps, rng_seed, proximity)
    f_order, f_time = [], []
    for sG, sB in zip(TG, TB):
        a, b = sG.mean(0), sB.mean(0)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            f_order.append(1.0 if np.array_equal(a, b) else 0.0)
        else:
            r = spearmanr(a, b).correlation
            f_order.append(0.0 if np.isnan(r) else max(0.0, float(r)))
        tG, tB = np.median(half_time(sG, net.n)), np.median(half_time(sB, net.n))
        f_time.append(float(np.clip(tG / tB, 0.0, 1.0)))
    fo, ft = np.array(f_order), np.array(f_time)
    return float((fo.mean() + ft.mean()) / 2), {"F_order": fo, "F_time": ft}


# ---------------------------------------------------------------- SIR robustness check (Reed-Frost)
@njit(cache=True)
def sir_final_sizes(indptr, indices, eids, dirs, probs, n, run_ids, max_steps, rng_seed):
    """Reed-Frost SIR: infected nodes transmit for one step then recover. One random seed node per run
    (the same for every backbone, via the hash). Returns the final outbreak fraction of each run."""
    out = np.zeros(len(run_ids))
    infected = np.zeros(n, dtype=np.bool_)
    cur = np.empty(n, dtype=np.int64)
    nxt = np.empty(n, dtype=np.int64)
    for ri in range(len(run_ids)):
        run = run_ids[ri]
        infected[:] = False
        s = int(_mix(np.uint64(rng_seed) ^ np.uint64(run) * np.uint64(0x9E3779B97F4A7C15)) % np.uint64(n))
        infected[s] = True
        cur[0] = s
        nc = 1
        total = 1
        for step in range(1, max_steps + 1):
            nn = 0
            for a in range(nc):
                i = cur[a]
                for p in range(indptr[i], indptr[i + 1]):
                    j = indices[p]
                    if infected[j]:
                        continue
                    if _uniform(rng_seed, run, step, eids[p], dirs[p]) < probs[p]:
                        infected[j] = True
                        nxt[nn] = j
                        nn += 1
            if nn == 0:
                break
            for b in range(nn):
                cur[b] = nxt[b]
            nc = nn
            total += nn
        out[ri] = total / n
    return out


SIR_LEVELS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0)  # multiples of the calibrated SI beta


def sir_profile(net: Network, mask, beta: float, n_runs: int = 100, rng_seed: int = 0, proximity="jaccard") -> np.ndarray:
    """Mean final outbreak fraction at each transmission level (from no outbreak to saturation on G)."""
    p = net.proximity(proximity)
    runs = np.arange(n_runs, dtype=np.int64)
    out = []
    for lv in SIR_LEVELS:
        adj = build_adj(net, mask, p, beta * lv)
        out.append(sir_final_sizes(*adj, net.n, runs, 100, rng_seed).mean())
    return np.array(out)


def f_sir(net: Network, mask, beta: float, n_runs: int = 100, rng_seed: int = 0, proximity="jaccard",
          PG_profile: np.ndarray | None = None) -> tuple[float, dict]:
    """F_SIR = 1 - mean absolute difference of mean final outbreak fractions over the transmission levels."""
    prof_g = PG_profile if PG_profile is not None else sir_profile(net, None, beta, n_runs, rng_seed, proximity)
    prof_b = sir_profile(net, mask, beta, n_runs, rng_seed, proximity)
    return float(1.0 - np.mean(np.abs(prof_g - prof_b))), {"profile_G": prof_g, "profile_B": prof_b}


def choose_seeds_uniform(net: Network, frac: float = 0.15, seed: int = 0) -> np.ndarray:
    """Alternative seed protocol for robustness: uniformly random nodes."""
    k = max(3, int(round(frac * net.n)))
    return np.sort(np.random.default_rng(seed).choice(net.n, k, replace=False)).astype(np.int64)
