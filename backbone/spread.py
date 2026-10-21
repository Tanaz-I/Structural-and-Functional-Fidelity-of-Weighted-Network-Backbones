"""SI spreading and its fidelity measures rho_t and xi^-1_0.5 (paper Sec. 5.3.2).

Model: one seed node starts infected; every step each infected node infects each susceptible neighbour with
probability beta * p_ij (synchronous update). Infection times are recorded; seed = time 0.

Conventions where the paper is silent (all documented in docs/RESULTS.md):
  * beta is calibrated so that the median half-infection time on the FULL graph is `target_t_half` steps.
  * a node never infected (other component, or step cap) gets time T_MAX; a run stops once the seed's whole
    connected component is infected, or at T_MAX.
  * half-infection time t_0.5 = first step at which ceil(n/2) nodes are infected (T_MAX if never).
  * rho_t excludes the seed node (its time is 0 in every network, so it carries no information).
  * xi^-1 is formed per run (original run r / sparsified run r), then the median over the 100 runs is taken.
Seeds: 15% of nodes, taken from the original network's eigenvector-centrality ranking
(top 5%, median 5%, bottom 5%), as in Supplement C.5.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from .backbones import adjusted_size
from .sparsify import METHODS, edges_to_keep, ranking
from .sweep import eigenvector_centrality

T_MAX = 1000
RUNS = 100


def choose_seeds(centrality, frac=0.15):
    """Indices of the seed nodes: top / median / bottom groups of the original eigenvector centrality."""
    n = len(centrality)
    total = round(frac * n)
    k_top = k_bot = round(0.05 * n)  # 5% most and least central; the median group takes the remainder
    k_mid = total - k_top - k_bot
    order = np.argsort(-centrality, kind="stable")
    mid0 = n // 2 - k_mid // 2
    return np.concatenate([order[:k_top], order[mid0:mid0 + k_mid], order[n - k_bot:]]).astype(int)


def simulate(A, seed, beta, runs, rng, t_max=T_MAX):
    """Infection times, shape (runs, n). A is the dense proximity matrix of the (sub)graph."""
    n = len(A)
    L = np.log1p(-np.minimum(beta * A, 1 - 1e-12))  # log(1 - beta p_ij); 0 for non-edges
    comp = connected_components(csr_matrix(A > 0), directed=False)[1]
    reach = comp == comp[seed]
    inf = np.zeros((runs, n), dtype=bool)
    inf[:, seed] = True
    times = np.full((runs, n), t_max, dtype=np.int32)
    times[:, seed] = 0
    for t in range(1, t_max + 1):
        p = 1.0 - np.exp(inf.astype(float) @ L)  # P(susceptible node infected this step)
        new = ~inf & (rng.random((runs, n)) < p)
        times[new] = t
        inf |= new
        if inf[:, reach].all():
            break
    return times


def half_time(times):
    """Per-run t_0.5: first step with ceil(n/2) infected nodes (T_MAX if never)."""
    n = times.shape[1]
    return np.sort(times, axis=1)[:, int(np.ceil(0.5 * n)) - 1].astype(float)


def calibrate_beta(A, seeds, target_t_half=20.0, runs=RUNS, rng_seed=0, iters=25):
    """Bisection on beta so that the mean over seeds of the median t_0.5 on the full graph equals the target."""
    lo, hi = 1e-4, 1.0 / A.max()  # beta * p_max <= 1

    def t_half_at(beta):
        rng = np.random.default_rng(rng_seed)
        return np.mean([np.median(half_time(simulate(A, s, beta, runs, rng))) for s in seeds])

    if t_half_at(hi) > target_t_half:  # even the largest valid beta (beta * p_max = 1) is too slow
        warnings.warn(f"target half-infection time {target_t_half} unreachable; using the largest valid beta")
        return float(hi)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if t_half_at(mid) > target_t_half else (lo, mid)
    return float(np.sqrt(lo * hi))


def reference_runs(A, seeds, beta, runs=RUNS, rng_seed=0):
    """Original-network infection times per seed (computed once and reused for every sparsification)."""
    rng = np.random.default_rng(rng_seed)
    return {int(s): simulate(A, s, beta, runs, rng) for s in seeds}


def spreading_fidelity(A_sub, seeds, beta, ref, runs=RUNS, rng_seed=1):
    """rho_t and xi^-1_0.5 per seed for one sparsified graph, as arrays aligned with `seeds`."""
    rng = np.random.default_rng(rng_seed)
    rho, xi = [], []
    for s in seeds:
        t_ref = ref[int(s)]
        t_sub = simulate(A_sub, s, beta, runs, rng)
        mask = np.arange(t_ref.shape[1]) != s
        a, b = t_ref.mean(0)[mask], t_sub.mean(0)[mask]
        # constant times (e.g. the seed reaches nothing) -> rank correlation undefined -> no agreement (0)
        rho.append(0.0 if np.ptp(a) == 0 or np.ptp(b) == 0 else spearmanr(a, b)[0])
        xi.append(np.median(half_time(t_ref) / half_time(t_sub)))
    return np.array(rho), np.array(xi)


def x_grid(net, n_points=21):
    """X values 0, 0.05, ..., 1 plus the metric backbone size; returned as sorted edge counts."""
    ks = {edges_to_keep(net.n, net.m, x) for x in np.linspace(0, 1, n_points)}
    return sorted(ks)


def si_setup(net, target_t_half=20.0, runs=RUNS, seed=0):
    """Seeds, calibrated beta and the original-network reference runs (shared by every sparsification)."""
    A_full = net.adjacency()
    seeds = choose_seeds(eigenvector_centrality(A_full))
    beta = calibrate_beta(A_full, seeds, target_t_half, runs, rng_seed=seed)
    return seeds, beta, reference_runs(A_full, seeds, beta, runs, rng_seed=seed)


def achieved_half_time(ref, seeds):
    """Mean over seeds of the median half-infection time on the full graph (what beta was calibrated to hit)."""
    return float(np.mean([np.median(half_time(ref[int(s)])) for s in seeds]))


def si_sweep(net, methods=METHODS, ks=None, target_t_half=20.0, runs=RUNS, seed=0, extra_ks=(), setup=None, orders=None):
    """Mean/std over seeds of rho_t and xi^-1_0.5 for each method and sparsification level.
    `setup` = (seeds, beta, ref) from si_setup, to share one calibration between sweeps.
    `orders` optionally maps a method name to a ready-made edge order (most important first)."""
    seeds, beta, ref = setup if setup is not None else si_setup(net, target_t_half, runs, seed)
    ks = sorted(set(ks if ks is not None else x_grid(net)) | set(extra_ks))
    rows = []
    for mi, method in enumerate(methods):
        order = orders[method] if orders and method in orders else ranking(net.table, method, seed)
        for k in ks:
            rho, xi = spreading_fidelity(net.adjacency(order[:k]), seeds, beta, ref, runs,
                                         rng_seed=1000 * (mi + 1) + k)
            rows.append((method, k, adjusted_size(net.n, net.m, k), rho.mean(), rho.std(), xi.mean(), xi.std()))
    df = pd.DataFrame(rows, columns=["method", "n_edges", "X", "rho_t", "rho_t_std", "xi", "xi_std"])
    df.attrs["beta"], df.attrs["seeds"] = beta, seeds
    return df
