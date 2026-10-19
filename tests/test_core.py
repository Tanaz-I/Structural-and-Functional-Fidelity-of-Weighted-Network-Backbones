"""Core tests: data and synthetic generator, six filters (incl. agreement with NetBone), budgets, metrics."""
import networkx as nx
import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from backbone import metrics
from backbone.filters import (BUDGETS, FILTERS, edge_order, k_of, metric_backbone_mask, score_disparity, score_hss,
                              score_noise_corrected, tie_permutation, top_k_mask)
from backbone.network import Network, describe, factorial_grid, make_synthetic, pilot_grid


def make_test_graph() -> nx.Graph:
    """Small connected weighted graph with integer, tied weights, so the tests need no dataset."""
    rng = np.random.default_rng(3)
    g = nx.connected_watts_strogatz_graph(80, 8, 0.1, seed=3)
    for a, b in g.edges():
        g[a][b]["weight"] = int(rng.lognormal(1.0, 1.0)) + 1
    return g


@pytest.fixture(scope="module")
def lm():
    return Network.from_networkx("test", make_test_graph())


def full(net):
    return np.ones(net.m, bool)


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def test_network_cleaning_merges_loops_parallel_and_keeps_lcc():
    net = Network.from_edges("t", [0, 0, 1, 2, 5], [1, 1, 1, 3, 6], [1.0, 2.0, 9.0, 1.0, 1.0])
    # loop 1-1 dropped, 0-1 merged (3.0); components {0,1} {2,3} {5,6} -> tie, largest picked deterministically
    assert net.m == 1 and net.n == 2 and net.w[0] == pytest.approx(3.0)


def test_synthetic_matches_targets():
    net = make_synthetic(0.3, 1.0, 0.5, seed=1)
    assert net.n >= 290 and net.n_components() == 1
    inter = net.communities[net.u] != net.communities[net.v]
    assert inter.mean() == pytest.approx(0.3, abs=0.05)  # achieved mixing
    assert net.w.mean() == pytest.approx(1.0, rel=1e-6)  # mean weight fixed


def test_weight_spread_and_coupling_move_as_intended():
    lo = describe(make_synthetic(0.3, 0.5, 0.0, 2))
    hi = describe(make_synthetic(0.3, 1.5, 0.0, 2))
    assert hi["weight_cv"] > lo["weight_cv"]
    c0 = describe(make_synthetic(0.3, 1.0, 0.0, 3))["weight_degree_coupling"]
    c1 = describe(make_synthetic(0.3, 1.0, 1.0, 3))["weight_degree_coupling"]
    assert c1 > c0 + 0.1


def test_grid_size_and_disjoint_pilot():
    main = {s for *_, s in factorial_grid()}
    assert len(main) == 270
    assert not main & {s for *_, s in pilot_grid()}


def test_syn_names_three_instances_per_cell():
    from backbone.network import syn_names
    names = syn_names()
    assert len(names) == 81 and len(set(names)) == 81
    assert all(int(n[3:]) % 10 < 3 for n in names)


# ----------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------
def test_scores_shape_and_finite(lm):
    for name, f in FILTERS.items():
        s = f(lm)
        assert s.shape == (lm.m,) and np.isfinite(s).all(), name


def test_budget_exact_k_and_nested(lm):
    perm = tie_permutation(lm.m, 0)
    for name, f in FILTERS.items():
        order = edge_order(f(lm), lm.w, perm)
        prev = np.zeros(lm.m, bool)
        for q in BUDGETS:
            mk = top_k_mask(order, k_of(q, lm.m))
            assert mk.sum() == k_of(q, lm.m) and (mk | prev).sum() == mk.sum()
            prev = mk


def test_metric_backbone_preserves_all_shortest_paths(lm):
    mask = metric_backbone_mask(lm)
    d = lm.distance()
    DG = dijkstra(lm.csr(d), directed=False)
    DB = dijkstra(lm.csr(d, mask), directed=False)
    assert np.allclose(DG, DB)
    assert mask.sum() < lm.m  # some semi-metric edges exist


def test_metric_backbone_matches_bruteforce(lm):
    d = lm.distance()
    mask = metric_backbone_mask(lm)
    brute = np.zeros(lm.m, bool)
    for e in range(lm.m):
        keep = np.ones(lm.m, bool); keep[e] = False
        ind = dijkstra(lm.csr(d, keep), directed=False, indices=lm.u[e])[lm.v[e]]
        brute[e] = d[e] <= ind + 1e-9
    assert (mask == brute).all()


def test_effective_resistance_path_triangle_and_pinv():
    from backbone.filters import effective_resistance_matrix as er
    path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0.]])
    assert np.isclose(er(path)[0, 2], 2.0) and np.isclose(er(path)[0, 1], 1.0)
    tri = np.ones((3, 3)) - np.eye(3)
    assert np.isclose(er(tri)[0, 1], 2 / 3)
    A = np.array([[0, 2, 1, 0], [2, 0, 0, 3], [1, 0, 0, 1], [0, 3, 1, 0.]])
    L = np.diag(A.sum(1)) - A
    Lp = np.linalg.pinv(L)
    R = np.array([[Lp[i, i] + Lp[j, j] - 2 * Lp[i, j] for j in range(4)] for i in range(4)])
    assert np.allclose(er(A), R)


# ----------------------------------------------------------------------
# netbone_agreement
# ----------------------------------------------------------------------
"""Validation of DF, NC, HSS against the NetBone reference on the test graph."""

try:
    import netbone
except ImportError:  # the NetBone agreement tests are skipped without it
    netbone = None
needs_netbone = pytest.mark.skipif(netbone is None, reason="netbone not installed")

JACCARD_MIN = 0.9  # fixed in protocol.yaml


def _edge_index(net, g, df):
    ids = {x: i for i, x in enumerate(g.nodes())}
    a = np.array([ids[x] for x in df.source]); b = np.array([ids[x] for x in df.target])
    keys = net.u * net.n + net.v
    return np.searchsorted(keys, np.minimum(a, b) * net.n + np.maximum(a, b))


def _jaccard(a, b):
    return (a & b).sum() / (a | b).sum()


@needs_netbone
@pytest.mark.parametrize("name,nbfun,col,asc,ours", [
    ("DF", lambda g: netbone.disparity(g), "p_value", True, score_disparity),
])
def test_statistical_filters_agree(name, nbfun, col, asc, ours):
    g = make_test_graph(); net = Network.from_networkx("test", g)
    df = nbfun(g).to_dataframe()
    idx = _edge_index(net, g, df)
    ref = np.zeros(net.m); ref[idx] = -df[col].to_numpy()  # smaller p = more important
    perm = tie_permutation(net.m, 0)
    for q in (0.1, 0.2, 0.3, 0.5):
        k = k_of(q, net.m)
        a = top_k_mask(edge_order(ours(net), net.w, perm), k)
        b = top_k_mask(edge_order(ref, net.w, perm), k)
        assert _jaccard(a, b) >= JACCARD_MIN, (name, q, _jaccard(a, b))


@needs_netbone
def test_hss_agrees():
    g = make_test_graph(); net = Network.from_networkx("test", g)
    df = netbone.high_salience_skeleton(g).to_dataframe()
    idx = _edge_index(net, g, df)
    ref = np.zeros(net.m); ref[idx] = df["salience"].to_numpy()
    ours = score_hss(net, proximity="inverse")
    assert np.corrcoef(ours, ref)[0, 1] > 0.98


@needs_netbone
def test_noise_corrected_agrees_with_exact_netbone():
    """NetBone's default NC is a binomial approximation (a different statistic); its
    approximation=False branch is the Coscia-Neffke score/sdev. We match that exactly when
    n.. = W (NetBone) and to Jaccard >= 0.9 with the authors' undirected double counting."""
    g = make_test_graph(); net = Network.from_networkx("test", g)
    df = netbone.noise_corrected(g, approximation=False).to_dataframe()
    ref = np.zeros(net.m); ref[_edge_index(net, g, df)] = (df["score"] / df["nc_sdev"]).to_numpy()
    assert np.allclose(score_noise_corrected(net, double_count=False), ref)
    perm = tie_permutation(net.m, 0)
    for q in (0.1, 0.2, 0.3, 0.5):
        k = k_of(q, net.m)
        a = top_k_mask(edge_order(score_noise_corrected(net), net.w, perm), k)
        b = top_k_mask(edge_order(ref, net.w, perm), k)
        assert _jaccard(a, b) >= 0.9


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def test_all_scores_are_one_at_q1(lm):
    m = full(lm)
    assert metrics.f_struct(lm, m)[0] == pytest.approx(1)
    assert metrics.f_path(lm, m)[0] == 1
    assert metrics.f_rank(lm, m)[0] == pytest.approx(1)
    assert metrics.f_comm(lm, m)[0] == pytest.approx(1)
    seeds = metrics.choose_seeds(lm)
    beta = metrics.calibrate_beta(lm, seeds)
    assert metrics.f_spread(lm, m, seeds, beta, n_runs=20)[0] == pytest.approx(1)


def test_scores_in_unit_interval_and_larger_budget_not_worse(lm):
    order = edge_order(FILTERS["W"](lm), lm.w, tie_permutation(lm.m, 0))
    small, big = top_k_mask(order, k_of(0.1, lm.m)), top_k_mask(order, k_of(0.7, lm.m))
    for f in (metrics.f_struct, metrics.f_path, metrics.f_rank, metrics.f_comm):
        a, b = f(lm, small)[0], f(lm, big)[0]
        assert 0 <= a <= 1 and 0 <= b <= 1 and a <= b + 1e-9, f.__name__


def test_path_fidelity_one_for_metric_backbone(lm):
    assert metrics.f_path(lm, metric_backbone_mask(lm))[0] == 1


def test_invariant_to_relabelling(lm):
    perm = np.random.default_rng(1).permutation(lm.n)
    other = Network.from_edges("p", perm[lm.u], perm[lm.v], lm.w)
    mask = top_k_mask(edge_order(FILTERS["DF"](lm), lm.w, tie_permutation(lm.m, 0)), 80)
    kept = {(min(perm[a], perm[b]), max(perm[a], perm[b])) for a, b in zip(lm.u[mask], lm.v[mask])}
    m2 = np.array([(a, b) in kept for a, b in zip(other.u, other.v)])
    assert m2.sum() == mask.sum()
    assert metrics.f_struct(lm, mask)[0] == pytest.approx(metrics.f_struct(other, m2)[0])
    assert metrics.f_path(lm, mask)[0] == pytest.approx(metrics.f_path(other, m2)[0])
    assert metrics.f_rank(lm, mask)[0] == pytest.approx(metrics.f_rank(other, m2)[0], abs=1e-6)


def test_spreading_deterministic_calibrated_and_slower_when_sparse(lm):
    seeds = metrics.choose_seeds(lm)
    beta = metrics.calibrate_beta(lm, seeds)
    TG = metrics.simulate(lm, None, seeds, beta, 20, 200, 0)
    assert np.array_equal(TG, metrics.simulate(lm, None, seeds, beta, 20, 200, 0))
    med = np.median(metrics.half_time(TG.reshape(-1, lm.n), lm.n))
    assert 10 <= med <= 40
    order = edge_order(FILTERS["W"](lm), lm.w, tie_permutation(lm.m, 0))
    TB = metrics.simulate(lm, top_k_mask(order, 60), seeds, beta, 20, 200, 0)
    assert TB.mean() >= TG.mean()


def test_fast_nmi_matches_sklearn():
    from sklearn.metrics import normalized_mutual_info_score
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.integers(0, 6, 80), rng.integers(0, 9, 80)
        assert metrics.nmi(a, b) == pytest.approx(normalized_mutual_info_score(a, b))
    one = np.zeros(10, int)
    assert metrics.nmi(one, one) == 1.0


def test_sir_profile_one_at_full_graph_and_decreases_when_sparse(lm):
    seeds = metrics.choose_seeds(lm)
    beta = metrics.calibrate_beta(lm, seeds)
    m = full(lm)
    assert metrics.f_sir(lm, m, beta, 30)[0] == pytest.approx(1.0)
    prof = metrics.sir_profile(lm, None, beta, 30)
    assert prof[0] < prof[-1] and 0 <= prof.min() and prof.max() <= 1  # outbreak grows with transmissibility
    order = edge_order(FILTERS["W"](lm), lm.w, tie_permutation(lm.m, 0))
    sparse = top_k_mask(order, 40)
    assert 0 <= metrics.f_sir(lm, sparse, beta, 30)[0] < 1

