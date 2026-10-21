import numpy as np
import pytest

from backbone.backbones import edge_table
from backbone.data import NETWORKS, load_graph

RAW_HOSPITAL = NETWORKS["hospital"][0]
from backbone.spread import (calibrate_beta, choose_seeds, half_time, reference_runs, simulate,
                             spreading_fidelity)
from backbone.sweep import Network, eigenvector_centrality


def path_graph(n):
    A = np.zeros((n, n))
    for k in range(n - 1):
        A[k, k + 1] = A[k + 1, k] = 1.0
    return A


def test_choose_seeds_groups():
    c = np.linspace(1, 0, 75)                       # node 0 most central, node 74 least
    s = choose_seeds(c)
    assert len(s) == 11 and len(set(s)) == 11       # 15% of 75
    assert list(s[:4]) == [0, 1, 2, 3] and list(s[-4:]) == [71, 72, 73, 74]
    assert all(30 <= m <= 44 for m in s[4:7])        # median group sits mid-ranking


def test_simulate_deterministic_path():
    t = simulate(path_graph(5), 0, beta=1.0, runs=7, rng=np.random.default_rng(0))
    assert (t == np.arange(5)).all()                 # beta*p = 1 -> infection time = hop distance


def test_unreachable_nodes_get_tmax():
    A = path_graph(6)
    A[2, 3] = A[3, 2] = 0                            # two components of size 3
    t = simulate(A, 0, beta=1.0, runs=4, rng=np.random.default_rng(0), t_max=50)
    assert (t[:, :3] == [0, 1, 2]).all() and (t[:, 3:] == 50).all()
    assert (half_time(t) == 2).all()                 # ceil(6/2)=3 infected reached at step 2


def test_half_time():
    t = np.array([[0, 1, 2, 3], [0, 5, 6, 7]])
    assert list(half_time(t)) == [1, 5]              # 2nd smallest time


def test_identical_graph_same_rng_gives_perfect_fidelity():
    A = path_graph(8) * 0.6
    ref = reference_runs(A, [0, 4], beta=1.0, runs=30, rng_seed=5)
    rho, xi = spreading_fidelity(A, [0, 4], 1.0, ref, runs=30, rng_seed=5)
    assert rho == pytest.approx([1, 1]) and xi == pytest.approx([1, 1])


def test_calibrate_beta_hits_target():
    n = 30
    A = np.zeros((n, n))
    for k in range(n):
        A[k, (k + 1) % n] = A[(k + 1) % n, k] = 0.5   # ring
    beta = calibrate_beta(A, [0, 10], target_t_half=10.0, runs=50)
    rng = np.random.default_rng(0)
    got = np.mean([np.median(half_time(simulate(A, s, beta, 50, rng))) for s in (0, 10)])
    assert got == pytest.approx(10.0, rel=0.15) and beta * A.max() <= 1


@pytest.mark.skipif(not RAW_HOSPITAL.exists(), reason="raw hospital data not present")
def test_hospital_smoke():
    net = Network(edge_table(load_graph(RAW_HOSPITAL)))
    A = net.adjacency()
    seeds = choose_seeds(eigenvector_centrality(A))
    beta = calibrate_beta(A, seeds, 20.0, runs=30, iters=12)
    got = np.mean([np.median(half_time(simulate(A, s, beta, 30, np.random.default_rng(0)))) for s in seeds])
    assert got == pytest.approx(20.0, rel=0.2)
    ref = reference_runs(A, seeds, beta, runs=30)
    rho, xi = spreading_fidelity(A, seeds, beta, ref, runs=30)   # full graph vs itself (independent runs)
    assert rho.mean() > 0.9 and 0.85 < xi.mean() < 1.15
