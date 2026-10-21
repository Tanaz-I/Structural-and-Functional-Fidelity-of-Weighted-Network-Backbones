import networkx as nx
import numpy as np
import pytest

from backbone.backbones import edge_table
from backbone.data import NETWORKS, load_graph

RAW_HOSPITAL = NETWORKS["hospital"][0]
from backbone.sparsify import METHODS
from backbone.sweep import Network, eigenvector_centrality, rho_C, sweep


def toy_table():
    G = nx.Graph()
    for u, v, d in [(1, 2, 1.0), (2, 3, 1.0), (1, 3, 3.0), (3, 4, 2.0), (4, 5, 0.5)]:
        G.add_edge(u, v, distance=d, proximity=1 / (1 + d))
    return edge_table(G)


def test_eigenvector_centrality_matches_networkx_toy():
    net = Network(toy_table())
    G = nx.Graph()
    for r in net.table.itertuples():
        G.add_edge(r.u, r.v, weight=r.proximity)
    ref = nx.eigenvector_centrality_numpy(G, weight="weight")
    ours = eigenvector_centrality(net.adjacency())
    assert ours / ours.sum() == pytest.approx([ref[n] for n in net.nodes] / np.sum(list(ref.values())), abs=1e-8)


def test_rho_C_edge_cases():
    c = np.array([3.0, 2.0, 1.0, 0.5])
    assert rho_C(c, c) == pytest.approx(1.0)
    assert rho_C(c, c[::-1]) == pytest.approx(-1.0)
    assert rho_C(c, 10 * c + 1) == pytest.approx(1.0)       # only the ranking matters


def test_sweep_endpoints_and_grid():
    net = Network(toy_table())
    df = sweep(net)
    for m in METHODS:
        d = df[df.method == m]
        assert d.n_edges.min() == net.n - 1 and d.n_edges.max() == net.m
        assert d.X.min() == 0 and d.X.max() == 1
        assert d[d.X == 1].rho_C.iloc[0] == pytest.approx(1.0)   # full graph preserves itself
        assert d[d.X == 0].components.iloc[0] >= 1


@pytest.mark.skipif(not RAW_HOSPITAL.exists(), reason="raw hospital data not present")
def test_hospital_eigenvector_vs_networkx_and_endpoints():
    net = Network(edge_table(load_graph(RAW_HOSPITAL)))
    G = nx.Graph()
    for r in net.table.itertuples():
        G.add_edge(r.u, r.v, weight=r.proximity)
    ref = nx.eigenvector_centrality_numpy(G, weight="weight")
    ours = eigenvector_centrality(net.adjacency())
    r = np.array([ref[n] for n in net.nodes])
    assert rho_C(ours, np.abs(r)) == pytest.approx(1.0)
    df = sweep(net, methods=["DBS", "W"])
    assert (df[df.X == 1].rho_C.round(9) == 1.0).all()
    assert (df[(df.method == "DBS")].components == 1).all()   # DBS connected at every X
