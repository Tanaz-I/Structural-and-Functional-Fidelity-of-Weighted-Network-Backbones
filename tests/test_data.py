import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import dijkstra

from backbone.data import NETWORKS, build_graph, edge_table, load_contacts, load_graph

RAW_HOSPITAL = NETWORKS["hospital"][0]


def test_edge_table_formula_on_toy_contacts():
    # 1-2 meet twice, 2-3 once. r_11=2, r_22=3, r_33=1
    c = pd.DataFrame({"t": [20, 40, 60], "i": [1, 2, 3], "j": [2, 1, 2], "Ai": "A", "Aj": "A"})
    e = edge_table(c).set_index(["u", "v"])
    assert e.loc[(1, 2), "r"] == 2 and e.loc[(2, 3), "r"] == 1
    assert e.loc[(1, 2), "proximity"] == pytest.approx(2 / (2 + 3 - 2))
    assert e.loc[(2, 3), "proximity"] == pytest.approx(1 / (3 + 1 - 1))
    assert e.loc[(1, 2), "distance"] == pytest.approx(1 / (2 / 3) - 1)


@pytest.mark.skipif(not RAW_HOSPITAL.exists(), reason="raw hospital data not present")
def test_hospital_matches_paper_table1():
    G = load_graph(RAW_HOSPITAL)
    assert (G.number_of_nodes(), G.number_of_edges()) == (75, 1139)
    assert nx.is_connected(G)
    p = np.array([d["proximity"] for *_, d in G.edges(data=True)])
    assert 0 < p.min() and p.max() < 1
    # Table 1: metric backbone of hospital = 13.43% adjusted size (217 edges)
    nodes = sorted(G)
    idx = {n: k for k, n in enumerate(nodes)}
    D = np.zeros((75, 75))
    for a, b, d in G.edges(data="distance"):
        D[idx[a], idx[b]] = D[idx[b], idx[a]] = d
    sp = dijkstra(D, directed=False)  # zero entries are non-edges in the dense input
    metric = sum(d <= sp[idx[a], idx[b]] * (1 + 1e-9) for a, b, d in G.edges(data="distance"))
    assert (metric - 74) / (1139 - 74) == pytest.approx(0.1343, abs=1e-4)


def _metric_X_percent(path):
    e = edge_table(load_contacts(path))
    nodes = sorted(set(e.u) | set(e.v))
    idx = {n: k for k, n in enumerate(nodes)}
    a, b, d = e.u.map(idx).to_numpy(), e.v.map(idx).to_numpy(), e.distance.to_numpy()
    D = np.zeros((len(nodes), len(nodes)))
    D[a, b] = D[b, a] = d
    sp = dijkstra(D, directed=False)  # zero entries are non-edges
    k = int((d <= sp[a, b] * (1 + 1e-9)).sum())
    return 100 * (k - (len(nodes) - 1)) / (len(e) - (len(nodes) - 1))


def test_hospital_metric_size_matches_table1():
    path, n, m, x_paper = NETWORKS["hospital"]
    assert _metric_X_percent(path) == pytest.approx(x_paper, abs=0.01)


def test_workplace1_has_the_papers_size_but_a_smaller_metric_backbone():
    # documented difference: same nodes and edges as Table 1, metric size 13.25% against the paper's 20.63%
    path, n, m, x_paper = NETWORKS["workplace1"]
    G = load_graph(path)
    assert (G.number_of_nodes(), G.number_of_edges()) == (n, m)
    x = _metric_X_percent(path)
    assert x == pytest.approx(13.25, abs=0.01) and abs(x - x_paper) > 0.5
