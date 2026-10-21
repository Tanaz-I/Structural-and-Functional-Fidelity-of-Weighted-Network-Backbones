import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import dijkstra

from backbone.data import RAW_HOSPITAL, build_graph, edge_table, load_contacts, load_graph


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
    G = load_graph()
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
