import networkx as nx
import numpy as np
import pytest

from backbone.backbones import (adjusted_size, backbone_at_lambda, edge_table, metric_edges,
                                ultrametric_edges)
from backbone.data import RAW_HOSPITAL, load_graph


def toy():
    # triangle 1-2 (d=1), 2-3 (d=1), 1-3 (d=3) plus pendant 3-4 (d=2)
    G = nx.Graph()
    for u, v, d in [(1, 2, 1.0), (2, 3, 1.0), (1, 3, 3.0), (3, 4, 2.0)]:
        G.add_edge(u, v, distance=d, proximity=1 / (1 + d))
    return G


def test_toy_triangle():
    t = edge_table(toy()).set_index(["u", "v"])
    assert t.loc[(1, 3), "distortion"] == pytest.approx(1.5)         # 3 / (1+1)
    assert t.loc[(1, 3), "lam"] == pytest.approx(np.log(2) / np.log(3), rel=1e-6)  # 3^lam = 2
    assert len(metric_edges(t.reset_index())) == 3
    assert len(ultrametric_edges(t.reset_index())) == 3              # spanning tree = 3 edges (n-1)
    assert np.isinf(t.loc[(3, 4), "lam"]) and np.isinf(t.loc[(1, 2), "lam"])


def test_nested_and_metric_is_lambda_one():
    t = edge_table(toy())
    assert set(map(tuple, backbone_at_lambda(t, 1.0)[["u", "v"]].values)) == \
        set(map(tuple, metric_edges(t)[["u", "v"]].values))
    sizes = [len(backbone_at_lambda(t, l)) for l in (0.1, 0.5, 0.7, 1, 5, 100)]
    assert sizes == sorted(sizes, reverse=True)


def test_adjusted_size():
    assert adjusted_size(75, 1139, 74) == 0 and adjusted_size(75, 1139, 1139) == 1


@pytest.mark.skipif(not RAW_HOSPITAL.exists(), reason="raw hospital data not present")
def test_hospital():
    G = load_graph()
    t = edge_table(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    assert (t["distortion"] >= 1 - 1e-9).all()
    assert len(metric_edges(t)) == 217                                  # paper Table 1: 13.43%
    assert len(ultrametric_edges(t)) == n - 1                           # unique MST (paper, Sec. 5.2)
    assert ((t["lam"] >= 1 - 1e-6) == (t["distortion"] <= 1 + 1e-9)).all()  # lam>=1  <=>  metric edge
    # bisection result agrees with a direct computation of B_lam at a few lambdas
    from backbone.backbones import _in_backbone, edge_arrays
    nodes, _, i, j, d = edge_arrays(G)
    for lam in (0.3, 0.6, 2.0):
        direct = _in_backbone(lam, n, i, j, np.log(d), range(m))
        by_table = (t["lam"].values >= lam * (1 - 1e-9))
        assert (direct == by_table).mean() > 0.999
