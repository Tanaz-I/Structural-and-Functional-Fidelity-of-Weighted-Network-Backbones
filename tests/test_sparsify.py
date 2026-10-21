import networkx as nx
import numpy as np
import pandas as pd
import pytest

from backbone.backbones import edge_table
from backbone.data import NETWORKS, load_graph

RAW_HOSPITAL = NETWORKS["hospital"][0]
from backbone.sparsify import (METHODS, disparity_alpha, edges_to_keep, importance, is_connected,
                               ranking, sparsify, weighted_effective_resistance)


def toy_table():
    G = nx.Graph()
    for u, v, d in [(1, 2, 1.0), (2, 3, 1.0), (1, 3, 3.0), (3, 4, 2.0)]:
        G.add_edge(u, v, distance=d, proximity=1 / (1 + d))
    return edge_table(G)


def test_W_orders_by_proximity():
    t = toy_table()
    order = t.iloc[ranking(t, "W")]["proximity"].to_numpy()
    assert (np.diff(order) <= 0).all()                     # strongest first


def test_SmD_and_DBS_keep_triangle_shortcut_last():
    t = toy_table()
    for m in ("SmD", "DBS"):
        last = t.iloc[ranking(t, m)[-1]]
        assert (last.u, last.v) == (1, 3)                  # the only semi-metric edge goes first


def test_disparity_formula_by_hand():
    t = pd.DataFrame({"u": [1, 1, 2], "v": [2, 3, 3], "proximity": [0.5, 0.2, 0.1]})
    # all nodes have degree 2 -> F(p) = p ; alpha = 1 - min(p, p) = 1 - p
    assert disparity_alpha(t) == pytest.approx([0.5, 0.8, 0.9])
    # degree-1 node: only the other endpoint counts
    t2 = pd.DataFrame({"u": [1, 2], "v": [2, 3], "proximity": [0.3, 0.4]})
    a = disparity_alpha(t2)
    assert a[0] == pytest.approx(1 - 0.3) and a[1] == pytest.approx(1 - 0.4)


def test_effective_resistance_properties():
    t = toy_table()
    x = weighted_effective_resistance(t)
    assert x.sum() == pytest.approx(len(set(t.u) | set(t.v)) - 1)   # leverage scores sum to n-1
    bridge = t[(t.u == 3) & (t.v == 4)].index[0]
    assert x[bridge] == pytest.approx(1.0)                           # bridge: leverage exactly 1


def test_sparsify_sizes_and_seed():
    t = toy_table()
    for m in METHODS:
        assert len(sparsify(t, m, 1.0)) == len(t) and len(sparsify(t, m, 0.0)) == 3
        assert importance(t, m).shape == (len(t),)
    assert (ranking(t, "SmD", seed=1) == ranking(t, "SmD", seed=1)).all()


@pytest.mark.skipif(not RAW_HOSPITAL.exists(), reason="raw hospital data not present")
def test_hospital_connectivity_guarantees():
    G = load_graph(RAW_HOSPITAL)
    t = edge_table(G)
    nodes = sorted(G)
    n, m = 75, 1139
    assert weighted_effective_resistance(t).sum() == pytest.approx(n - 1)
    for X in (0.0, 0.05, 0.3, 0.7):                       # DBS: connected all the way to the MST
        assert is_connected(sparsify(t, "DBS", X), nodes)
    assert is_connected(sparsify(t, "SmD", (217 - 74) / (m - 74)), nodes)   # SmD: down to metric backbone size
    # the same m' for every method
    k = edges_to_keep(n, m, 0.3)
    assert all(len(sparsify(t, mm, 0.3)) == k for mm in METHODS)
