import networkx as nx
import numpy as np
import pandas as pd
import pytest

from backbone.backbones import edge_table
from backbone.diagnose import (bridge_retention, dissociation_summary, louvain_labels, matched_x_correlations, merged_curves,
                               retention_table, sparse_regime)
from backbone.sweep import Network


def two_cliques():
    """Two triangles joined by one bridge (3-4); proximities make the bridge the weakest edge."""
    G = nx.Graph()
    for u, v in [(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6)]:
        G.add_edge(u, v, proximity=0.8, distance=0.25)
    G.add_edge(3, 4, proximity=0.1, distance=9.0)
    return edge_table(G)


def test_louvain_finds_the_two_cliques():
    net = Network(two_cliques())
    lab = louvain_labels(net, n_partitions=3)[0]
    assert len(set(lab)) == 2 and lab[0] == lab[1] == lab[2] and lab[3] == lab[4] == lab[5] and lab[0] != lab[3]


def test_bridge_retention_counts_the_bridge():
    net = Network(two_cliques())
    lab = louvain_labels(net, 1)[0]
    bridge = int(np.where(net.p == 0.1)[0][0])
    assert bridge_retention(net, lab, [bridge]) == 1.0
    assert bridge_retention(net, lab, [k for k in range(net.m) if k != bridge]) == 0.0


def test_retention_table_weight_threshold_drops_the_bridge_first():
    net = Network(two_cliques())
    t = retention_table(net, [6, 7], methods=["W"], labels=louvain_labels(net, 3)).set_index("n_edges")
    assert t.loc[6, "bridge_retention"] == 0.0 and t.loc[7, "bridge_retention"] == 1.0
    assert t.loc[6, "components"] == 2 and t.loc[7, "components"] == 1


def _curves():
    si = pd.DataFrame({"method": ["A"] * 4 + ["B"] * 4, "n_edges": [1, 2, 3, 4] * 2, "X": [0.1, 0.4, 0.7, 1.0] * 2,
                       "rho_t": [0.5, 0.7, 0.95, 0.99, 0.95, 0.96, 0.97, 0.99], "xi": 0.5})
    rho = pd.DataFrame({"method": ["A"] * 4 + ["B"] * 4, "n_edges": [1, 2, 3, 4] * 2,
                        "rho_C": [0.95, 0.95, 0.96, 1.0, 0.5, 0.6, 0.97, 1.0], "components": 1})
    return merged_curves(rho, si)


def test_dissociation_flags_structure_ok_function_bad():
    d = dissociation_summary(_curves()).set_index("method")
    assert d.loc["A", "structure_ok_function_bad"] == pytest.approx(0.5)   # budgets 0.1 and 0.4
    assert d.loc["B", "structure_ok_function_bad"] == 0
    assert d.loc["B", "function_ok_structure_bad"] == pytest.approx(0.5)


def test_sparse_regime_selects_low_x():
    s = sparse_regime(_curves(), 0.4)
    assert set(s.X) == {0.1, 0.4}


def test_matched_x_correlations_sign():
    rows = []
    for X in [0.1, 0.2, 0.3]:
        for m, comps in zip("ABCD", [1, 2, 3, 4]):
            rows.append((m, X, comps, 0.9 - 0.1 * comps + X * 0.01, 0.5 + 0.1 * comps, 0.3 * comps, 0.5))
    df = pd.DataFrame(rows, columns=["method", "X", "components", "rho_t", "rho_C", "bridge_retention", "strength_rho"])
    c = matched_x_correlations(df).set_index(["variable", "versus"])
    assert c.loc[("components", "rho_t"), "mean_spearman"] == pytest.approx(-1.0)   # more components, lower rho_t
    assert c.loc[("components", "rho_C"), "mean_spearman"] == pytest.approx(1.0)
    assert c.loc[("components", "rho_t"), "n_budgets"] == 3
