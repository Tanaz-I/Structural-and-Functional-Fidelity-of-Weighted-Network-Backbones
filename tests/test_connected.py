import networkx as nx
import numpy as np
import pandas as pd
import pytest

from backbone.backbones import edge_table
from backbone.connected import gap_closed, verdict
from backbone.sparsify import ranking, spanning_tree_edges
from backbone.sweep import Network


def table(n=20, seed=4):
    G = nx.connected_watts_strogatz_graph(n, 4, 0.3, seed=seed)
    rng = np.random.default_rng(seed)
    for u, v in G.edges:
        p = float(rng.uniform(0.05, 0.9))
        G[u][v].update(proximity=p, distance=1 / p - 1)
    return edge_table(G)


@pytest.mark.parametrize("method", ["W+T", "W+R", "DispFilt+T", "DispFilt+R"])
def test_connected_variants_are_connected_at_every_budget_and_keep_edge_count(method):
    t = table()
    net = Network(t)
    order = ranking(t, method, seed=1)
    assert sorted(order) == list(range(net.m))                      # a permutation of all edges
    for k in range(net.n - 1, net.m + 1):
        keep = order[:k]
        assert len(keep) == k and net.n_components(keep) == 1


def test_plain_W_can_be_disconnected_at_the_tree_budget():
    t = table()
    net = Network(t)
    assert net.n_components(ranking(t, "W")[: net.n - 1]) > 1


def test_guided_tree_for_W_is_the_maximum_proximity_spanning_tree():
    t = table()
    net = Network(t)
    tree = spanning_tree_edges(t, ranking(t, "W"))
    G = nx.Graph()
    G.add_weighted_edges_from(zip(net.i.tolist(), net.j.tolist(), net.p.tolist()))
    best = sorted(w for *_, w in nx.maximum_spanning_tree(G).edges(data="weight"))
    assert len(tree) == net.n - 1
    assert sorted(net.p[tree]) == pytest.approx(best)


def test_random_tree_is_a_tree_and_depends_on_seed():
    t = table()
    net = Network(t)
    a = set(ranking(t, "W+R", seed=0)[: net.n - 1])
    b = set(ranking(t, "W+R", seed=1)[: net.n - 1])
    assert net.n_components(list(a)) == 1 and len(a) == net.n - 1
    assert a != b


def _auc(rows):
    return pd.DataFrame(rows, columns=["method", "AUC_rho_t", "AUC_xi", "AUC_rho_C"]).assign(limit="metric")


def test_gap_closed_and_verdict():
    auc = _auc([("W", 0.80, 0.70, 0.95), ("W+T", 0.85, 0.80, 0.93), ("W+R", 0.82, 0.75, 0.93), ("DBS", 0.98, 0.90, 0.91)])
    g = gap_closed(auc, "W", "W+T")
    d = g.set_index("measure")
    assert d.loc["AUC_rho_t", "gap_closed"] == pytest.approx((0.85 - 0.80) / (0.98 - 0.80))
    assert verdict(g, {"AUC_rho_t": 0.05, "AUC_xi": 0.05}) == "partial"           # closes < 50% of the rho_t gap
    auc2 = _auc([("W", 0.80, 0.70, 0.95), ("W+T", 0.95, 0.85, 0.93), ("DBS", 0.98, 0.90, 0.91)])
    assert verdict(gap_closed(auc2, "W", "W+T"), {"AUC_rho_t": 0.02, "AUC_xi": 0.02}) == "supported"
    assert verdict(gap_closed(auc2, "W", "W+T"), {"AUC_rho_t": 0.30, "AUC_xi": 0.30}) == "partial"
    auc3 = _auc([("W", 0.80, 0.70, 0.95), ("W+T", 0.80, 0.70, 0.95), ("DBS", 0.98, 0.90, 0.91)])
    assert verdict(gap_closed(auc3, "W", "W+T"), {"AUC_rho_t": 0.0, "AUC_xi": 0.0}) == "not supported"


def test_metric_variant_forces_the_metric_backbone_first_and_stays_connected():
    t = table()
    net = Network(t)
    metric = set(np.where(t.distortion <= 1 + 1e-9)[0])
    for m in ["W+B", "DispFilt+B"]:
        order = ranking(t, m, seed=0)
        assert sorted(order) == list(range(net.m))
        assert set(order[: len(metric)]) == metric                     # all metric edges come first
        assert all(net.n_components(order[:k]) == 1 for k in range(net.n - 1, net.m + 1))


def test_random_control_forces_as_many_edges_as_the_metric_backbone_and_contains_the_tree():
    t = table()
    net = Network(t)
    n_metric = int((t.distortion <= 1 + 1e-9).sum())
    order = ranking(t, "W+X", seed=2)
    assert sorted(order) == list(range(net.m))
    assert net.n_components(order[: net.n - 1]) == 1                   # tree first
    metric = set(np.where(t.distortion <= 1 + 1e-9)[0])
    assert set(order[:n_metric]) != metric or n_metric == net.m        # generally not the metric edges
    assert set(ranking(t, "W+X", seed=2)[:n_metric]) != set(ranking(t, "W+X", seed=3)[:n_metric]) or n_metric == net.m


def test_path_verdict():
    from backbone.connected import path_verdict
    rows = {("W+T", "metric"): 0.80, ("W+B", "metric"): 0.88, ("W+X", "metric"): 0.81, ("DBS", "metric"): 0.90}
    a = pd.DataFrame({"AUC_xi": rows.values()}, index=pd.MultiIndex.from_tuples(rows.keys(), names=["method", "limit"]))
    assert path_verdict(a, "W")["verdict"] == "supported"
    a.loc[("W+X", "metric"), "AUC_xi"] = 0.88
    assert path_verdict(a, "W")["verdict"] == "partial"
    a.loc[("W+B", "metric"), "AUC_xi"] = 0.80
    assert path_verdict(a, "W")["verdict"] == "not supported"


def test_fill_order_puts_metric_first_then_the_pool_and_is_a_permutation():
    from backbone.connected import fill_order
    from backbone.diagnose import louvain_labels
    t = table()
    net = Network(t)
    labels = louvain_labels(net, 1)[0]
    base = ranking(t, "W")
    metric = set(np.where(t.distortion <= 1 + 1e-9)[0])
    n_metric = len(metric)
    between = labels[net.i] != labels[net.j]
    for pool in ["between", "within"]:
        o = fill_order(net, base, labels, pool)
        assert sorted(o) == list(range(net.m)) and set(o[:n_metric]) == metric
        rest = o[n_metric:]
        want = (rest[0:1] if len(rest) else rest)
        first_flags = between[rest] if pool == "between" else ~between[rest]
        k = int(first_flags.sum())
        assert first_flags[:k].all() and not first_flags[k:].any()      # pool edges come before the others
    r = fill_order(net, base, labels, "random", np.random.default_rng(0))
    assert sorted(r) == list(range(net.m)) and set(r[:n_metric]) == metric


def test_bridge_verdict():
    from backbone.connected import bridge_verdict
    idx = pd.MultiIndex.from_tuples([(m, "metric") for m in ["W+B", "W|between", "W|within", "W|random", "DBS"]],
                                    names=["method", "limit"])
    a = pd.DataFrame({"AUC_xi": [0.80, 0.86, 0.80, 0.79, 0.90]}, index=idx)
    assert bridge_verdict(a, "W")["verdict"] == "supported"
    a.loc[("W|within", "metric"), "AUC_xi"] = 0.86
    assert bridge_verdict(a, "W")["verdict"] == "partial"
    a.loc[("W|between", "metric"), "AUC_xi"] = 0.80
    assert bridge_verdict(a, "W")["verdict"] == "not supported"
