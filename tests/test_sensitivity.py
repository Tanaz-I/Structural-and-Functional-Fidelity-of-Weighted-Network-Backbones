import networkx as nx
import numpy as np
import pandas as pd
import pytest

from backbone.backbones import edge_table
from backbone.sensitivity import SETTINGS, dispfilt_variants, ranking_table, run_setting, spread_over_settings
from backbone.sparsify import disparity_alpha_normalised, importance
from backbone.sweep import Network


def small_table(n=14, seed=3):
    G = nx.connected_watts_strogatz_graph(n, 4, 0.4, seed=seed)
    rng = np.random.default_rng(seed)
    for u, v in G.edges:
        p = float(rng.uniform(0.05, 0.9))
        G[u][v].update(proximity=p, distance=1 / p - 1)
    return edge_table(G)


def test_normalised_disparity_hand_computed():
    # path 1-2-3 with proximities .6 and .2: strengths s1=.6, s2=.8, s3=.2; degrees 1, 2, 1
    t = pd.DataFrame({"u": [1, 2], "v": [2, 3], "proximity": [0.6, 0.2]})
    a = disparity_alpha_normalised(t)
    # edge (1,2): node 1 has degree 1 -> alpha 0, so min is 0; edge (2,3): node 3 has degree 1 -> 0
    assert a.tolist() == [0.0, 0.0]
    # triangle with equal weights: p_i = 1/2 at every node, k = 2 -> alpha = (1 - 1/2)^1 = 0.5
    tri = pd.DataFrame({"u": [1, 1, 2], "v": [2, 3, 3], "proximity": [0.3, 0.3, 0.3]})
    assert disparity_alpha_normalised(tri) == pytest.approx([0.5, 0.5, 0.5])


def test_normalised_disparity_prefers_locally_strong_edge():
    # node 1 has a strong and a weak edge; the strong one must be more significant (smaller alpha)
    t = pd.DataFrame({"u": [1, 1, 2, 3], "v": [2, 3, 4, 4], "proximity": [0.8, 0.1, 0.4, 0.4]})
    a = disparity_alpha_normalised(t)
    assert a[0] < a[1]
    assert (importance(t, "DispFiltNorm") == -a).all()


def test_settings_list_has_base_and_variants():
    assert SETTINGS[0] == "base" and "dispfilt_norm" in SETTINGS and "beta_40" in SETTINGS


def test_run_setting_smoke_and_unknown_setting():
    net = Network(small_table())
    res = run_setting(net, "dispfilt_norm", runs=5)
    assert set(res.limit) == {"own", "metric"} and set(res.method) == {"DispFiltNorm"}
    assert res[["AUC_rho_C", "AUC_rho_t", "AUC_xi"]].notna().all().all()
    with pytest.raises(ValueError):
        run_setting(net, "nope", runs=5)


def test_ranking_table_orders_methods():
    df = pd.DataFrame({"setting": "base", "limit": "metric", "method": ["A", "B"],
                       "AUC_rho_C": [0.5, 0.9], "AUC_rho_t": [0.7, 0.6], "AUC_xi": [0.1, 0.2]})
    r = ranking_table(df).set_index("measure").order
    assert r["AUC_rho_C"] == "B > A" and r["AUC_rho_t"] == "A > B"


def _toy_results():
    rows = []
    for setting in ["base", "beta_40", "runs_300"]:
        for m, v in [("DBS", 0.9), ("W", 0.7), ("DispFilt", 0.6)]:
            rows.append((setting, "metric", m, v, v, v + (0.05 if setting == "runs_300" else 0)))
    rows.append(("dispfilt_norm", "metric", "DispFiltNorm", 0.8, 0.8, 0.8))
    return pd.DataFrame(rows, columns=["setting", "limit", "method", "AUC_rho_C", "AUC_rho_t", "AUC_xi"])


def test_dispfilt_variants_and_spread():
    r = _toy_results()
    assert set(dispfilt_variants(r).method) == {"DispFilt", "DispFiltNorm"}
    s = spread_over_settings(r).set_index("method")
    assert s.loc["W", "AUC_xi"] == pytest.approx(0.05) and s.loc["W", "AUC_rho_C"] == 0
