import numpy as np
import pandas as pd
import pytest

from backbone.summary import SCENARIOS, auc, optimal_lambda, x_tilde


def test_scenario_weights_sum_to_one():
    assert len(SCENARIOS) == 17                               # the 17 rows of Table 3
    for name, w in SCENARIOS.items():
        assert sum(w) == pytest.approx(1.0), name


def test_auc_known_curves():
    X = np.linspace(0, 1, 11)
    assert auc(X, np.full(11, 0.7), 0.3) == pytest.approx(0.7)       # constant curve
    assert auc(X, X, 0.5) == pytest.approx(0.75)                      # mean of y=X on [0.5, 1]
    assert auc(X, X, 0.55) == pytest.approx(0.775)                    # limit between grid points is interpolated
    assert np.isnan(auc(X, X, 1.0))


def test_x_tilde_floor():
    assert x_tilde(0.3, 0.1) == pytest.approx(0.3)
    assert x_tilde(0.0, 0.13) == pytest.approx(0.13)             # DBS: never below the metric backbone size
    assert x_tilde(0.1, 0.5) == pytest.approx(0.5)


def test_connectivity_limits_on_toy_network():
    from backbone.sweep import Network, connectivity_limits
    # a path 1-2-3-4 (bridges) plus chord 1-3: DBS/SmD rank bridges high; W ranks by proximity
    t = pd.DataFrame({"u": [1, 2, 3, 1], "v": [2, 3, 4, 3], "proximity": [0.5, 0.5, 0.05, 0.4],
                      "distance": [1.0, 1.0, 19.0, 1.5], "distortion": [1, 1, 1, 1.5], "lam": [np.inf, np.inf, np.inf, 0.8]})
    lim = connectivity_limits(Network(t), methods=["DBS", "W"])
    assert lim["DBS"] == 0.0                                      # tree edges kept first: connected all the way down
    assert 0.0 <= lim["W"] <= 1.0


def test_cross_network_tests_shape():
    from backbone.summary import cross_network_tests
    rows = []
    for net in range(6):
        for m, v in [("DBS", .9), ("SmD", .88), ("wEffRes", .85), ("DispFilt", .6), ("W", .5)]:
            rows.append((f"n{net}", m, v + 0.01 * net, v, v))
    df = pd.DataFrame(rows, columns=["network", "method", "AUC_rho_C", "AUC_rho_t", "AUC_xi"])
    t = cross_network_tests(df)
    assert len(t) == 3 * 4 and (t.networks_DBS_better >= 0).all()
    w = t[(t.measure == "AUC_rho_C") & (t.versus == "W")].iloc[0]
    assert w.p_wilcoxon < 0.05 and w.p_mannwhitney < 0.05


def test_optimal_lambda_follows_weights():
    lam = pd.DataFrame({"lam": [0.1, 0.5, 5.0], "n_edges": [100, 50, 10], "X": [1.0, 0.4, 0.0],
                        "rho_C": [1.0, 0.8, 0.1], "rho_t": [1.0, 0.9, 0.2], "xi": [1.0, 0.7, 0.1]})
    opt = optimal_lambda(lam).set_index("scenario")
    assert opt.loc["Mainly rho_C", "lam_opt"] == 0.1                  # fidelity dominates -> full graph
    assert opt.loc["Mainly X", "lam_opt"] == 5.0                      # sparsity dominates -> sparsest
    assert opt.loc["Balanced", "lam_opt"] in (0.1, 0.5, 5.0)
