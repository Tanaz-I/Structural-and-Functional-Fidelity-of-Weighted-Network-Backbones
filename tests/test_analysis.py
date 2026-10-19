"""Analysis tests: tables, bootstrap, statistics, experiments (known-truth data), verdicts and the advisor."""
import numpy as np
import pandas as pd
import pytest

from backbone import advisor, analysis, report


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------
def _methods(perfect: bool, n_net=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for net in range(n_net):
        for q in (0.1, 0.3):
            struct = rng.permutation(6) / 6 + 0.01
            for i, meth in enumerate("ABCDEF"):
                f = struct[i] if perfect else rng.random()
                rows.append({"network": f"n{net}", "method": meth, "q": q, "struct": struct[i],
                             "spread": f, "comm": f, "path": f, "rank": f})
    return pd.DataFrame(rows)


def test_agreement_perfect_vs_random():
    perfect = analysis.agreement_summary(_methods(True), n_boot=200)
    assert (perfect.mean_tau > 0.99).all() and not perfect.below_threshold.any()
    rand = analysis.agreement_summary(_methods(False), n_boot=200)
    assert (rand.mean_tau.abs() < 0.5).all() and rand.below_threshold.all()
    assert (rand.n_networks == 6).all()


def test_cluster_bootstrap_ci_contains_point():
    groups = {i: np.random.default_rng(i).normal(1.0, 0.3, 20) for i in range(8)}
    p, lo, hi, k = analysis.cluster_bootstrap(groups, analysis.mean_of_concat, 300)
    assert lo <= p <= hi and k == 8


def test_gap_and_failure_flags():
    m = pd.DataFrame({"network": ["a"], "method": ["W"], "q": [0.2], "struct": [0.9], "spread": [0.3],
                      "comm": [0.9], "path": [0.9], "rank": [0.9], "bridge_ratio": [0.5],
                      "hub_concentration": [0.4], "connected": [True]})
    g = analysis.gap_table(m).set_index("function")
    assert g.failure["spread"] and not g.failure["comm"]
    assert g.gap["spread"] == pytest.approx(0.6)


def test_skill_against_null_and_near_ceiling():
    def row(kind, name, f, draw):
        return {"network": "a", "kind": kind, "name": name, "draw": draw, "q": 0.5, "struct": f, "spread": f,
                "comm": f, "path": f, "rank": f}
    w = pd.DataFrame([row("method", "W", 0.8, -1), row("null", "R-U", 0.4, 0), row("null", "R-U", 0.6, 1),
                      row("method", "DF", 0.99, -1)])
    s = analysis.skill_table(w)
    sk = s[(s.function == "spread")].set_index("method")
    assert sk.skill["W"] == pytest.approx((0.8 - 0.5) / 0.5)
    w2 = pd.DataFrame([row("method", "W", 0.8, -1), row("null", "R-U", 0.98, 0)])
    s2 = analysis.skill_table(w2)
    assert s2[s2.function == "spread"].near_ceiling.all()


# ----------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------
METHODS = ["W", "DF", "NC", "SMDS", "HSS", "wEffRes"]


def test_holm_matches_known_values():
    assert np.allclose(analysis.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert (analysis.holm([0.5, 0.5]) <= 1).all()


def _gap(effect: float, seed=0, n_net=12):
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(n_net):
        u = rng.normal(0, 0.05)
        for m in METHODS:
            for q in (0.1, 0.3, 0.5):
                br = rng.uniform(0.2, 1.5)
                for f in ("spread", "comm"):
                    rows.append({"network": f"n{n}", "method": m, "q": q, "function": f, "bridge_ratio": br,
                                 "hub_concentration": rng.uniform(0.2, 0.6),
                                 "gap": 0.4 + effect * br + u + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_bridge_regression_detects_true_effect_and_rejects_null():
    real = analysis.bridge_regression(_gap(-0.3))
    assert real.supported.all() and (real.coef_bridge_ratio < 0).all()
    null = analysis.bridge_regression(_gap(0.0, seed=1))
    assert not null.supported.any()


def test_weight_heterogeneity_interaction_recovered():
    rng = np.random.default_rng(0)
    meta, rows = [], []
    for n in range(40):
        s, c, mx = rng.choice([0.5, 1.0, 1.5]), rng.choice([0, 0.5, 1.0]), rng.choice([0.1, 0.3, 0.5])
        meta.append({"network": f"n{n}", "sigma": s, "coupling": c, "mixing": mx})
        u = rng.normal(0, 0.03)
        for m in ("W", "DF", "SMDS"):
            for q in (0.1, 0.3):
                wb = m in ("W", "DF")
                rows.append({"network": f"n{n}", "method": m, "q": q, "function": "spread",
                             "gap": 0.2 + (0.25 * s if wb else 0.0) + u + rng.normal(0, 0.03)})
    out = analysis.weight_heterogeneity_model(pd.DataFrame(rows), pd.DataFrame(meta))
    r = out.set_index("term")
    assert r.positive_excludes_zero["wb:sigma"] and not r.positive_excludes_zero["wb:mixing"]


def _methods_table(different_winners: bool, n_net=8):
    rows = []
    rng = np.random.default_rng(0)
    for n in range(n_net):
        for q in (0.1, 0.3):
            for i, m in enumerate(METHODS):
                base = rng.random()
                row = {"network": f"n{n}", "method": m, "q": q}
                for j, f in enumerate(("spread", "comm", "path", "rank")):
                    row[f] = (1.0 if i == (j if different_winners else 0) else 0.1) + 0.001 * base
                rows.append(row)
    return pd.DataFrame(rows)


def test_function_dependence_h5():
    yes = analysis.function_dependence(analysis.winners(_methods_table(True)), n_boot=200)
    no = analysis.function_dependence(analysis.winners(_methods_table(False)), n_boot=200)
    assert yes["supported"] and yes["share"] == 1.0 and no["share"] == 0.0 and not no["supported"]
    assert yes["n_networks"] == 8


# ----------------------------------------------------------------------
# conclusions
# ----------------------------------------------------------------------
def _agree(hi_vals, lo_vals, n=10):
    return pd.DataFrame({"function": ["spread", "comm", "path", "rank"], "mean_tau": [0.0] * 4,
                         "ci_lo": lo_vals, "ci_hi": hi_vals, "n_networks": [n] * 4})


def test_rq1_supported_rejected_inconclusive_and_min_networks():
    assert report.rq1(_agree([0.3, 0.4, 0.9, 0.9], [-0.1] * 4))["verdict"] == "supported"
    assert report.rq1(_agree([1, 1, 1, 1], [0.8] * 4))["verdict"] == "rejected"
    assert report.rq1(_agree([0.3, 0.9, 0.9, 0.9], [-0.1, 0.5, 0.5, 0.5]))["verdict"] == "inconclusive"
    small = report.rq1(_agree([0.3, 0.4, 0.9, 0.9], [-0.1] * 4, n=3))
    assert small["verdict"] == "inconclusive" and small["caveats"]


def test_rq4_and_rq3_not_run():
    assert report.rq4({"share": 0.9, "ci_lo": 0.8, "ci_hi": 0.95, "n_networks": 12})["verdict"] == "supported"
    assert report.rq4({"share": 0.2, "ci_lo": 0.1, "ci_hi": 0.4, "n_networks": 12})["verdict"] == "rejected"
    assert report.rq4({"share": 0.5, "ci_lo": 0.3, "ci_hi": 0.7, "n_networks": 12})["verdict"] == "inconclusive"
    assert report.rq3_synthetic(None)["verdict"] == "not_run"


def _spread_table(path_better: bool, n_net=8):
    """struct is 0.9 everywhere; W and DF spread 0.3, SMDS and HSS spread 0.8 (or the reverse)."""
    rng = np.random.default_rng(0)
    rows = []
    for n in range(n_net):
        for q in (0.1, 0.2, 0.3, 0.5):
            for m in METHODS:
                good = (m in ("SMDS", "HSS")) == path_better
                rows.append({"network": f"n{n}", "method": m, "q": q, "struct": 0.9, "comm": 0.8,
                             "spread": (0.8 if good else 0.3) + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def _skill_from(m: pd.DataFrame) -> pd.DataFrame:
    return m.assign(null="R-W", function="spread", skill=m.spread)[["network", "q", "method", "null", "function", "skill"]]


def test_h1_pathbased_vs_weight_filters():
    yes = _spread_table(True)
    out = analysis.pathbased_vs_weight_filters(yes, _skill_from(yes))
    assert len(out) == 8 and out.supported.all() and (out.median_diff > 0).all()
    assert report.h1(out)["verdict"] == "supported"
    no = _spread_table(False)
    out = analysis.pathbased_vs_weight_filters(no, _skill_from(no))
    assert not out.supported.any() and report.h1(out)["verdict"] == "rejected"


def test_h2_metric_backbone_communities_uses_first_budget_holding_the_metric_backbone():
    m = _spread_table(True).assign(comm=lambda d: np.where(d.q >= 0.2, 0.9, 0.1))
    res = analysis.metric_backbone_communities(m, {f"n{i}": 0.15 for i in range(8)}, 0.6, n_boot=200)
    assert res["n_networks"] == 8 and abs(res["mean"] - 0.9) < 1e-9
    assert report.h2(res)["verdict"] == "supported"
    low = analysis.metric_backbone_communities(m.assign(comm=0.2), {f"n{i}": 0.15 for i in range(8)}, 0.6, n_boot=200)
    assert report.h2(low)["verdict"] == "rejected"
    assert report.h2(analysis.metric_backbone_communities(m.iloc[:0], {}, 0.6))["verdict"] == "inconclusive"


# ----------------------------------------------------------------------
# advisor
# ----------------------------------------------------------------------
ADV_METHODS = ["A", "B", "C"]


def _inputs(scores, noise=0.01, n_units=40, seed=0):
    rng = np.random.default_rng(seed)
    rows, samples = [], {}
    for m, s in zip(ADV_METHODS, scores):
        rows.append({"method": m, "struct": 0.95, "spread": s, "comm": s, "path": s, "rank": s, "connected": True,
                     "bridge_ratio": 1.0, "hub_concentration": 0.3})
        for f in ("spread", "comm", "path"):
            samples[f"{m}|0.20|{f}"] = np.clip(rng.normal(s, noise, n_units), 0, 1)
    nulls = pd.DataFrame({"null": ["R-U", "R-W"], "struct": 0.5, "spread": 0.3, "comm": 0.3, "path": 0.3, "rank": 0.3}).set_index("null")
    return pd.DataFrame(rows), nulls, samples


W = {"spread": 1, "comm": 1, "path": 1, "rank": 1}


def test_clear_recommendation_when_one_method_dominates():
    m, n, s = _inputs([0.9, 0.6, 0.5])
    r = advisor.recommend(m, n, s, 0.2, W)
    assert r["recommendation"] == "A" and r["label"] == "clear" and r["p_best"]["A"] > 0.95
    assert r["pareto"] == ["A"] and r["tradeoffs"]["spread"]["give_up"] == pytest.approx(0)


def test_close_call_when_methods_tie():
    m, n, s = _inputs([0.80, 0.80, 0.5], noise=0.1)
    for f in ("spread", "comm", "path"):
        s[f"B|0.20|{f}"] = s[f"A|0.20|{f}"] + 0.002 * np.random.default_rng(1).standard_normal(40)  # genuinely close
    r = advisor.recommend(m, n, s, 0.2, W)
    assert r["label"] == "close call" and abs(r["p_best"]["A"] - r["p_best"]["B"]) < 0.5


def test_connected_constraint_excludes_and_weights_matter():
    m, n, s = _inputs([0.9, 0.7, 0.5])
    m.loc[m.method == "A", "connected"] = False
    r = advisor.recommend(m, n, s, 0.2, W, require_connected=True)
    assert r["recommendation"] == "B" and r["excluded"] == ["A"] and any("Excluded" in x for x in r["warnings"])
    m2, n2, s2 = _inputs([0.7, 0.7, 0.7])
    m2.loc[m2.method == "C", ["spread"]] = 0.95
    only_spread = advisor.recommend(m2, n2, s2, 0.2, {"spread": 1.0})
    assert only_spread["recommendation"] == "C"


def test_failure_warning_when_structure_looks_fine():
    m, n, s = _inputs([0.5, 0.4, 0.3])
    r = advisor.recommend(m, n, s, 0.2, W)
    assert any("structurally fine" in x for x in r["warnings"])


def test_preset_end_to_end():
    from pathlib import Path
    name = "hospital"
    if not (Path(__file__).resolve().parents[1] / "results" / "samples" / f"{name}.npz").exists():
        pytest.skip("no cached results for the preset network")
    r = advisor.recommend_for_network(name, 0.3, {"spread": 1.0})
    assert r["recommendation"] in {"W", "DF", "NC", "SMDS", "HSS", "wEffRes"}
    assert abs(sum(r["p_best"].values()) - 1) < 1e-9 and r["sensitivity"] is not None
    assert "Recommendation" in advisor.to_markdown(r, name)

