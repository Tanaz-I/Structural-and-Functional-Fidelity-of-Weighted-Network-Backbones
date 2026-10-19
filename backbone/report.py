"""Verdicts for the paper hypotheses H1-H3, the research-question analyses and the auto-written findings and overall conclusion text."""
from __future__ import annotations


import pandas as pd
from .analysis import FUNCS

# ---------------------------------------------------------------- verdicts
FAMILY_LEVEL = 1 - 0.05 / 6  # Holm family: H1-H3 and the three research-question analyses
MIN_NETWORKS = 5


def _v(hid, statement, rule, verdict, observed, n_networks, caveats=(), source=()):
    caveats = list(caveats)
    if verdict in ("supported", "rejected") and n_networks < MIN_NETWORKS:
        caveats.append(f"only {n_networks} independent networks (< {MIN_NETWORKS}): downgraded to inconclusive")
        verdict = "inconclusive"
    return {"id": hid, "statement": statement, "rule": rule, "verdict": verdict, "observed": observed,
            "n_networks": int(n_networks), "caveats": caveats, "source_tables": list(source)}


# Hypotheses H1-H3 follow the project paper; the RQ entries answer its research questions (no stated direction).
def h1(t: pd.DataFrame) -> dict:
    """Paper H1: path-based methods preserve spreading better than W and DF on contact networks."""
    ok = t[t.p_holm.notna()]
    n_sup = int(ok.supported.sum())
    verdict = "supported" if len(ok) and n_sup == len(ok) else ("rejected" if len(ok) and n_sup == 0 else "inconclusive")
    obs = {"tests_supported": f"{n_sup}/{len(ok)}",
           "median_effects": {f"{r.challenger}-{r.baseline}-{r.function}-{r.measure}": round(r.median_diff, 4) for r in ok.itertuples()}}
    cav = ["Wilcoxon over (network, budget) cells: cells of one network are not independent, so p-values are optimistic"]
    return _v("H1", "At comparable budgets, weight thresholding and the disparity filter preserve spreading less well than "
                    "shortest-path-based methods (SMDS, HSS) on contact networks.",
              "all paired one-sided Wilcoxon tests (gap and skill; SMDS, HSS vs W, DF) significant after Holm correction",
              verdict, obs, int(ok.n_networks.min()) if len(ok) else 0, cav, ["e4_h1"])


def h2(res: dict) -> dict:
    """Paper H2: the metric backbone preserves community structure on contact networks."""
    lo, hi, thr = res["ci_lo"], res["ci_hi"], res["threshold"]
    verdict = "inconclusive" if not res["n_networks"] else ("supported" if lo >= thr else ("rejected" if hi < thr else "inconclusive"))
    obs = {"mean_F_comm": round(res["mean"], 3), "ci": [round(lo, 3), round(hi, 3)]}
    return _v("H2", "The metric backbone preserves community structure on contact networks.",
              f"mean F_comm of SMDS at the metric-backbone size, {FAMILY_LEVEL:.0%} CI lower limit >= {thr}",
              verdict, obs, res["n_networks"], source=["e4_h2"])


def h3(br: pd.DataFrame) -> dict:
    """Paper H3 (tentative): lost bridging edges explain failures; associational on real networks."""
    obs = {r.function: {"coef": round(r.coef_bridge_ratio, 4), "ci": [round(r.ci_lo, 4), round(r.ci_hi, 4)],
                        "p_lr_holm": round(r.p_lr_holm, 4), "supported": bool(r.supported)} for r in br.itertuples()}
    n_sup = int(br.supported.sum())
    verdict = "supported" if n_sup == len(br) else ("rejected" if n_sup == 0 else "inconclusive")
    cav = ["associational: regression on real networks does not establish causation"]
    if verdict == "inconclusive":
        cav.append("supported for some functions only")
    return _v("H3", "Where spreading preservation fails, bridging edges between communities have been lost disproportionately.",
              "negative bridge_ratio coefficient, Holm-adjusted likelihood-ratio test p < 0.05, for spreading and communities",
              verdict, obs, int(br.n_networks.min()), cav, ["e3_bridge"])


def rq1(agree: pd.DataFrame, threshold: float = 0.6) -> dict:
    """RQ1: does structural fidelity predict functional fidelity? agree: agreement_summary at FAMILY_LEVEL."""
    n_low = int((agree.ci_hi < threshold).sum())
    n_high = int((agree.ci_lo >= threshold).sum())
    verdict = "supported" if n_low >= 2 else ("rejected" if n_high >= 3 else "inconclusive")
    obs = {r.function: {"mean_tau": round(r.mean_tau, 3), "ci": [round(r.ci_lo, 3), round(r.ci_hi, 3)]}
           for r in agree.itertuples()}
    return _v("RQ1", "Structural fidelity agrees only weakly with functional fidelity when ranking methods.",
              f"mean Kendall tau-b and the {FAMILY_LEVEL:.0%} CI upper limit below {threshold} for >= 2 of 4 functions",
              verdict, obs, int(agree.n_networks.min()), source=["e1_summary"])


def rq3_synthetic(het: pd.DataFrame | None) -> dict:
    """RQ3, controlled test: weight heterogeneity and coupling versus weight-based filters (synthetic factorial)."""
    stmt = "Weight heterogeneity and weight-degree coupling hurt weight-based filters more than SMDS."
    rule = "wb:sigma and wb:coupling interaction CIs above zero (spreading)"
    if het is None:
        return _v("RQ3", stmt, rule, "not_run", {}, 0, ["synthetic factorial sweep not run yet"])
    s = het[het.function == "spread"].set_index("term")
    pos = [t for t in ("wb:sigma", "wb:coupling") if s.positive_excludes_zero[t]]
    verdict = "supported" if len(pos) == 2 else ("rejected" if not pos else "inconclusive")
    obs = {t: {"coef": round(s.coef[t], 4), "ci": [round(s.ci_lo[t], 4), round(s.ci_hi[t], 4)]} for t in s.index}
    return _v("RQ3", stmt, rule, verdict, obs, int(s.n_networks.iloc[0]),
              ["causal reading is allowed only for the synthetic design"], ["e3_heterogeneity"])


def rq4(res: dict) -> dict:
    """RQ4: the best method depends on the function, so a recommendation must name the function."""
    verdict = "supported" if res["ci_lo"] > 0.5 else ("rejected" if res["ci_hi"] < 0.5 else "inconclusive")
    obs = {"share": round(res["share"], 3), "ci": [round(res["ci_lo"], 3), round(res["ci_hi"], 3)]}
    return _v("RQ4", "No single method wins on all four functions.",
              f"share of cells with different best methods; {FAMILY_LEVEL:.0%} CI lower limit above 0.5",
              verdict, obs, res["n_networks"], source=["e4_h5"])


# ---------------------------------------------------------------- findings
NAMES = {"spread": "spreading", "comm": "community structure", "path": "shortest paths", "rank": "node rankings"}
REGIMES = {"sparse (5-15% of edges)": (0.0, 0.15), "moderate (20-40%)": (0.15, 0.40), "dense (50-70%)": (0.40, 1.0)}


def agreement_findings(agree: pd.DataFrame) -> list[dict]:
    out = []
    for r in agree.sort_values("mean_tau").itertuples():
        level = "weakly" if r.ci_hi < 0.6 else ("moderately" if r.mean_tau < 0.8 else "strongly")
        out.append({"kind": "agreement", "strength": "confirmatory-pending", "figure": "atlas",
                    "text": f"Structural fidelity {level} predicts {NAMES[r.function]} fidelity when ranking methods "
                            f"(mean Kendall tau {r.mean_tau:.2f}, CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}], "
                            f"{r.n_networks} networks).",
                    "n_networks": int(r.n_networks)})
    return out


def failure_findings(gap: pd.DataFrame, top: int = 4) -> list[dict]:
    g = gap.groupby(["method", "function"]).agg(rate=("failure", "mean"), n=("network", "nunique")).reset_index()
    out = []
    for r in g.sort_values("rate", ascending=False).head(top).itertuples():
        out.append({"kind": "failure", "strength": "exploratory", "figure": "atlas",
                    "text": f"{r.method} fails on {NAMES[r.function]} in {r.rate:.0%} of cases where the structural check "
                            f"looked fine (pooled over budgets, {r.n} networks).", "n_networks": int(r.n)})
    best = g.sort_values("rate").iloc[0]
    out.append({"kind": "failure", "strength": "exploratory", "figure": "atlas",
                "text": f"The lowest failure rate is {best.method} on {NAMES[best.function]} ({best.rate:.0%}).",
                "n_networks": int(best.n)})
    rev = gap.reverse_failure.mean()
    out.append({"kind": "failure", "strength": "exploratory", "figure": "atlas",
                "text": f"The reverse case (structure poor but function preserved) occurs in {rev:.1%} of cases; "
                        f"the misleading-pass case occurs in {gap.failure.mean():.1%}.", "n_networks": int(gap.network.nunique())})
    return out


def chance_findings(skill: pd.DataFrame, null: str = "R-W") -> list[dict]:
    s = skill[(skill.null == null) & (skill.function != "struct")]
    g = s.groupby(["method", "function"]).apply(lambda d: (d.skill <= 0).mean()).rename("share").reset_index()
    n = s.network.nunique()
    out = []
    for r in g[g.share > 0.2].sort_values("share", ascending=False).head(5).itertuples():
        out.append({"kind": "beyond_chance", "strength": "exploratory", "figure": "curves",
                    "text": f"{r.method} is no better than weight-proportional random removal on {NAMES[r.function]} in "
                            f"{r.share:.0%} of (network, budget) cells.", "n_networks": int(n)})
    if not out:
        out.append({"kind": "beyond_chance", "strength": "exploratory", "figure": "curves",
                    "text": "Every method beats weight-proportional random removal in at least 80% of cells for every function.",
                    "n_networks": int(n)})
    return out


def regime_table(skill: pd.DataFrame, null: str = "R-W") -> pd.DataFrame:
    """Best method (highest mean skill versus the null) per function and budget regime."""
    s = skill[(skill.null == null) & (skill.function != "struct")]
    rows = []
    for label, (lo, hi) in REGIMES.items():
        d = s[(s.q > lo) & (s.q <= hi)]
        for f in FUNCS:
            m = d[d.function == f].groupby("method").skill.mean().sort_values(ascending=False)
            rows.append({"regime": label, "function": f, "best": m.index[0], "mean_skill": float(m.iloc[0]),
                         "runner_up": m.index[1], "margin": float(m.iloc[0] - m.iloc[1])})
    return pd.DataFrame(rows)


def regime_findings(reg: pd.DataFrame, n_networks: int) -> list[dict]:
    out = []
    for f in FUNCS:
        d = reg[reg.function == f]
        bests = ", ".join(f"{r.best} in the {r.regime} regime" for r in d.itertuples())
        out.append({"kind": "method_choice", "strength": "exploratory", "figure": "leaderboard",
                    "text": f"For {NAMES[f]} the best average method is {bests}.", "n_networks": n_networks})
    return out


def robustness_findings(rob: pd.DataFrame, n_networks: int) -> list[dict]:
    """rob: verdict_by_variant table from E5 (columns RQ1_changed, H1_changed, RQ4_changed, best_*_changed)."""
    out = []
    v = rob[rob.variant != "default"]
    for h, label in (("RQ1", "structure-versus-function agreement"), ("H1", "the path-based advantage over W and DF"), ("RQ4", "function dependence of the best method")):
        flipped = v.loc[v[f"{h}_changed"], "variant"].tolist()
        text = (f"The verdict on {label} ({h}) is unchanged under all {len(v)} modelling variants." if not flipped else
                f"The verdict on {label} ({h}) changes under: {', '.join(flipped)}.")
        out.append({"kind": "robustness", "strength": "exploratory", "figure": "conclusions", "text": text, "n_networks": n_networks})
    for f, name in (("spread", "spreading"), ("comm", "community"), ("path", "path"), ("rank", "ranking")):
        flipped = v.loc[v[f"best_{f}_changed"], "variant"].tolist()
        if flipped:
            out.append({"kind": "robustness", "strength": "exploratory", "figure": "conclusions",
                        "text": f"The best method on {name} fidelity depends on the modelling choice: it changes under {', '.join(flipped)}.",
                        "n_networks": n_networks})
    return out


def overall_summary(verdicts: list[dict], reg: pd.DataFrame, n_real: int, n_syn: int) -> dict:
    by = {v["id"]: v for v in verdicts}

    def say(h, yes, no, maybe):
        v = by[h]["verdict"]
        return {"supported": yes, "rejected": no, "inconclusive": maybe, "not_run": f"{h} has not been run yet."}[v]

    answer = say("RQ1", "Structural fidelity is a poor guide to functional fidelity: it ranks methods very differently from how "
                       "spreading, communities, paths and node rankings do.",
                 "Structural fidelity predicts functional fidelity well in these data, so cheap structural checks are adequate.",
                 "The evidence on whether structural fidelity predicts functional fidelity is mixed.")
    why = [say("H3", "Failures are associated with losing inter-community (bridging) edges; this is an association on real networks.",
               "No association between bridge retention and failure was found, so the bridge explanation is not supported here.",
               "The bridge explanation is supported for some functions only."),
           say("RQ3", "On synthetic networks, heterogeneous or degree-coupled weights hurt weight-based filters more than SMDS (causal).",
               "On synthetic networks, weight heterogeneity and coupling did not hurt weight-based filters more than SMDS.",
               "On synthetic networks, only one of the two weight factors hurt weight-based filters more than SMDS.")]
    methods = [say("H1", "Shortest-path-based methods (SMDS, HSS) preserve spreading better than the global threshold and disparity filter.",
                   "Shortest-path-based methods do not preserve spreading better than the global threshold and disparity filter.",
                   "Path-based methods beat the weight-based filters on some, not all, measures."),
               say("H2", "The metric backbone preserves community structure on contact networks.",
                   "The metric backbone does not preserve community structure on these contact networks.",
                   "Whether the metric backbone preserves community structure is unclear."),
               say("RQ4", "No single method is best on all four functions.",
                   "One method is best on all functions in most cases, so a single default is defensible.",
                   "Whether one method can win on every function is unclear.")]
    limits = [f"{n_real} real networks and {n_syn} synthetic networks; claims hold only for these domains and sizes.",
              "SI spreading with one weight-to-probability map; static, undirected networks.",
              "Verdicts are confirmatory only if the protocol was frozen before the runs; otherwise they are exploratory."]
    guide = [f"{r.function}/{r.regime}: prefer {r.best} (margin over {r.runner_up}: {r.margin:.2f} skill)" for r in reg.itertuples()]
    return {"answer": answer, "why": why, "methods": methods, "limits": limits, "guidance": guide}
