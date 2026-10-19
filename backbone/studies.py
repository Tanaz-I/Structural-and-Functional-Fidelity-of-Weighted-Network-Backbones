"""The experiments: E1/E2 (agreement, failure atlas), E3/E4 (mechanisms, method contrasts), E5 (robustness)
and the conclusion builder. Each function reads cached results and writes tables under results/analysis/.
Run them through `python run.py <command>`."""
from __future__ import annotations

import datetime
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from . import analysis, report
from .filters import metric_backbone_mask
from .network import dataset_info, load_network, syn_names
from .pipeline import EvalConfig, evaluate_network

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def protocol() -> dict:
    return yaml.safe_load((ROOT / "protocol.yaml").read_text())


def dev_networks() -> list[str]:
    """The real networks used in every analysis (protocol.yaml); the unit tests use a built-in test graph."""
    return list(protocol()["real_networks"])


def metric_shares(names) -> dict:
    """Metric-backbone size as a share of edges, per network (the structural floor of SMDS)."""
    cache = RESULTS / "metric_share.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    for n in names:
        if n not in known:
            net = load_network(n)
            known[n] = float(metric_backbone_mask(net).sum() / net.m)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(known))
    return {n: known[n] for n in names}


def _contact(m: pd.DataFrame) -> pd.DataFrame:
    """Rows of the method table that belong to contact networks (the scope of H1 and H2)."""
    return m[m.network.map(domain_of) == "contact"]


def domain_of(name: str) -> str:
    if re.fullmatch(r"(syn|pilot)\d+", name):
        return "synthetic"
    return dataset_info(name).get("domain") or "unknown"


# ------------------------------------------------------------------ E1 / E2
def run_e1_e2(names, label: str, tag: str = "", boot: int = 1000) -> None:
    """E1: Kendall agreement of structure and function rankings (RQ1). E2: failure atlas."""
    P = protocol()
    out = RESULTS / "analysis" / label
    out.mkdir(parents=True, exist_ok=True)
    w = analysis.wide(analysis.load_results(names, tag=tag))
    m = analysis.method_table(w)
    m["domain"] = m.network.map(domain_of)
    tau_s, tau_f = P["thresholds"]["tau_s"], P["thresholds"]["tau_f"]
    analysis.kendall_per_cell(m).to_csv(out / "e1_tau_cells.csv", index=False)
    summ = analysis.agreement_summary(m, n_boot=boot, seed=P["global_seed"])
    summ.to_csv(out / "e1_summary.csv", index=False)
    gap = analysis.gap_table(m, tau_s, tau_f).merge(m[["network", "method", "q", "domain"]], on=["network", "method", "q"])
    gap.to_csv(out / "e2_atlas.csv", index=False)
    (gap.groupby(["method", "q", "function"]).agg(failure_rate=("failure", "mean"), mean_gap=("gap", "mean"),
                                                  n_networks=("network", "nunique")).reset_index()
     .to_csv(out / "e2_failure_rates.csv", index=False))
    analysis.skill_table(w[w.network.isin(names)]).to_csv(out / "skill.csv", index=False)
    print(f"[{label}] {m.network.nunique()} networks, tau_s={tau_s}, tau_f={tau_f}\n")
    print("E1  mean Kendall tau (structure vs function ranking of the 6 methods)")
    print(summ.round(3).to_string(index=False))
    n_low = int(summ.below_threshold.sum())
    print(f"\nRQ1 rule: CI upper limit < 0.6 for >= 2 functions -> {'SUPPORTED' if n_low >= 2 else 'not supported'} ({n_low}/4)")
    print("\nE2  failure rate by method and function (all budgets pooled)")
    print(gap.groupby(["method", "function"]).failure.mean().unstack().round(3).to_string())


# ------------------------------------------------------------------ E3 / E4
def run_e3_e4(names, label: str = "real_dev", tag: str = "") -> None:
    """E3: bridge regression (H3). E4: winners (RQ4), leaderboard, path-based versus W/DF (H1), metric-backbone communities (H2)."""
    P = protocol()
    out = RESULTS / "analysis" / label
    out.mkdir(parents=True, exist_ok=True)
    w = analysis.wide(analysis.load_results(names, tag=tag))
    m = analysis.method_table(w)
    gap = analysis.gap_table(m, P["thresholds"]["tau_s"], P["thresholds"]["tau_f"])
    skill = analysis.skill_table(w)
    share = metric_shares(names)

    br = analysis.bridge_regression(gap)
    br.to_csv(out / "e3_bridge.csv", index=False)
    print("E3 / H3  gap ~ bridge_ratio + hub_concentration + budget + method + (1|network)   [associational]")
    print(br.round(4).to_string(index=False))
    win = analysis.winners(m)
    win.to_csv(out / "e4_winners.csv", index=False)
    h5 = analysis.function_dependence(win, seed=P["global_seed"])
    pd.DataFrame([h5]).to_csv(out / "e4_h5.csv", index=False)
    print(f"\nE4 / RQ4  share of cells with different best methods: {h5['share']:.3f} [{h5['ci_lo']:.3f}, {h5['ci_hi']:.3f}] "
          f"over {h5['n_networks']} networks -> {'supported' if h5['supported'] else 'not supported'}")
    lb = analysis.leaderboard(skill)
    lb.to_csv(out / "e4_leaderboard.csv", index=False)
    print("\nE4  mean skill versus R-W (weight-aware null)")
    print(lb.pivot(index="method", columns="function", values="mean_skill").round(3).to_string())
    contact = _contact(m)
    h1 = analysis.pathbased_vs_weight_filters(contact, skill[skill.network.isin(contact.network.unique())])
    h1.to_csv(out / "e4_h1.csv", index=False)
    print("\nE4 / H1  SMDS, HSS versus W and DF on spreading, contact networks (paired one-sided Wilcoxon, Holm)")
    print(h1.round(4).to_string(index=False))
    h2 = analysis.metric_backbone_communities(contact, share, P["thresholds"]["tau_f"], seed=P["global_seed"])
    pd.DataFrame([h2]).to_csv(out / "e4_h2.csv", index=False)
    print(f"\nE4 / H2  F_comm of SMDS at the metric-backbone size: {h2['mean']:.3f} [{h2['ci_lo']:.3f}, {h2['ci_hi']:.3f}] "
          f"over {h2['n_networks']} contact networks")


def synthetic_meta(names) -> pd.DataFrame:
    return pd.DataFrame([{"network": n, **{k: load_network(n).meta[k] for k in ("mixing", "sigma", "coupling")}} for n in names])


def run_synthetic_e3(tag: str = "") -> None:
    """E3 / RQ3 on the synthetic factorial design (causal)."""
    P = protocol()
    out = RESULTS / "analysis" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    names = syn_names()
    m = analysis.method_table(analysis.wide(analysis.load_results(names, tag=tag)))
    gap = analysis.gap_table(m, P["thresholds"]["tau_s"], P["thresholds"]["tau_f"])
    meta = synthetic_meta(m.network.unique())
    res = pd.concat([analysis.weight_heterogeneity_model(gap, meta, f) for f in ("spread", "comm", "path")])
    res.to_csv(out / "e3_heterogeneity.csv", index=False)
    print("E3 / RQ3  gap ~ weight_based x (sigma, coupling, mixing) + budget + (1|network)   [causal: synthetic]")
    print(res.round(4).to_string(index=False))


# ------------------------------------------------------------------ E5
E5_BUDGETS = (0.10, 0.20, 0.30, 0.50)
E5_BASE = dict(budgets=E5_BUDGETS, n_runs=50, seed_frac=0.03, n_null_draws=0, louvain_seeds=10)
E5_VARIANTS = {
    "default": {}, "sir": {"with_sir": True}, "proximity_max": {"proximity": "max"},
    "half_time_10": {"target_half_time": 10.0}, "half_time_40": {"target_half_time": 40.0},
    "epsilon_0.05": {"eps": 0.05}, "epsilon_0.2": {"eps": 0.2},
    "louvain_res_0.5": {"louvain_resolution": 0.5}, "louvain_res_2": {"louvain_resolution": 2.0},
    "seeds_uniform": {"seed_mode": "uniform"},
}

def _e5_cell(variant: str, name: str, force: bool = False) -> Path:
    path = RESULTS / "e5" / variant / f"{name}.parquet"
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    df = evaluate_network(load_network(name), EvalConfig(**{**E5_BASE, **E5_VARIANTS[variant]}), with_nulls=False)
    df.attrs = {}
    df.to_parquet(path)
    return path


def _e5_summary(variant: str, names) -> dict:
    P = protocol()
    df = pd.concat([pd.read_parquet(RESULTS / "e5" / variant / f"{n}.parquet") for n in names], ignore_index=True)
    if variant == "sir":  # swap the SI spreading score for the SIR-based one
        sir = df[df.function == "sir"].assign(function="spread")
        df = pd.concat([df[~df.function.isin(["spread", "sir"])], sir], ignore_index=True)
    m = analysis.method_table(analysis.wide(df))
    summ = analysis.agreement_summary(m, n_boot=300, seed=P["global_seed"])
    row = {"variant": variant, "n_networks": int(m.network.nunique())}
    for r in summ.itertuples():
        row[f"tau_{r.function}"] = round(r.mean_tau, 3)
    row["RQ1_functions_below_0.6"] = int((summ.mean_tau < 0.6).sum())
    piv = _contact(m).pivot_table(index=["network", "q"], columns="method", values="spread")
    for chal in ("SMDS", "HSS"):
        for base in ("W", "DF"):
            row[f"{chal}_minus_{base}_spread"] = round(float((piv[chal] - piv[base]).median()), 4) if len(piv) else float("nan")
    row["H1_direction_holds"] = bool(len(piv)) and all(row[f"{c}_minus_{b}_spread"] > 0 for c in ("SMDS", "HSS") for b in ("W", "DF"))
    rq4 = analysis.function_dependence(analysis.winners(m), n_boot=300, seed=P["global_seed"])
    row["RQ4_share_different_winners"], row["RQ4_holds"] = round(rq4["share"], 3), rq4["share"] > 0.5
    mean_f = m.groupby("method")[analysis.FUNCS].mean()
    for f in analysis.FUNCS:
        row[f"best_{f}"] = mean_f[f].idxmax()
    return row


def run_e5(names=None, variants=None, jobs: int = 4, force: bool = False) -> None:
    """E5: re-run the study under modelling variants and report which verdicts change."""
    names, variants = names or dev_networks(), variants or list(E5_VARIANTS)
    Parallel(n_jobs=jobs)(delayed(_e5_cell)(v, n, force) for v in variants for n in names)
    out = pd.DataFrame([_e5_summary(v, names) for v in variants])
    d = RESULTS / "analysis" / "e5"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "variants.csv", index=False)
    base = out[out.variant == "default"].iloc[0]
    rq1c, h1c, rq4c = ("RQ1 (>=2 functions with mean tau < 0.6)", "H1 direction (SMDS, HSS > W, DF)", "RQ4 (> 50% cells differ)")
    v = pd.DataFrame({"variant": out.variant, rq1c: out["RQ1_functions_below_0.6"] >= 2, h1c: out.H1_direction_holds,
                      rq4c: out.RQ4_holds, **{f"best_{f}": out[f"best_{f}"] for f in ("spread", "comm", "path", "rank")}})
    for c in (rq1c, h1c, rq4c):
        v[c.split(" ")[0] + "_changed"] = v[c] != v.loc[v.variant == "default", c].iloc[0]
    for f in ("spread", "comm", "path", "rank"):
        v[f"best_{f}_changed"] = v[f"best_{f}"] != base[f"best_{f}"]
    v.to_csv(d / "verdict_by_variant.csv", index=False)
    pd.set_option("display.width", 250)
    print(v.to_string(index=False))


# ------------------------------------------------------------------ conclusions
def replication_table(m, share) -> list[dict]:
    """Claims from the literature versus what these data show (contact networks only)."""
    contact = m[m.network.map(domain_of) == "contact"]
    rows = []
    if contact.network.nunique():
        above = contact[contact.apply(lambda r: r.q > share.get(r.network, 1.0), axis=1)]
        piv = above.pivot_table(index=["network", "q"], columns="method", values="spread")
        for base in ("W", "DF"):
            d = (piv["SMDS"] - piv[base]).dropna()
            rows.append({"source": "Pereira et al. 2026 / Soriano-Panos et al. 2025",
                         "claim": f"SMDS preserves spreading better than {base} on contact networks",
                         "ours": f"median F_spread difference {d.median():+.3f}, SMDS higher in {(d > 0).mean():.0%} of {len(d)} cells "
                                 f"({contact.network.nunique()} contact networks)",
                         "agrees": bool(d.median() > 0 and (d > 0).mean() > 0.5), "n_networks": int(contact.network.nunique())})
        sm = above[above.method == "SMDS"]
        if len(sm):
            rows.append({"source": "Correia et al. 2023", "claim": "metric-backbone-sized SMDS keeps community structure in contact networks",
                         "ours": f"mean F_comm of SMDS above the metric size {sm.comm.mean():.2f}",
                         "agrees": bool(sm.comm.mean() >= 0.7), "n_networks": int(sm.network.nunique())})
    return rows


def build_conclusions(names, tag: str = "", synthetic_tag: str | None = None, boot: int = 1000) -> dict:
    """Verdicts (H1-H3, RQ1/3/4), findings, tables and overall text -> results/conclusions.json and results/final_report.md."""
    P = protocol()
    seed, tau_s, tau_f = P["global_seed"], P["thresholds"]["tau_s"], P["thresholds"]["tau_f"]
    w = analysis.wide(analysis.load_results(names, tag=tag))
    m = analysis.method_table(w)
    gap, skill, share = analysis.gap_table(m, tau_s, tau_f), analysis.skill_table(w), metric_shares(names)

    agree = analysis.agreement_summary(m, boot, seed, level=report.FAMILY_LEVEL)
    br = analysis.bridge_regression(gap)
    rq4res = analysis.function_dependence(analysis.winners(m), boot, seed, level=report.FAMILY_LEVEL)
    contact = _contact(m)
    h1t = analysis.pathbased_vs_weight_filters(contact, skill[skill.network.isin(contact.network.unique())])
    h2res = analysis.metric_backbone_communities(contact, share, tau_f, boot, seed, report.FAMILY_LEVEL)
    het, n_syn = None, 0
    if synthetic_tag is not None:
        ms = analysis.method_table(analysis.wide(analysis.load_results(syn_names(), tag=synthetic_tag)))
        het = pd.concat([analysis.weight_heterogeneity_model(analysis.gap_table(ms, tau_s, tau_f), synthetic_meta(ms.network.unique()), f)
                         for f in ("spread", "comm", "path")])
        n_syn = int(ms.network.nunique())

    vs = [report.h1(h1t), report.h2(h2res), report.h3(br), report.rq1(agree), report.rq3_synthetic(het), report.rq4(rq4res)]
    lb, reg, n_real = analysis.leaderboard(skill), report.regime_table(skill), int(m.network.nunique())
    fl = (report.agreement_findings(agree) + report.failure_findings(gap) + report.chance_findings(skill)
          + report.regime_findings(reg, n_real))
    rob_path = RESULTS / "analysis" / "e5" / "verdict_by_variant.csv"
    rob = pd.read_csv(rob_path) if rob_path.exists() else None
    if rob is not None:
        fl += report.robustness_findings(rob, n_real)
        for v in vs:
            col = f"{v['id']}_changed"
            if col in rob.columns and rob[col].any():
                v["caveats"].append("not robust: verdict changes under " + ", ".join(rob.loc[rob[col], "variant"]))
    summary = report.overall_summary(vs, reg, n_real, n_syn)
    out = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "protocol_status": P.get("status"), "confirmatory": P.get("status") == "frozen",
        "networks": sorted(m.network.unique().tolist()), "n_networks": n_real, "n_synthetic": n_syn,
        "thresholds": {"tau_s": tau_s, "tau_f": tau_f}, "verdicts": vs, "findings": fl, "summary": summary,
        "tables": {"agreement": agree.round(4).to_dict("records"), "leaderboard": lb.round(4).to_dict("records"),
                   "regimes": reg.round(4).to_dict("records"), "replication": replication_table(m, share),
                   "h1": h1t.round(5).to_dict("records"), "h2": h2res, "bridge": br.round(5).to_dict("records"),
                   "heterogeneity": None if het is None else het.round(5).to_dict("records"),
                   "robustness": None if rob is None else rob.to_dict("records")},
    }
    (RESULTS / "conclusions.json").write_text(json.dumps(out, indent=1, default=float))
    _write_report(out)
    print(f"{n_real} networks; verdicts: " + ", ".join(f"{v['id']}={v['verdict']}" for v in vs))
    print(summary["answer"])
    return out


def _write_report(c: dict) -> None:
    L = ["# Findings: structural versus functional fidelity of network backbones", "",
         f"_Generated {c['generated']} from {c['n_networks']} networks"
         f"{' and ' + str(c['n_synthetic']) + ' synthetic networks' if c['n_synthetic'] else ''}. Protocol status: "
         f"**{c['protocol_status']}** ({'confirmatory' if c['confirmatory'] else 'EXPLORATORY: protocol not frozen'})._", "",
         "## Answer", c["summary"]["answer"], "", "## Why", *[f"- {t}" for t in c["summary"]["why"]], "",
         "## Which method", *[f"- {t}" for t in c["summary"]["methods"]], "", "## Hypotheses",
         "| ID | Statement | Verdict | n networks | Evidence |", "|---|---|---|---|---|"]
    for v in c["verdicts"]:
        L.append(f"| {v['id']} | {v['statement']} | **{v['verdict']}** | {v['n_networks']} | `{json.dumps(v['observed'])}` |")
    L += ["", "Caveats:", *[f"- {v['id']}: {x}" for v in c["verdicts"] for x in v["caveats"]], "", "## Findings"]
    L += [f"- ({f['strength']}) {f['text']}" for f in c["findings"]]
    L += ["", "## Guidance by budget regime", *[f"- {g}" for g in c["summary"]["guidance"]], "",
          "## Replication of earlier claims", "| Source | Claim | Ours | Agrees |", "|---|---|---|---|"]
    L += [f"| {r['source']} | {r['claim']} | {r['ours']} | {r['agrees']} |" for r in c["tables"]["replication"]]
    L += ["", "## Limits", *[f"- {t}" for t in c["summary"]["limits"]], ""]
    (RESULTS / "final_report.md").write_text("\n".join(L), encoding="utf-8")
