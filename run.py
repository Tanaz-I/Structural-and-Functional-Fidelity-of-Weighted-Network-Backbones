"""Replication of Pereira, Costa & Rocha (2026) on two SocioPatterns contact networks, plus the extension.

    python run.py                run everything from scratch (hospital and workplace1), about 30 minutes
    python run.py --quick        same code path with 10 SI runs per seed (a few minutes), written to results_quick/
    python run.py --force        delete the cached results of the chosen networks first
    python run.py --networks hospital

Stages per network (all outputs under results/<network>/):
    1. replication   edge_scores.csv, sweep_rhoC.csv, sweep_si.csv, auc_xtilde.csv, auc_fixed_limit.csv,
                     lambda_sweep.csv, optimal_lambda.csv, si_setup.json, meta.json, figures/
    2. sensitivity   sensitivity/   beta target, SI runs and DispFilt-normalisation checks
    3. diagnosis     diagnose/      where structure and function disagree, sparse regime, bridge retention
    4. extension     extension/     connectivity fix, shortest-path test, causal bridge test, figure
Cross-network tables are written to results/summary/ and the dashboard data to ui/data.js (open ui/index.html to demo
the results). The numbers discussed in docs/RESULTS.md come from this run.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from backbone import connected, diagnose, export, plots, sensitivity, summary
from backbone.backbones import edge_table
from backbone.data import NETWORKS, load_network
from backbone.spread import achieved_half_time, si_setup, si_sweep, x_grid
from backbone.sweep import Network, connectivity_limits, sweep

RUNS = 100        # SI runs per seed
QUICK_RUNS = 10
SI_POINTS = 21    # X grid for the SI sweeps (0, 0.05, ..., 1, plus the metric backbone size)
MATCH_TOL = 0.01  # percentage points: metric-backbone size counted as matching the paper's Table 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached(path, make):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    df = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def run_network(name, results_dir="results", runs=RUNS, G=None):
    """Stage 1 (replication) for one network. G can be injected (used by the tests)."""
    out = Path(results_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table = cached(out / "edge_scores.csv", lambda: edge_table(G if G is not None else load_network(name)))
    net = Network(table)
    k_metric = int((table.distortion <= 1 + 1e-9).sum())
    x_metric = summary.metric_backbone_X(table, net.n)
    log(f"{name}: {net.n} nodes, {net.m} edges, metric backbone {k_metric} edges (X = {100 * x_metric:.2f}%)")

    rho_df = cached(out / "sweep_rhoC.csv", lambda: sweep(net))
    limits = connectivity_limits(net)
    pd.Series(limits, name="X_conn").rename_axis("method").to_csv(out / "connectivity_limits.csv")
    log(f"{name}: rho_C sweep done; connectivity limits {({m: round(v, 3) for m, v in limits.items()})}")

    ctx = {}

    def setup():  # seeds, calibrated beta and original-network runs; computed once
        if "v" not in ctx:
            seeds, beta, ref = si_setup(net, runs=runs)
            beta_max = 1.0 / float(net.p.max())
            info = {"beta": beta, "beta_max_valid": beta_max,
                    "beta_capped": bool(beta >= beta_max * (1 - 1e-9)),  # target unreachable: fastest possible spread used
                    "target_half_time": 20.0, "achieved_half_time": achieved_half_time(ref, seeds),
                    "n_seeds": int(len(seeds)), "seed_nodes": [int(net.nodes[s]) for s in seeds]}
            (out / "si_setup.json").write_text(json.dumps(info, indent=2))
            ctx["v"] = (seeds, beta, ref)
        return ctx["v"]

    si_df = cached(out / "sweep_si.csv",
                   lambda: si_sweep(net, ks=x_grid(net, SI_POINTS), extra_ks=[k_metric], runs=runs, setup=setup()))
    log(f"{name}: SI sweep done")

    auc_main = summary.auc_table(rho_df, si_df, x_metric, limits=limits)
    auc_fixed = summary.auc_table(rho_df, si_df, x_metric, fixed_limit=x_metric)
    auc_main.to_csv(out / "auc_xtilde.csv", index=False)
    auc_fixed.to_csv(out / "auc_fixed_limit.csv", index=False)

    lam = cached(out / "lambda_sweep.csv",
                 lambda: summary.lambda_sweep(net, *setup(), summary.lambda_grid(), runs))
    opt = summary.optimal_lambda(lam)
    opt.to_csv(out / "optimal_lambda.csv", index=False)
    log(f"{name}: lambda study done")

    fig = out / "figures"
    plots.curves(rho_df, si_df, x_metric, fig / "fidelity_curves.png", network=name)
    plots.auc_bars(auc_main, fig / "auc_xtilde.png", f"{name}: AUC above each method's own connectivity limit")
    plots.auc_bars(auc_fixed, fig / "auc_fixed_limit.png", f"{name}: AUC above X = {x_metric:.3f} (metric size)")
    plots.lambda_curves(lam, opt, fig / "dbs_lambda.png")

    if not (out / "si_setup.json").exists():
        setup()
    si_info = json.loads((out / "si_setup.json").read_text())
    balanced = opt[opt.scenario == "Balanced"].iloc[0]
    paper = NETWORKS.get(name, (None,) * 4)
    meta = {"network": name, "nodes": net.n, "edges": net.m, "metric_edges": k_metric,
            "metric_X_percent": round(100 * x_metric, 3), "paper_nodes": paper[1], "paper_edges": paper[2],
            "paper_metric_X_percent": paper[3], "X_conn": limits, "balanced_lambda_opt": float(balanced.lam_opt),
            "runs": runs, "beta": si_info["beta"], "beta_capped": si_info["beta_capped"],
            "target_half_time": si_info["target_half_time"], "achieved_half_time": si_info["achieved_half_time"],
            "n_seeds": si_info["n_seeds"], "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"{name}: replication done in {meta['seconds']} s")
    return meta


def run_stages(name, results_dir, runs, G=None):
    """All four stages for one network."""
    out = Path(results_dir) / name
    meta = run_network(name, results_dir, runs, G)
    t0 = time.time()
    sensitivity.run(out, runs=runs, log=lambda m: log(f"{name}: {m.strip()}"))
    log(f"{name}: sensitivity done in {time.time() - t0:.0f} s")
    t0 = time.time()
    diagnose.run(out)
    log(f"{name}: diagnosis done in {time.time() - t0:.0f} s")
    t0 = time.time()
    connected.run(out, runs=runs, log=lambda m: log(f"{name}: {m.strip()}"))
    auc = pd.read_csv(out / "extension" / "auc.csv")
    plots.extension_bars(auc[auc.limit == "metric"], out / "figures" / "extension.png",
                         f"{name}: spanning tree and metric backbone added to W and DispFilt")
    log(f"{name}: extension done in {time.time() - t0:.0f} s")
    return meta


def build_summary(names, results_dir):
    """Cross-network tables in results/summary/."""
    root = Path(results_dir)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((root / n / "meta.json").read_text()) for n in names]
    def table1_row(m):
        px = m["paper_metric_X_percent"]
        status = ("not in the paper" if px is None else "replication: matches Table 1"
                  if abs(m["metric_X_percent"] - px) <= MATCH_TOL else "second network: metric size differs from Table 1")
        return {"network": m["network"], "nodes": m["nodes"], "paper_nodes": m["paper_nodes"], "edges": m["edges"],
                "paper_edges": m["paper_edges"], "metric_X_percent": m["metric_X_percent"], "paper_metric_X_percent": px,
                "X_difference": None if px is None else round(m["metric_X_percent"] - px, 2), "status": status,
                "balanced_lambda_opt": round(m["balanced_lambda_opt"], 3)}

    pd.DataFrame([table1_row(m) for m in metas]).to_csv(out / "table1_check.csv", index=False)
    pd.DataFrame([{k: m[k] for k in ("network", "beta", "beta_capped", "target_half_time", "achieved_half_time")}
                  for m in metas]).to_csv(out / "si_calibration.csv", index=False)
    for fname, label in [("auc_xtilde.csv", "xtilde"), ("auc_fixed_limit.csv", "fixed")]:
        pd.concat([pd.read_csv(root / n / fname).assign(network=n) for n in names]).to_csv(
            out / f"auc_{label}_all.csv", index=False)
    sens = {n: pd.read_csv(root / n / "sensitivity" / "sensitivity.csv") for n in names}
    for label, f in [("rankings", sensitivity.ranking_table), ("dispfilt_variants", sensitivity.dispfilt_variants),
                     ("spread_over_settings", sensitivity.spread_over_settings)]:
        pd.concat([f(df).assign(network=n) for n, df in sens.items()]).to_csv(out / f"sensitivity_{label}.csv", index=False)
    pd.concat([pd.read_csv(root / n / "extension" / "verdicts.csv").assign(network=n) for n in names]).to_csv(
        out / "extension_verdicts.csv", index=False)
    log(f"summary written to {out}/")
    fixed = pd.read_csv(out / "auc_fixed_all.csv")
    for m in ("AUC_rho_C", "AUC_rho_t", "AUC_xi"):
        log(f"{m} (same lower limit for all methods):\n{fixed.pivot(index='method', columns='network', values=m).round(3)}")
    log("extension verdicts (metric limit):\n" + pd.read_csv(out / "extension_verdicts.csv").query("limit == 'metric'")[
        ["network", "test", "base", "xi_start", "xi_variant", "xi_control", "xi_DBS", "gap_closed", "verdict"]
    ].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--networks", nargs="*", default=list(NETWORKS), choices=list(NETWORKS))
    ap.add_argument("--quick", action="store_true", help="10 SI runs per seed, written to results_quick/")
    ap.add_argument("--force", action="store_true", help="delete cached results of the chosen networks first")
    a = ap.parse_args()
    runs, results_dir = (QUICK_RUNS, "results_quick") if a.quick else (RUNS, "results")
    t0 = time.time()
    for name in a.networks:
        if a.force:
            shutil.rmtree(Path(results_dir) / name, ignore_errors=True)
        run_stages(name, results_dir, runs)
    build_summary(a.networks, results_dir)
    if not a.quick:
        log(f"dashboard data written to {export.main(a.networks, results_dir)} (open ui/index.html)")
    log(f"all done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
