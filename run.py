"""Replication of the base paper on the SocioPatterns contact networks.

    python run.py                              the four default networks (hospital, workplace1, conference,
                                               workplace2); cached CSVs are reused
    python run.py --networks hospital conference
    python run.py --networks french-high-school french-primary-school   optional, slow (about 30 min each)
    python run.py --force                      recompute everything
    python run.py --summary-only               rebuild the cross-network summary from existing results

Everything is written under results/:
    results/<network>/      edges.csv, edge_scores.csv, sweep_rhoC.csv, sweep_si.csv, connectivity_limits.csv,
                            auc_xtilde.csv, auc_fixed_limit.csv, lambda_sweep.csv, optimal_lambda.csv, meta.json,
                            figures/*.png
    results/summary/        table1_check.csv, auc_xtilde_all.csv, auc_fixed_all.csv, tests_xtilde.csv,
                            tests_fixed.csv, figures/*.png
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from backbone import plots, summary
from backbone.backbones import edge_table
from backbone.data import ALL_NETWORKS, NETWORKS, load_network
from backbone.spread import achieved_half_time, si_setup, si_sweep, x_grid
from backbone.sweep import Network, connectivity_limits, sweep

RESULTS = Path("results")
RUNS = 100
SI_POINTS = 21  # X grid for the SI sweeps (0, 0.05, ..., 1, plus the metric backbone size)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached(path, make, force):
    path = Path(path)
    if path.exists() and not force:
        return pd.read_csv(path)
    df = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def run_network(name, force=False, G=None, results_dir=None):
    """Full pipeline for one network; G and results_dir can be injected (used by the tests)."""
    out = Path(results_dir or RESULTS) / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"{name}: loading")
    if G is None:
        G = load_network(name, save_to=out / "edges.csv")
    table = cached(out / "edge_scores.csv", lambda: edge_table(G), force)
    net = Network(table)
    k_metric = int((table.distortion <= 1 + 1e-9).sum())
    x_metric = summary.metric_backbone_X(table, net.n)
    log(f"{name}: {net.n} nodes, {net.m} edges, metric backbone {k_metric} edges (X = {100 * x_metric:.2f}%)")

    rho_df = cached(out / "sweep_rhoC.csv", lambda: sweep(net), force)
    limits = connectivity_limits(net)
    pd.Series(limits, name="X_conn").rename_axis("method").to_csv(out / "connectivity_limits.csv")
    log(f"{name}: rho_C sweep done; connectivity limits {({m: round(v, 3) for m, v in limits.items()})}")

    ctx = {}

    def setup():  # seeds, calibrated beta and original-network runs; computed once, only if some sweep needs it
        if "v" not in ctx:
            seeds, beta, ref = si_setup(net, runs=RUNS)
            info = {"beta": beta, "beta_max_valid": 1.0 / float(net.p.max()),
                    "target_half_time": 20.0, "achieved_half_time": achieved_half_time(ref, seeds),
                    "n_seeds": int(len(seeds)), "seed_nodes": [int(net.nodes[s]) for s in seeds]}
            (out / "si_setup.json").write_text(json.dumps(info, indent=2))
            ctx["v"] = (seeds, beta, ref)
        return ctx["v"]

    si_df = cached(out / "sweep_si.csv",
                   lambda: si_sweep(net, ks=x_grid(net, SI_POINTS), extra_ks=[k_metric], runs=RUNS, setup=setup()),
                   force)
    log(f"{name}: SI sweep done")

    auc_main = summary.auc_table(rho_df, si_df, x_metric, limits=limits)
    auc_fixed = summary.auc_table(rho_df, si_df, x_metric, fixed_limit=x_metric)
    auc_main.to_csv(out / "auc_xtilde.csv", index=False)
    auc_fixed.to_csv(out / "auc_fixed_limit.csv", index=False)

    def lam_df():
        seeds, beta, ref = setup()
        return summary.lambda_sweep(net, seeds, beta, ref, summary.lambda_grid(), RUNS)

    lam = cached(out / "lambda_sweep.csv", lam_df, force)
    opt = summary.optimal_lambda(lam)
    opt.to_csv(out / "optimal_lambda.csv", index=False)
    log(f"{name}: lambda study done")

    fig = out / "figures"
    plots.curves(rho_df, si_df, x_metric, fig / "fidelity_curves.png")
    plots.auc_bars(auc_main, fig / "auc_xtilde.png", f"{name}: AUC above each method's own connectivity limit")
    plots.auc_bars(auc_fixed, fig / "auc_fixed_limit.png", f"{name}: AUC above X = {x_metric:.3f} (metric size)")
    plots.lambda_curves(lam, opt, fig / "dbs_lambda.png")

    if force or not (out / "si_setup.json").exists():
        setup()
    si_info = json.loads((out / "si_setup.json").read_text())
    balanced = opt[opt.scenario == "Balanced"].iloc[0]
    meta = {"network": name, "nodes": net.n, "edges": net.m, "metric_edges": k_metric,
            "metric_X_percent": round(100 * x_metric, 3), "paper_metric_X_percent": ALL_NETWORKS.get(name, (None,) * 4)[3],
            "X_conn": limits, "balanced_lambda_opt": float(balanced.lam_opt), "runs": RUNS,
            "beta": si_info["beta"], "achieved_half_time": si_info["achieved_half_time"],
            "n_seeds": si_info["n_seeds"],
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"{name}: done in {meta['seconds']} s -> {out}/")
    return meta


def build_summary(names):
    out = RESULTS / "summary"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    metas = [json.loads((RESULTS / n / "meta.json").read_text()) for n in names
             if (RESULTS / n / "meta.json").exists()]
    t1 = pd.DataFrame([{"network": m["network"], "nodes": m["nodes"], "paper_nodes": ALL_NETWORKS[m["network"]][1],
                        "edges": m["edges"], "paper_edges": ALL_NETWORKS[m["network"]][2],
                        "metric_X_percent": m["metric_X_percent"],
                        "paper_metric_X_percent": m["paper_metric_X_percent"],
                        "X_difference": round(m["metric_X_percent"] - m["paper_metric_X_percent"], 2),
                        "balanced_lambda_opt": round(m["balanced_lambda_opt"], 3)} for m in metas])
    t1.to_csv(out / "table1_check.csv", index=False)
    done = [m["network"] for m in metas]

    for tag, fname, title in [("xtilde", "auc_xtilde.csv", "AUC above each method's own connectivity limit"),
                              ("fixed", "auc_fixed_limit.csv", "AUC above the metric-backbone size (same limit)")]:
        allauc = pd.concat([pd.read_csv(RESULTS / n / fname).assign(network=n) for n in done])
        allauc.to_csv(out / f"auc_{tag}_all.csv", index=False)
        tests = summary.cross_network_tests(allauc)
        tests.to_csv(out / f"tests_{tag}.csv", index=False)
        plots.auc_boxplots(allauc, out / "figures" / f"auc_boxplots_{tag}.png",
                           f"{title}  ({len(done)} networks, one dot each)")
        med = allauc.groupby("method")[["AUC_rho_C", "AUC_rho_t", "AUC_xi"]].median().round(3)
        log(f"summary [{tag}] median AUC over {len(done)} networks:\n{med.to_string()}")
    log(f"summary written to {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="*", default=list(NETWORKS), choices=list(ALL_NETWORKS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        for name in a.networks:
            run_network(name, a.force)
    build_summary(a.networks)


if __name__ == "__main__":
    main()
