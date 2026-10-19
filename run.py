"""Command line for the whole project.  Usage: python run.py <command> --help

  eval              fidelity of 6 methods + 2 random baselines on networks (results/eval/)
  samples           per-unit samples for advisor uncertainty (results/samples/)
  analyze           E1 agreement + E2 failure atlas          (results/analysis/)
  mechanisms        E3 bridge / heterogeneity + E4 contrasts (results/analysis/)
  robustness        E5 modelling variants                    (results/analysis/e5/)
  conclusions       verdicts H1-H3 and RQ1/RQ3/RQ4 -> results/conclusions.json, results/final_report.md
  advise            recommend a method for a network, goal and budget
  export            JSON for the web UI (ui/data/)
  prepare-data      turn a raw edge list or contact list into data/processed/<name>.csv + manifest row
  pilot-thresholds  calibrate tau_s / tau_f on the pilot synthetic networks
  ui-check          headless smoke test of the UI with screenshots
"""
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed  # noqa: E402

from backbone import analysis, studies  # noqa: E402
from backbone.network import load_network, prepare_dataset, syn_names  # noqa: E402
from backbone.pipeline import EvalConfig, evaluate_network  # noqa: E402

ROOT = Path(__file__).resolve().parent
EVAL, SAMPLES = ROOT / "results" / "eval", ROOT / "results" / "samples"


# ---------------------------------------------------------------- compute stages
def _all_names(names):
    return studies.dev_networks() if names == ["all"] else names


def _run_eval(name, cfg, tag):
    warnings.filterwarnings("ignore")
    net, t = load_network(name), time.time()
    df = evaluate_network(net, cfg)
    df.attrs = {}
    EVAL.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EVAL / f"{name}{tag}.parquet")
    print(f"{name}: n={net.n} m={net.m} beta={df.beta.iloc[0]:.4f} rows={len(df)} {time.time() - t:.0f}s", flush=True)


def _run_samples(name, cfg):
    import numpy as np
    warnings.filterwarnings("ignore")
    net, t = load_network(name), time.time()
    df = evaluate_network(net, cfg, with_nulls=False, collect=True)
    samples = df.attrs["samples"]
    np.savez_compressed(SAMPLES / f"{name}.npz", **{k.replace("|", "__"): v for k, v in samples.items()})
    df.attrs = {}
    df.to_parquet(SAMPLES / f"{name}_methods.parquet")
    print(f"{name}: samples {len(samples)} arrays, {time.time() - t:.0f}s", flush=True)


def cmd_eval(a):
    cfg = EvalConfig(n_null_draws=3, n_runs=30, n_runs_null=10) if a.quick else EvalConfig()
    if a.reduced:  # synthetic-sweep protocol: 9 seeds x 50 runs, 10 null draws x 20 runs
        cfg = EvalConfig(seed_frac=0.01, n_runs=50, n_runs_null=20, n_null_draws=10)
    names = syn_names() if a.network == ["syn"] else _all_names(a.network)
    tag = "_quick" if a.quick else ""
    todo = [n for n in names if a.force or not (EVAL / f"{n}{tag}.parquet").exists()]
    print(f"computing {len(todo)} networks (skipping {len(names) - len(todo)} already done)")
    Parallel(n_jobs=a.jobs)(delayed(_run_eval)(n, cfg, tag) for n in todo)


def cmd_samples(a):
    cfg = EvalConfig(seed_frac=0.01, n_runs=50) if a.reduced else EvalConfig()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    todo = [n for n in _all_names(a.network) if not (SAMPLES / f"{n}.npz").exists()]
    Parallel(n_jobs=a.jobs)(delayed(_run_samples)(n, cfg) for n in todo)


# ---------------------------------------------------------------- analysis stages
def _scope(a):
    if getattr(a, "pilot", False):
        return [f"pilot{i}" for i in range(15)], "pilot"
    if a.networks:
        return a.networks, "custom"
    return studies.dev_networks(), "real_dev"


def cmd_analyze(a):
    names, label = _scope(a)
    studies.run_e1_e2(names, label, a.tag, a.boot)


def cmd_mechanisms(a):
    if a.synthetic:
        studies.run_synthetic_e3(a.tag)
    else:
        names, label = _scope(a)
        studies.run_e3_e4(names, label, a.tag)


def cmd_robustness(a):
    studies.run_e5(a.networks, a.variants, a.jobs, a.force)


def cmd_conclusions(a):
    names, _ = _scope(a)
    studies.build_conclusions(names, a.tag, a.synthetic_tag, a.boot)


def cmd_advise(a):
    from backbone import advisor
    wts = {k: float(v) for k, v in (x.split("=") for x in a.weights)}
    if a.edgelist:
        from backbone.network import load_edgelist
        net = load_edgelist(a.edgelist, Path(a.edgelist).stem, "uploaded")
        rec, label = advisor.recommend_for_live_network(net, a.q, wts, a.connected), net.name
    else:
        rec, label = advisor.recommend_for_network(a.network, a.q, wts, a.connected), a.network
    md = advisor.to_markdown(rec, label)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)


def cmd_export(a):
    from backbone.export import export_all
    for row in export_all(a.network, atlas_label=a.atlas_label, with_spread=not a.no_spread):
        print(row)


# ---------------------------------------------------------------- data and tooling
def cmd_prepare_data(a):
    """Standardise a raw file into data/processed/<name>.csv (u,v,w) and record it in data/MANIFEST.csv."""
    net = prepare_dataset(a.raw, a.name, a.format, a.domain, a.source, a.note, a.licence)
    print(f"{net.name}: n={net.n} m={net.m} -> data/processed/{net.name}.csv")


def cmd_pilot_thresholds(a):
    """Failure rate (structure >= tau_s and function < tau_f) over a threshold grid on the pilot networks only."""
    import pandas as pd
    names = [f"pilot{i}" for i in range(15)]
    m = analysis.method_table(analysis.wide(analysis.load_results(names, tag=a.tag)))
    print(f"pilot networks: {m.network.nunique()}  method backbones: {len(m)}\n")
    print(m[m.q >= 0.1][["struct"] + analysis.FUNCS].quantile([.1, .25, .5, .75, .9]).round(2).to_string(), "\n")
    grid = pd.DataFrame({ts: {tf: float(analysis.gap_table(m, ts, tf).failure.mean()) for tf in (0.4, 0.5, 0.6, 0.7)}
                         for ts in (0.7, 0.75, 0.8, 0.85, 0.9)})
    grid.index.name = "tau_f \\ tau_s"
    print(grid.round(3).to_string())


def cmd_ui_check(a):
    """Load every screen in headless Edge/Chrome (light and dark), report console errors, save screenshots."""
    import functools
    import http.server
    import socketserver
    import threading

    from playwright.sync_api import sync_playwright
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "ui"))
    handler.log_message = lambda *x, **k: None
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port, errors = srv.server_address[1], []
    screens = ["arena", "atlas", "curves", "mechanisms", "lens", "theatre", "conclusions", "advisor"]
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge")
        except Exception:
            b = p.chromium.launch(channel="chrome")
        for theme in ("light", "dark"):
            pg = b.new_page(viewport={"width": a.width, "height": 1000})
            pg.on("console", lambda m: errors.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
            for s in screens:
                pg.goto(f"http://127.0.0.1:{port}/#/{s}?net={a.net}&q=0.2&theme={theme}")
                pg.reload()
                pg.wait_for_timeout(1500)
                pg.screenshot(path=str(out / f"{s}_{theme}.png"), full_page=(s == "conclusions"))
            pg.close()
        b.close()
    print(f"{len(errors)} console errors; screenshots in {out}")
    for e in errors[:20]:
        print(" ", e)


# ---------------------------------------------------------------- argument parsing
def main():
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, **kw)
        p.set_defaults(fn=fn)
        return p

    p = add("eval", cmd_eval)
    p.add_argument("--network", nargs="+", required=True, help="dataset names, 'all' (the protocol list) or 'syn' (synthetic networks)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true", help="3 null draws, 30 SI runs (smoke test)")
    p.add_argument("--reduced", action="store_true", help="synthetic-sweep protocol")
    p = add("samples", cmd_samples)
    p.add_argument("--network", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--reduced", action="store_true")

    def scope(p):
        p.add_argument("--networks", nargs="+")
        p.add_argument("--pilot", action="store_true")
        p.add_argument("--tag", default="")

    p = add("analyze", cmd_analyze)
    scope(p)
    p.add_argument("--boot", type=int, default=1000)
    p = add("mechanisms", cmd_mechanisms)
    scope(p)
    p.add_argument("--synthetic", action="store_true")
    p = add("robustness", cmd_robustness)
    p.add_argument("--networks", nargs="+")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p = add("conclusions", cmd_conclusions)
    scope(p)
    p.add_argument("--synthetic-tag", default=None, help="tag of syn0..syn269 results; omit if not run")
    p.add_argument("--boot", type=int, default=1000)
    p = add("advise", cmd_advise)
    p.add_argument("--network")
    p.add_argument("--edgelist", help="CSV with columns u,v[,weight]; computed live")
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--weights", nargs="+", default=["spread=1", "comm=1", "path=1", "rank=1"])
    p.add_argument("--connected", action="store_true")
    p.add_argument("--out")
    p = add("export", cmd_export)
    p.add_argument("--network", nargs="+", required=True)
    p.add_argument("--no-spread", action="store_true")
    p.add_argument("--atlas-label", default="real_dev")
    p = add("prepare-data", cmd_prepare_data)
    p.add_argument("--raw", required=True, help="raw file (edge list 'u v [w]' or temporal contact list 't i j ...')")
    p.add_argument("--name", required=True)
    p.add_argument("--format", choices=["edgelist", "contacts"], default="edgelist")
    p.add_argument("--domain", default="")
    p.add_argument("--source", default="")
    p.add_argument("--note", default="")
    p.add_argument("--licence", default="")
    p = add("pilot-thresholds", cmd_pilot_thresholds)
    p.add_argument("--tag", default="_quick")
    p = add("ui-check", cmd_ui_check)
    p.add_argument("--net", default="hospital")
    p.add_argument("--out", default=str(ROOT / "results" / "ui_shots"))
    p.add_argument("--width", type=int, default=1440)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
