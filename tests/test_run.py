import json

import numpy as np
import pandas as pd

from backbone.data import build_graph, edge_table
from backbone import export
from run import build_summary, run_network, run_stages


def tiny_graph(n=14, seed=0):
    """Ring plus random chords, with random contact counts: connected, weighted, a few semi-metric edges."""
    rng = np.random.default_rng(seed)
    pairs = {tuple(sorted((k, (k + 1) % n))) for k in range(n)}
    while len(pairs) < 3 * n:
        a, b = rng.choice(n, 2, replace=False)
        pairs.add((int(min(a, b)), int(max(a, b))))
    rows = [(t, a, b) for a, b in sorted(pairs) for t in range(1, int(rng.integers(1, 30)) + 1)]
    return build_graph(edge_table(pd.DataFrame(rows, columns=["t", "i", "j"])))


def test_run_network_writes_all_outputs(tmp_path):
    meta = run_network("tiny", results_dir=tmp_path, runs=10, G=tiny_graph())
    out = tmp_path / "tiny"
    for f in ["edge_scores.csv", "sweep_rhoC.csv", "sweep_si.csv", "connectivity_limits.csv", "auc_xtilde.csv",
              "auc_fixed_limit.csv", "lambda_sweep.csv", "optimal_lambda.csv", "si_setup.json", "meta.json"]:
        assert (out / f).exists(), f
    for f in ["fidelity_curves.png", "auc_xtilde.png", "auc_fixed_limit.png", "dbs_lambda.png"]:
        assert (out / "figures" / f).exists(), f
    saved = json.loads((out / "meta.json").read_text())
    assert saved["nodes"] == 14 and saved["edges"] == meta["edges"] == 42
    assert saved["beta"] > 0 and saved["n_seeds"] >= 1 and saved["paper_metric_X_percent"] is None
    auc = pd.read_csv(out / "auc_fixed_limit.csv")
    assert set(auc.method) == {"W", "DispFilt", "wEffRes", "SmD", "DBS"}
    assert auc[["AUC_rho_C", "AUC_rho_t", "AUC_xi"]].notna().all().all()


def test_run_stages_end_to_end_on_a_tiny_network(tmp_path):
    """All four stages and the summary run on a tiny network and write their outputs."""
    run_stages("tiny", tmp_path, runs=3, G=tiny_graph())
    out = tmp_path / "tiny"
    for f in ["sensitivity/sensitivity.csv", "diagnose/dissociation.csv", "diagnose/sparse_regime.csv",
              "diagnose/retention.csv", "extension/auc.csv", "extension/gap_closed.csv", "extension/verdicts.csv",
              "figures/extension.png"]:
        assert (out / f).exists(), f
    verdicts = pd.read_csv(out / "extension" / "verdicts.csv")
    assert set(verdicts.test) == {"fix", "paths", "bridges"} and set(verdicts.base) == {"W", "DispFilt"}
    assert set(verdicts.verdict) <= {"supported", "partial", "not supported"}
    build_summary(["tiny"], tmp_path)
    t1 = pd.read_csv(tmp_path / "summary" / "table1_check.csv")
    assert t1.status.iloc[0] == "not in the paper"
    assert (tmp_path / "summary" / "extension_verdicts.csv").exists()


def test_export_writes_dashboard_data_with_variants_and_no_nan(tmp_path):
    run_stages("tiny", tmp_path, runs=3, G=tiny_graph())
    build_summary(["tiny"], tmp_path)
    out = export.main(["tiny"], tmp_path, tmp_path / "data.js")
    text = out.read_text()
    assert text.startswith("window.DATA = ") and "NaN" not in text and "Infinity" not in text
    data = json.loads(text[len("window.DATA = "):-2])
    net = data["networks"]["tiny"]
    for v in export.VARIANTS:                      # connected variants for the lens, the epidemic tab and the repair card
        assert len(net["rank"][v]) == net["meta"]["edges"] and v in net["si"] and v in net["rho"]
    assert {"sens", "diag", "ext"} <= set(net) and "matched" in net["diag"]
