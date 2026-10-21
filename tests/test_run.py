import json

import numpy as np
import pandas as pd

from backbone.data import build_graph, edge_table
from run import run_network


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
    meta = run_network("tiny", G=tiny_graph(), results_dir=tmp_path)
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
