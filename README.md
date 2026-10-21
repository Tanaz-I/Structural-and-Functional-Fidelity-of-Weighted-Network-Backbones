# Structural and Functional Fidelity of Weighted Network Backbones

Does preserving a weighted network's structure also preserve what it does? This project extracts backbones
with six methods at identical edge budgets, scores them on structural and functional fidelity (spreading,
communities, shortest paths, node rankings), tests when and why the two disagree, writes the conclusions
from the stored results, and presents everything in an interactive UI.

**Full documentation (status, flow, datasets, methods, measures, every command): [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).
What to do next, in order: [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md). The base paper, summarised with page quotes:
[docs/BASE_PAPER.md](docs/BASE_PAPER.md). Possible extensions of it: [docs/EXTENSIONS.md](docs/EXTENSIONS.md).**

## Layout
```
run.py          command line for every stage (python run.py --help)
protocol.yaml   budgets, seeds, thresholds, hypotheses, the real-network list
backbone/       the library (10 modules: network, filters, metrics, pipeline, analysis, studies, report, advisor, export)
ui/             index.html + style.css + app.js, and data/ with the exported JSON
tests/          test_core.py, test_analysis.py
data/           raw downloads (git-ignored), MANIFEST.csv      results/   everything generated
docs/           DOCUMENTATION.md (the full reference), the concept note, the design PDF; PLAN.md is a pointer
```

## Setup
```
.venv\Scripts\python.exe -m pip install -r requirements.lock
.venv\Scripts\python.exe -m pytest -q          # 37 tests, including agreement with NetBone
```
The code works on any weighted network. A dataset is a standard edge list `data/processed/<name>.csv` (columns u,v,w);
`prepare-data` builds it from a raw file (largest component, no loops, parallel edges merged) and records source,
date and checksum in `data/MANIFEST.csv`:
```
python run.py prepare-data --raw data/raw/hospital_lyon_contacts.dat.gz --name hospital --format contacts --domain contact
python run.py prepare-data --raw my_edges.csv --name mynet --format edgelist     # u v [w]
```
Currently one dataset is prepared: `hospital` (SocioPatterns Lyon ward). More are added by preparing them and
listing the name under `real_networks` in protocol.yaml.

## Pipeline
```
python run.py eval --network all --jobs 8            # 6 methods + 2 random baselines, 8 budgets  -> results/eval (all = protocol list)
python run.py eval --network syn --reduced --jobs 8  # 81 synthetic networks (causal tests)
python run.py samples --network all --jobs 6         # per-unit samples for advisor uncertainty
python run.py analyze                                # E1 agreement + E2 failure atlas
python run.py mechanisms [--synthetic]               # E3 mechanisms + E4 method contrasts
python run.py robustness --jobs 6                    # E5 modelling variants
python run.py conclusions --synthetic-tag ""         # verdicts H1-H3 and RQ1/RQ3/RQ4 -> results/final_report.md
python run.py export --network hospital ...   # JSON for the UI -> ui/data
```
The analysis uses the networks in `real_networks` (protocol.yaml), currently `hospital` only; the synthetic generator
(81 networks, 3 instances for each of the 27 design cells) is used for the causal test. With one network the
cross-network tests (H1, H2, RQ1, RQ4) have no meaningful confidence intervals. The advisor is not validated on unseen
networks. Results are labelled exploratory until the protocol is frozen and tagged.

## Advisor
```
python run.py advise --network hospital --q 0.2 --weights spread=0.7 comm=0.3
python run.py advise --edgelist my_network.csv --q 0.2     # u,v,weight columns; computed live
```

## UI (interactive dashboard)
```
python -m backbone.export            # results/<network>/*.csv -> ui/data.js (needs python run.py first)
python -m http.server -d ui 8000     # open http://localhost:8000  (ui/index.html also opens straight from disk)
```
Dependency-free (vanilla JS + canvas, no build step). Tabs: Overview, Network lens (X slider, five methods, compare-all),
Curves, AUC (own X-tilde vs common limit), Epidemic (SI simulated live in the browser, click a node to seed),
and a lambda study with live Eq. 17 weights.
