# Structural and Functional Fidelity of Weighted Network Backbones: project documentation

This is the single reference for what the project is, what exists, what you can do with it and how the parts
connect. The specification it follows is the design paper in `docs/` ("Beyond Structural Fidelity: A
Function-Aware Benchmark of Backbone Extraction ..."). Section 12 lists where the code deliberately differs from it.

## 1. What the project is

**Question.** A backbone keeps a small share of a weighted network's edges. It is normally judged by structural
checks (is it still connected, are the weight and degree distributions similar). Does a backbone that passes those
checks also behave like the original network, that is, does it spread an epidemic, form communities, route along
shortest paths and rank nodes the same way? Where it does not, why not?

**What is being claimed.** The contribution is the structure-versus-function failure analysis: how well structural
fidelity predicts functional fidelity, where the prediction fails, and which mechanism explains the failures
(tested on controlled synthetic networks). Six backbone methods are the *instrument* of this analysis, not the
result. Ranking the methods is a by-product.

**What is not claimed.** That this is the first comparison of backbone methods (Hamann et al. 2016 did a
multi-property comparison), that metric backbones preserve spreading (Correia, Soriano-Paños and Pereira show
this), or that bridge removal explains failures before the mechanism test supports it. Novelty is **unverified**
until the five papers in `papers/` are read in full (section 13).

## 2. Current status

| Area | State |
| --- | --- |
| Library (`backbone/`), command line (`run.py`), 37 tests | Done |
| Six filters, two random baselines, matched-budget controller | Done, validated against NetBone (DF, NC, HSS) and by independent checks (SMDS, wEffRes) |
| Five fidelity measures, skill score, gap and failure rule | Done |
| Experiments E1 to E5, conclusion engine, advisor (evidence mode), UI (8 screens) | Done |
| Datasets | **One real network: hospital** (75 nodes, 1,139 edges). More are added one at a time (section 4) |
| Synthetic networks (for the causal test) | Generator done. 81 networks planned (3 per design cell), 12 evaluated so far |
| Protocol | **Draft**, not frozen. All results are labelled exploratory |
| Reading the five key papers (novelty check) | Not done |

**What one dataset can and cannot show.** All hypothesis tests that pool across networks (see section 7) need
several networks. With one network they return *inconclusive*, and confidence intervals collapse to a single value.
Use the hospital run to check the pipeline and to read per-method numbers (section 3 example), not to draw
conclusions.

## 3. The flow

```
 raw file                                                    results/ and ui/data/
    |  python run.py prepare-data --raw ... --name NAME       data/processed/NAME.csv + data/MANIFEST.csv
    v
 standard edge list (u, v, w): largest component, no loops, parallel edges merged
    |  python run.py eval --network NAME                      results/eval/NAME.parquet
    v
 for each of 6 methods (W, DF, NC, SMDS, HSS, wEffRes) and 2 random nulls (R-U, R-W, 20 draws):
    score every edge -> keep the top q of edges, q = 5, 10, 15, 20, 30, 40, 50, 70 %   (same k for all methods)
    |
    v
 score each backbone against the original network on 5 measures
    structure | spreading (SI epidemic) | communities (Louvain) | shortest paths | node rankings
    |  python run.py samples --network NAME                   results/samples/ (per-unit samples for uncertainty)
    |  python run.py analyze   / mechanisms / robustness      results/analysis/
    v
 E1 agreement   E2 failure atlas   E3 mechanisms   E4 method contrasts   E5 robustness
    |  python run.py conclusions                              results/conclusions.json, results/final_report.md
    v
 written verdicts (H1-H3, RQ1, RQ3, RQ4) + findings, each with the number of networks behind it
    |  python run.py export --network NAME                    ui/data/
    |  python run.py advise --network NAME --q 0.2 ...        advisor report
    v
 interactive UI (python -m http.server -d ui 8000)  and  the Backbone Advisor
```

Every stage reads files written by the previous one, and nothing is recomputed to draw a figure. Evaluation is
restartable: finished networks are skipped.

**Worked example (hospital, from the stored results).** At a 10% budget, HSS scores 0.96 on structure but only 0.53
on spreading, and wEffRes 0.93 on structure but 0.31 on spreading. A backbone that looks faithful structurally can
behave quite differently, which is the central point of the project. At 30% budget, SMDS leads on spreading (0.84),
wEffRes and DF tie on communities (0.84), and HSS and SMDS reach 1.00 on paths: no single method wins every function.

## 4. Datasets

**Standard form.** A dataset is `data/processed/<name>.csv` with columns `u,v,w` (undirected, positive weights).
`load_network("<name>")` reads it. A path to an edge-list file, or a synthetic name (`syn12`, `pilot3`), also works.
Source, date, size and a checksum of the raw file go in `data/MANIFEST.csv`.

**Adding a dataset.**
```
python run.py prepare-data --raw FILE --name NAME --format edgelist    # lines "u v [w]", labels may be text
python run.py prepare-data --raw FILE --name NAME --format contacts    # temporal contacts "t i j ...", weight = number of snapshots
```
Then add `NAME` to `real_networks` in `protocol.yaml`, and run `python run.py eval --network NAME`.
Options: `--domain`, `--source`, `--note`, `--licence`.

**Current dataset.** `hospital`: face-to-face contacts in a Lyon hospital ward (SocioPatterns, CC BY-NC-SA 4.0),
75 nodes, 1,139 edges. The raw file is `data/raw/hospital_lyon_contacts.dat.gz`.

**Synthetic networks.** A weighted community model with 300 nodes. Three factors, three levels each: community mixing
(0.1, 0.3, 0.5), log-normal weight spread (sigma 0.5, 1.0, 1.5), and weight-degree coupling (0, 0.5, 1). That is 27
design cells. `syn_names()` lists the 81 networks used (instances 0 to 2 of each cell; `syn<i>`, i = cell*10 + k).
A separate set of 15 pilot networks (`pilot0` to `pilot14`) was used only to set the failure thresholds.

## 5. The methods

Each method gives every edge a score. The backbone is the top k edges, with k = round(q × number of edges), so all
methods keep the same number of edges. Ties are broken by weight, then by one seeded random order shared by all
methods. The node set is fixed, so a node that loses all its edges stays as an isolate. Retained edges keep their weights.

| Code | Method | Edge score |
| --- | --- | --- |
| W | Global weight threshold | the weight |
| DF | Disparity filter (Serrano 2009) | 1 minus the smaller endpoint null probability (1 − p)^(k − 1) |
| NC | Noise-corrected (Coscia and Neffke 2017) | standard deviations of the weight above its binomial expectation |
| SMDS | Semi-metric distortion (Soriano-Paños 2025) | distortion d / indirect distance; smallest first, so the metric backbone comes first |
| HSS | High-salience skeleton (Grady 2012) | share of shortest-path trees that contain the edge |
| wEffRes | Weighted effective resistance (Mercier 2022) | proximity × effective resistance |
| R-U, R-W | Random baselines | uniform random edges; random edges proportional to weight (20 draws each) |

**Proximity and distance.** p = w / (s_i + s_j − w), where s is node strength; d = 1/p − 1. W, DF and NC use the raw
weight; paths, SMDS, HSS, wEffRes and the epidemic use p and d. The alternative p = w / w_max is tested in E5.

## 6. The measures

All scores are between 0 and 1 and equal 1 when the backbone behaves exactly like the original network.

| Measure | What it compares | Randomness | Sanity check |
| --- | --- | --- | --- |
| F_struct | mean of reachability ratio, non-isolated share, strength-distribution similarity | ties only | 1 at q = 1 |
| F_spread | SI epidemic: infection-order correlation and half-infection-time ratio, 15% of nodes as seeds (top, median, bottom 5% by eigenvector centrality), 100 runs per seed, shared random numbers between original and backbone | runs, seeds | 1 at q = 1 |
| F_comm | NMI between Louvain partitions of original and backbone, divided by Louvain's agreement with itself | Louvain seeds | 1 at q = 1 |
| F_path | share of connected node pairs whose distance grows by at most 10% | ties only | 1 for SMDS at the metric-backbone size |
| F_rank | Spearman correlation of strength, eigenvector and betweenness centrality, clipped at 0 | ties only | 1 at q = 1 |

- **Skill score** = (F_method − F_random) / (1 − F_random), against both random baselines. Negative means worse
  than random removal.
- **Gap** = F_struct − F_function. A **failure** is F_struct ≥ 0.8 together with F_function < 0.6 (thresholds
  calibrated on the pilot networks only).
- **Mechanism measures:** bridge-retention ratio (share of between-community edges kept divided by the share of
  within-community edges kept) and hub concentration.

## 7. Experiments and hypotheses

| Experiment | Question | Command | Output folder |
| --- | --- | --- | --- |
| E1 agreement | Does structural fidelity rank the methods like functional fidelity does? (Kendall tau per cell) | `analyze` | `results/analysis/<label>/e1_*.csv` |
| E2 failure atlas | Where does the structural check pass while function fails? | `analyze` | `e2_atlas.csv`, `e2_failure_rates.csv` |
| E3 mechanisms | Do lost bridges explain failures (real networks, associational)? Do weight spread and coupling hurt weight-based filters (synthetic, causal)? | `mechanisms`, `mechanisms --synthetic` | `e3_*.csv` |
| E4 method contrasts | Which method wins where? Skill against random, winners per cell | `mechanisms` | `e4_*.csv` |
| E5 robustness | Do the verdicts survive other modelling choices (SIR, beta, proximity map, epsilon, Louvain resolution, seed rule)? | `robustness` | `results/analysis/e5/` |

**Verdicts written by `conclusions`** (Holm correction over six tests at 0.05; a verdict is downgraded to
*inconclusive* when fewer than 5 networks stand behind it):

| ID | Statement | Test |
| --- | --- | --- |
| H1 | SMDS and HSS preserve spreading better than W and DF on contact networks | paired Wilcoxon over network-by-budget cells |
| H2 | The metric backbone preserves community structure on contact networks | CI lower limit of mean F_comm of SMDS at the metric size |
| H3 | Failures coincide with fewer between-community edges kept (associational) | mixed-effects regression, likelihood-ratio test |
| RQ1 | Structural fidelity agrees only weakly with functional fidelity | Kendall tau below 0.6 for at least 2 of 4 functions |
| RQ3 | Weight spread and coupling hurt weight-based filters more than SMDS (causal) | method × factor interaction, synthetic design |
| RQ4 | No single method wins on all four functions | share of cells with different winners, CI above 0.5 |

H1 to H3 are the hypotheses of the concept note. RQ1, RQ3 and RQ4 answer research questions, for which the design
paper states no direction. A rejected hypothesis is reported as a finding.

## 8. What you can do (command reference)

Run from the project root with the project's Python: `.venv\Scripts\python.exe run.py <command>`.

| Task | Command |
| --- | --- |
| Prepare a dataset | `prepare-data --raw FILE --name NAME --format edgelist|contacts` |
| Evaluate networks (6 methods, 2 nulls, 8 budgets, 5 measures) | `eval --network hospital --jobs 4` |
| Quick smoke run | `eval --network hospital --quick` |
| Evaluate the synthetic sweep | `eval --network syn --reduced --jobs 8` |
| Per-unit samples for advisor uncertainty | `samples --network hospital` |
| E1 and E2 | `analyze` (or `analyze --networks hospital`) |
| E3 and E4 | `mechanisms`; the synthetic part: `mechanisms --synthetic` |
| E5 robustness | `robustness --jobs 4` |
| Verdicts and written report | `conclusions` (add `--synthetic-tag ""` once the synthetic sweep is done) |
| Recommend a method | `advise --network hospital --q 0.2 --weights spread=0.7 comm=0.3` |
| Recommend for your own edge list | `advise --edgelist my_network.csv --q 0.2` (computed live) |
| Export data for the UI | `export --network hospital` |
| Open the UI | `python -m http.server -d ui 8000`, then http://localhost:8000 |
| UI smoke test with screenshots | `ui-check` (writes `results/ui_shots/`) |
| Run the tests | `.venv\Scripts\python.exe -m pytest -q` |

`--networks` (analysis commands) and `--network` (eval, samples, advise, export) take names. `eval --network all` and
`analyze` without options use the list `real_networks` in `protocol.yaml`.

**Running time (measured).** One synthetic network takes about 6.5 minutes at the reduced settings, of which about
half is betweenness centrality (rankings), 40% is Louvain (communities), and only about 7% is the epidemic. The random
baselines (320 of the 368 backbones per network) account for most of the time. Use `--jobs` to run networks in parallel.

## 9. Where things are

```
run.py            command line for every stage
protocol.yaml     budgets, seeds, thresholds, hypothesis text, real_networks, synthetic_instances
backbone/
  network.py      Network class, dataset loading and preparation, synthetic generator, descriptors
  filters.py      six edge filters, budget controller, random nulls
  metrics.py      structure, spreading (SI, SIR), communities, paths, rankings
  pipeline.py     evaluates every method and null at every budget on one network
  analysis.py     result tables, bootstrap, statistics, E1/E3/E4 analyses
  studies.py      experiments E1 to E5 and the conclusion builder
  report.py       verdicts and the auto-written findings
  advisor.py      Backbone Advisor (evidence mode)
  export.py       JSON export for the UI
ui/               index.html, style.css, app.js; data/ holds the exported JSON
tests/            test_core.py, test_analysis.py
data/             raw/ (downloads), processed/ (standard edge lists), MANIFEST.csv
results/          everything generated: eval/, samples/, analysis/, e5/, conclusions.json, final_report.md
docs/             this document, the concept note, the design paper (PDF)
papers/           the five key papers for the novelty check
```

## 10. The UI and the advisor

The UI needs no build step and no server code. Screens:

| Screen | Question it answers |
| --- | --- |
| Arena | How do the six methods differ at the same budget? |
| Failure atlas | When does a structural check mislead? |
| Curves | Where does each method hold up, and where is it no better than random? |
| Mechanism bench | Why do failures occur? |
| Network lens | What does each method keep? Why was this edge kept or dropped? |
| Spread theatre | Does the epidemic spread the same way on the backbone? |
| Conclusions | What do the verdicts and findings say? |
| Advisor | Which method should I use, and how sure is that? |

**Advisor (evidence mode).** Inputs: a network, objective weights over spreading, communities, paths and rankings
(summing to 1), a budget q, and an optional "must stay connected" constraint. It runs the six methods on that network,
computes a weighted utility with a bootstrap interval and the probability that each method is best, and recommends the
top one: *clear* if that probability is at least 0.8, a *close call* otherwise. It also shows the trade-offs, the
evidence behind them, and whether the choice changes one budget step either way. Example for hospital at q = 0.2 with
weights spread 0.7 and communities 0.3: wEffRes, clear (95%). Uncertainty here comes from simulation and algorithm noise
on one network, not from variation across networks. The advisor has **not** been validated on unseen networks.

## 11. Reproducibility and tests

- One global seed (`global_seed` in `protocol.yaml`) with derived seeds per component. Dataset sources, dates and
  checksums are in `data/MANIFEST.csv`. Software versions are pinned in `requirements.lock`.
- The protocol is a **draft**. Until it is frozen and tagged in git, every result is labelled exploratory.
- 37 tests: filters agree with NetBone (DF, HSS, and the exact variant of NC), the metric backbone preserves every
  shortest path (checked by brute force), effective resistance matches the pseudoinverse, all scores equal 1 at q = 1
  and stay within [0, 1], plus checks of the analysis and advisor code. The tests use a small built-in graph and need no dataset.

## 12. Limits and deviations from the design paper

| Topic | Design paper | Here |
| --- | --- | --- |
| Real networks | About 12 | 1 (hospital) for now, added one at a time |
| Locked test networks for E6 | 3, used once for the predictive advisor (tier P2, cut first) | Not used; advisor is evidence mode only |
| Synthetic instances per cell | 10 ("at least 10 per setting") | 3, so 81 networks; fewer instances widen the intervals |
| Hypotheses | H1 agreement, H2 bridges, H3 weight heterogeneity, H4 path-based beats W/DF, H5 function dependence | The concept note's H1 to H3, plus RQ1, RQ3, RQ4 for the rest (mapping below) |
| Functions | Spreading and communities lead; paths and rankings support | All four are scored and tested equally |
| Dashboard | Streamlit and Plotly | A custom HTML/JS app |

Mapping of the design paper's hypotheses to the verdicts in the code: agreement → RQ1; bridge mechanism → H3;
weight heterogeneity → RQ3; path-based advantage → H1 (now SMDS and HSS, spreading, contact networks);
function dependence → RQ4. H2 (metric backbone and communities) comes from the concept note.

**Known inconsistency to fix before freezing the protocol:** the `hypotheses:` text in `protocol.yaml` still lists the
design paper's H1 to H5 and does not match the verdict names above.

**Limits of the study.** Static, undirected networks; SI and SIR with one weight-to-probability map; networks up to a
few hundred nodes; few real networks, so claims hold only for the tested domains and sizes; causal statements only
from the synthetic design.

## 13. How to extend

The prioritised work list, with the evidence for each item and a test for when it is done, is in [NEXT_STEPS.md](NEXT_STEPS.md). In short:

1. **Settle the novelty claim.** Read, in full, the papers in `papers/`: Pereira et al. 2026 (2603.14564), Soriano-Paños
   et al. 2025 (2311.14817), Correia et al. 2023 (journal.pcbi.1010854), Hamann et al. 2016 (1601.00286) and Yan et al.
   2018 (1806.07479). For each, answer: did they test whether structural fidelity predicts functional fidelity across
   methods, networks and budgets? If one did, narrow the central question to what remains.
2. **Add datasets one at a time** (section 4), re-running `eval`, `analyze`, `mechanisms`, `conclusions` and `export`.
   Aim for at least 5 real networks from at least 3 domains; below 5, verdicts stay *inconclusive* by design.
3. **Run the synthetic sweep** (`eval --network syn --reduced`) and `mechanisms --synthetic` to test RQ3.
4. **Run the robustness checks** on the final network set (`robustness`).
5. **Freeze the protocol:** align the hypothesis text, set `status: frozen`, and tag the commit `protocol-v1`.
6. **Write the paper** around the findings, with the number of networks beside every claim.
