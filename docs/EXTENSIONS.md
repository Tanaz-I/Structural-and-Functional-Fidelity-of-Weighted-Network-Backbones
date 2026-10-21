# Extensions of the base paper

The base paper is Pereira, Costa and Rocha (2026), *Distance backbones optimize spreading dynamics and centrality
ranks in the sparsification of complex networks*, arXiv:2603.14564v2 (summary with page quotes:
[BASE_PAPER.md](BASE_PAPER.md)). This document lists what the project can add to it, with the reason, the test and the
code for each. Pick from it; the work order is in [NEXT_STEPS.md](NEXT_STEPS.md).

**Where the code is.** Entries marked "exists" refer to the full pipeline committed as `a5ef8cc` (tag it as
`full-pipeline-v0` if not yet tagged). The `base-paper` branch is being reduced to the base paper only, so modules such as
`pipeline.py`, `analysis.py`, `studies.py`, `report.py`, `advisor.py` and `export.py` are absent from its working tree. To use an
"exists" item, bring the module back from that commit (`git checkout a5ef8cc -- backbone/<file>`) when you start the extension.

Page numbers refer to the paper's PDF. "Not found" means a keyword search and a read of the extracted text, not proof of
absence, and figure content was not extracted.

## 1. The rule for choosing extensions

**Project goal:** does a backbone that preserves a network's *structure* also preserve what the network *does*, and when
and why do the two disagree?

An extension is worth doing only if it helps answer that. Extensions that do not are listed in section 6.

## 2. Summary

| Id | Extension | Gap in the base paper | Goal fit | Data needed | Effort | Code |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | Structure versus function | measures only function (p. 10 to 12) | **Core** | hospital; pooled test needs 5+ | S | exists |
| X2 | Measure the bridge explanation | stated, not measured (p. 6) | **Core** | hospital; pooled needs more | S | partly, one flaw |
| X3 | Controlled bridge intervention | stated, not tested (p. 6) | **Core** | hospital; then synthetic | M | new |
| X4 | Communities and shortest paths | only centrality and SI (p. 5 to 6) | **Core** | hospital | S | exists |
| X5 | Random baselines and skill scores | none found | **Core** | hospital | S | exists |
| X6 | Test the paper's own operating point | judged on 3 function measures only (p. 6) | **Core** | hospital | M | needs DBS |
| X7 | Methods the paper omitted (NC, HSS) | not compared (Table 2) | Supporting | hospital | S | exists |
| X8 | Other network domains | contact networks only (p. 8) | Supporting | more networks | M | pipeline ready |
| X9 | SIR and other robustness | SIR, SIS listed as future work (p. 8); no beta rule | Supporting | hospital | S | exists |
| X10 | Network properties as explanations | "not completely understood" (p. 8) | Supporting | more networks, synthetic | M | partly |
| X11 | Better statistics | unpaired test on 12 AUC values (p. 5) | Supporting | more networks | S | partly |
| X12 | Fixed-budget view | AUC above a connectivity limit (p. 12) | Supporting | hospital | S | partly |
| X13 | Advisor as a weighted objective | weighted search over one parameter (p. 12) | Optional | hospital | S | exists |
| X14 | Leave-one-network-out advisor test | none | Optional | 5+ networks | M | new |

Effort: S small (hours to a day or two), M medium (a few days), L large. These are rough guesses, not measurements.

Recommended package: **X1 to X5** first, then **X6** and **X9**, then **X8, X10, X11** when more networks exist.

## 3. Core extensions

### X1. Structure versus function

- **What.** Add structural fidelity next to the paper's functional measures, and test whether the structural score
  ranks the methods like the functional scores do. Count the cases where structure passes and function fails.
- **Why.** The paper wants backbones that are "dynamically and structurally faithful" (p. 7) but measures no structure.
  It evaluates only eigenvector-centrality rank (`ρ_C`), infection order (`ρ_t`) and half-infection time (`ξ⁻¹_0.5`).
- **Hypothesis.** Structural fidelity agrees only weakly with the functional scores when ranking methods. *Support:*
  mean Kendall tau-b below 0.6 with an interval over networks. *Refute:* tau at or above 0.6, meaning structural checks
  are adequate. Both are findings.
- **How.** `metrics.f_struct` (reachability, non-isolated share, strength similarity), `analysis.kendall_per_cell`,
  `analysis.gap_table` (failure = structure at least 0.8 while function below 0.6), `analysis.agreement_summary`.
  New: report the paper's own three measures separately (the pipeline currently stores averages; see NEXT_STEPS 3.1).
- **Data.** Hospital gives per-budget scores and a descriptive failure map. The pooled test and its interval need at
  least 5 networks; below that the verdict stays inconclusive by design.
- **Output.** Table of tau per function, failure map by method and budget, both directions of disagreement.
- **Risks.** The thresholds 0.8 and 0.6 were calibrated on synthetic pilot networks; check they make sense on the
  paper's measures. With few networks the interval is wide.

### X2. Measure the bridge explanation

- **What.** Measure how many between-community edges each method keeps, and relate it to how much function is lost.
- **Why.** The paper explains the failure of W and DispFilt this way: "they removed bridges very early on in the
  sparsification process, making the networks disconnected and hindering their spreading ability" (p. 6). No bridge
  measure or test appears in the text I searched.
- **Hypothesis.** Methods that keep proportionally fewer between-community edges lose more spreading fidelity.
  *Support:* a negative coefficient of the bridge-retention ratio on the gap, surviving correction. *Refute:* no
  association. Association on real networks is not causation (see X3).
- **How.** `pipeline.mechanism` computes the ratio (between-community edges kept ÷ between-community edges, over the
  same for within-community) and hub concentration; `analysis.bridge_regression`.
  **Known flaw to fix first:** on real networks the ratio uses the first Louvain partition of the original network, which
  is also one of the 20 partitions that `F_comm` compares against. Use a separate set of Louvain seeds, or report both.
- **Data.** Hospital shows the ratio by method and budget. The regression needs several networks.
- **Output.** Plot of bridge ratio against the loss of spreading fidelity, by method.
- **Risks.** Louvain communities are not ground truth. Synthetic networks give an exact partition.

### X3. Controlled bridge intervention

- **What.** Test the bridge explanation by changing only the bridges. Take a backbone with `k` edges and swap `s` kept
  within-community edges (lowest ranked) for `s` dropped between-community edges (highest ranked), keeping `k` fixed.
  Also do the reverse. Re-measure spreading and communities with the same random numbers.
  Try `s` = 0, 2, 5, 10 and 20 % of `k`, for W, DF and SMDS.
- **Why.** An observation (X2) cannot say whether bridges cause the loss. The paper proposes the explanation but does
  not test it. A swap at a fixed budget isolates the effect of bridges from the effect of size.
- **Hypothesis.** For W and DF, adding bridges at fixed `k` raises spreading fidelity; for SMDS, which already keeps
  bridges, the effect is small. *Refute:* no change, in which case the explanation is not supported.
- **How.** New `run_bridge_intervention` in `studies.py` and a `bridges` command in `run.py`. Reuse
  `pipeline.build_gcache` and `pipeline.score_backbone` (the simulator already shares random numbers between the original
  graph and the backbone) and `filters.edge_order`.
- **Data.** Hospital works, using its Louvain partition. On synthetic networks the planted partition gives an exact
  definition (`network.make_synthetic`, `Network.communities`); that needs the synthetic sweep.
- **Output.** Change in `F_spread` and `F_comm` per bridge added or removed, by method, with a bootstrap interval.
- **Risks.** Swapping edges can disconnect the graph; record connectivity. The result on hospital is one network.

### X4. Communities and shortest paths

- **What.** Add the two functions the paper omits.
- **Why.** The paper's functional evaluation is centrality rank and SI spreading only (section 3.2, p. 5 to 6). It cites
  earlier work that metric backbones keep communities (p. 3), but does not test communities for its methods.
- **Hypothesis.** The best method differs by function. Yan et al. (2018) found community structure robust under weight
  thresholding, so W may do well on communities. *Refute:* one method wins every function (a finding).
- **How.** `metrics.f_comm` (Louvain NMI divided by Louvain's own agreement), `metrics.f_path` (share of pairs whose
  distance grows by at most 10 %), `analysis.winners`, `analysis.function_dependence`.
- **Data.** Hospital shows the per-method values; "winners across networks" needs several.
- **Output.** Scores by function, method and budget; the best method per function and budget.
- **Risks.** Louvain disagrees with itself; the normalisation handles this but check the ceiling on small networks.

### X5. Random baselines and skill scores

- **What.** Compare each method with random removal, so that "preserves function" means more than chance.
- **Why.** The paper reports no random baseline (not found in the text). Hamann et al. (2016) found random edge deletion
  "performs surprisingly well".
- **Hypothesis.** Some methods are no better than weight-proportional random removal on some functions.
- **How.** `filters.NULLS` (R-U uniform, R-W weight-proportional), `analysis.skill_table`
  (skill = (F − F_random) / (1 − F_random)).
- **Data.** Hospital.
- **Output.** Skill by method, function and budget; the share of cells where skill is zero or negative.
- **Risks.** Near the ceiling the skill is unstable; the code reports a plain difference when `1 − F_random < 0.05`.

### X6. The paper's own operating point

- **What.** The paper recommends a backbone at `λ_D ≈ 0.3` that keeps all three of its measures above 80 % of their
  original value (p. 6). Test that backbone on structural fidelity, communities and paths.
- **Why.** It is the paper's headline recommendation, judged on three function measures only. Does the "best" backbone
  also pass structural checks and preserve the functions it ignored?
- **Hypothesis.** The `λ_D ≈ 0.3` backbone preserves function by the paper's measures but may not by communities or
  paths. *Refute:* it passes everything.
- **How.** Needs DBS, which is not implemented: for each edge, find the largest `λ` at which it is a shortest path under
  summed distances `(1/p − 1)^λ`, by binary search on [1e-4, 1e3] with 50 steps (paper Algorithms 1 and 2, p. 16 to 17).
  Then evaluate the backbone at `λ_D = 0.3`. Checks: `λ = 1` equals `filters.metric_backbone_mask`; for hospital the
  paper's Table 2 (p. 18) gives min `λ_ij` 0.133, max 16.276 and mode 0.337.
- **Data.** Hospital (small enough for the exhaustive search).
- **Output.** Scores of the `λ_D ≈ 0.3` backbone on all five measures, next to SMDS at the same size.
- **Risks.** DBS costs about `m × 50` shortest-path runs; fine for hospital, slow for the largest networks.

## 4. Supporting extensions

### X7. Methods the paper omitted
NC and HSS are not in the paper's comparison (Table 2, p. 9). Both exist (`filters.score_noise_corrected`,
`score_hss`). Keep them only if they change a conclusion; otherwise they add comparisons without adding to the goal.

### X8. Other network domains
The paper uses 12 contact networks and 69 exhibit networks, all contact data (Table 1, p. 9; supplement C.7). It says the
approach "is easily generalizable to networks beyond the social domain" (p. 8) but tests none. Add one social or economic
and one biological network (Netzschleuder). The pipeline reads any edge list. Check the weight column of each file before
preparing it. **Needs more datasets.**

### X9. SIR and other robustness
The paper lists SIR and SIS as future work (p. 8) and gives no rule for the transmission parameter `β` (p. 10). Your
choice (median half-infection time about 20 steps) is your own. Report the paper's conclusions under half-infection
targets of 10 and 40, SIR (`EvalConfig.with_sir`), the other proximity map, epsilon, Louvain resolution and uniform
seeds (the existing E5 variants in `studies.py`). Support: verdicts that do not change. Refute: any that flip are
reported, not hidden.

### X10. Network properties as explanations
The paper says "it is still not completely understood how each of those network properties affects sparsification
methods" (p. 8). `network.describe()` computes density, weight spread, weight-degree coupling, clustering, modularity
and the semi-metric share, but nothing calls it. Regress the gap on these across real networks, and use the synthetic
factorial design (mixing, weight spread, coupling; `network.make_synthetic`) for the causal part. **Needs more networks.**

### X11. Better statistics
The paper compares methods with a Mann-Whitney test on 12 AUC values and reports standard deviations over seeds (p. 5, 11).
That treats the methods as independent samples. Add paired per-network differences and a cluster bootstrap over
networks (`analysis.cluster_bootstrap`), and print the number of networks next to every claim. With 4 networks neither
test has much power, so report per-network differences too. **Needs more networks.**

### X12. Fixed-budget view
The paper summarises each curve by the area under it from a limit `X̃` (smallest connectivity-preserving size; for DBS
the metric-backbone size) to 1 (Eq. 16, p. 12). Add fixed-budget scores and the minimum size needed to reach a fidelity
threshold (the paper's own Fig. 9, p. 26) as a cross-check. The two views can rank methods differently.

## 5. Optional extensions

- **X13. Advisor as a weighted objective.** The paper picks a backbone by maximising a weighted sum of its three measures
  and sparsity, over 16 weight scenarios (section 5.4, Table 3, p. 12). `advisor.recommend` already does weighted
  utility with a bootstrap and a "probability of being best". Frame it as a generalisation, and be honest that it is not
  validated on unseen networks.
- **X14. Leave-one-network-out advisor test.** Needs at least 5 networks and will probably give a negative result.
  Report it either way, or state the limit instead.

## 6. Not recommended

| Idea | Why not |
| --- | --- |
| Make DBS faster, or add new path-length families | Improving a method is a different project from structure versus function |
| Directed, bipartite or temporal networks | Out of scope in the design paper |
| Adding more methods only to enlarge the comparison | Does not help the goal unless it changes a conclusion |
| Predicting the best method from network descriptors | Needs many networks; likely to fail with this few. Only as X14 |

## 7. Order and dependencies

```
X1, X4, X5, X12  (existing evaluation, hospital)       X7, X9 (hospital, cheap)
X2 -> X3         (X3 needs the bridge measure and the partition choice)
DBS -> X6        (new code first)
more networks -> X8, X10, X11, X14
synthetic sweep -> X3 on synthetic, X10 causal part
```

## 8. Mapping to the project's questions

| Concept-note question | Extensions |
| --- | --- |
| RQ1: how often does structure pass while function fails, and the reverse? | X1, X4, X5 |
| RQ2: does it depend on network type and budget? | X8, X10, X12 |
| RQ3: which properties of the method and network explain it? | X2, X3, X10 |
| RQ4: can a method be recommended, and does it hold on unseen networks? | X13, X14 |

## 9. What hospital alone can and cannot show

| Extension | Hospital alone | Needs more networks |
| --- | --- | --- |
| X1 | Per-budget scores, the failure map | The verdict on agreement and its interval |
| X2 | Bridge ratio by method and budget | The regression across networks |
| X3 | The swap experiment on one network | Generalisation; exact partitions need synthetic networks |
| X4, X5, X7, X9, X12, X13 | Full result for this network | Statements about networks in general |
| X6 | Complete, if DBS is implemented | Generality |
| X8, X10, X11, X14 | Nothing | Required |

With one network, report results as "on hospital" and do not claim significance.
