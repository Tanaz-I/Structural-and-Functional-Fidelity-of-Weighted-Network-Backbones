# Replication of Pereira, Costa and Rocha (2026), and an extension: results

Base paper: *Distance backbones optimize spreading dynamics and centrality ranks in the sparsification of complex networks*,
arXiv:2603.14564v2 (`papers/2603.14564v2.pdf`). Every number below comes from one run of `python run.py` (about 30 minutes) and
is stored in `results/`; the dashboard (`ui/index.html`) shows the same tables interactively.

## Scope
Two SocioPatterns contact networks: **hospital** (75 nodes, 1,139 edges) and **workplace1** (92 nodes, 755 edges). Five methods are
compared: W (weight threshold), DispFilt (disparity filter), wEffRes (weighted effective resistance), SmD (semi-metric distortion)
and DBS (distance backbone synthesis). Measures: `ρ_C` (eigenvector-centrality rank), `ρ_t` (SI infection-order rank) and `ξ⁻¹_0.5`
(SI half-infection-time ratio), each summarised as the area under the curve (AUC) over the adjusted size X (0 = spanning tree,
1 = full graph). Structural fidelity here means measures computed from the static graph (`ρ_C`, connectivity, bridge retention);
functional fidelity means outcomes of the SI process (`ρ_t`, `ξ⁻¹`). This split is ours, not the paper's.

## Data check against the paper's Table 1 (metric-backbone adjusted size)
| Network | Nodes, edges | Ours | Paper | Status |
| --- | --- | --- | --- | --- |
| hospital | 75, 1,139 | 13.43% | 13.43% | **matches** |
| workplace1 | 92, 755 | 13.25% | 20.63% | **differs** |

workplace1 has the paper's node and edge counts, so only its edge weights differ. The file is the official SocioPatterns one (row count
equal to Netzschleuder's). The paper's backbone needs 228 metric edges, ours has 179; Correia et al. 2023 reports the same larger
backbones for the other workplace and conference data. Ruled out: duplicate rows, other proximity formulas (r/max, r/min, cosine,
r/(rii+rjj), d = 1/r), looser tolerance, rounding, merging 20 s steps into contact intervals (any mix of raw and merged counts), and
single days. The cause is not found and the paper gives no code or data link. **Hospital is the verified replication; workplace1 is a
second network** (`status` column of `results/summary/table1_check.csv`). The two French school networks match Table 1 exactly in a
separate one-off check, but are too slow to run through the pipeline.

One-off check of the λ distribution against paper Table 2 (not part of `run.py`): hospital's min and max λ match exactly (0.133 and
16.276), which supports the DBS implementation. workplace1's do not (ours 0.140 and 17.4 against 0.207 and 39.0), more evidence that
its weights differ. The log-normal fit of X(λ) is not reproduced: the paper's μ agrees (hospital −0.917 against −0.937), but our σ is
0.6 to 0.7 against 0.387, for every fitting variant tried (all edges, log grid, central range, probit); the fitting recipe is unstated.

## Conventions where the paper is silent
- **β:** calibrated so the full graph's mean median half-infection time is 20 steps: hospital β = 1.51, workplace1 β = 1.23 (both reach
  the target; `results/summary/si_calibration.csv`).
- **AUC lower limit X̃:** the smallest X from which a method stays connected, never below the metric-backbone size. Reported two ways:
  each method's own X̃, and one common limit (the metric size) for every method.
- **λ_ij:** the largest λ at which an edge is in the backbone; ties in SmD are broken by a seeded random order.
- **wEffRes** uses proximity as conductance. **DispFilt** uses the paper's formula with p = proximity.
- SI model: 100 runs per seed, 15% of nodes as seeds (top, median, bottom by eigenvector centrality).

## Replication: AUC with the same lower limit for all methods (X̃ = metric-backbone size)
| Method | `ρ_C` hospital | `ρ_C` workplace1 | `ρ_t` hospital | `ρ_t` workplace1 | `ξ⁻¹` hospital | `ξ⁻¹` workplace1 |
| --- | --- | --- | --- | --- | --- | --- |
| W | 0.946 | 0.955 | 0.897 | 0.953 | 0.782 | 0.870 |
| DispFilt | 0.954 | 0.964 | 0.781 | 0.881 | 0.672 | 0.777 |
| wEffRes | 0.896 | 0.944 | 0.958 | 0.963 | 0.875 | 0.884 |
| SmD | 0.920 | 0.964 | 0.971 | 0.971 | 0.870 | 0.896 |
| DBS | 0.911 | 0.967 | 0.970 | 0.971 | 0.862 | 0.896 |

- **Spreading (`ρ_t`, `ξ⁻¹`):** the paper's pattern holds on both networks. DBS, SmD and wEffRes are ahead of W and DispFilt, and DBS and SmD
  are practically tied, as the paper also says.
- **`ρ_C`:** mixed. It fails on hospital (W and DispFilt beat DBS by 0.035 and 0.043) and is a near-tie on workplace1 (DBS 0.967, DispFilt
  0.964, W 0.955). With two networks this is a direction, not a test.
- **Caveat on X̃:** with each method's own X̃ (the paper's rule; `results/summary/auc_xtilde_all.csv`) W and DispFilt look best, for example hospital
  `ρ_C` AUC 0.99 and 1.00, because their X̃ is large (hospital 0.474 and 0.800; workplace1 0.220 and 0.524) and they are only scored where they
  are connected. The common-limit table is the fairer comparison; the paper's own rule is not reproduced.
- W and DispFilt lose connectivity earlier than SmD and DBS on both networks.
- **Optimal λ_D (Eq. 17, balanced weights):** hospital 0.346 (X = 0.558), workplace1 0.607 (X = 0.271). The paper reports about 0.3 and a
  log-normal mode of 0.3 ± 0.1; hospital agrees, workplace1 is higher. One stochastic run per λ on a coarse grid, so differences of a few
  hundredths are not meaningful.

## Sensitivity (`results/<network>/sensitivity/`, summary in `results/summary/sensitivity_*.csv`)
- **β target 40 and 80, and 300 SI runs:** the ranking of the methods does not change apart from ties of about 0.01 between DBS and SmD on
  `ξ⁻¹`. Each method's AUC moves by at most 0.025 across these settings (at most 0.014 on `ρ_t`); `ρ_C` does not move because it does not
  depend on the SI model. More SI runs lift every `ρ_t` AUC by about 0.01, which is the noise floor.
- SmD tie-breaking seeds (checked on hospital): no effect on the AUC, because above the metric-backbone size all tied edges are kept whatever the
  order.
- **DispFilt with per-node normalised weights (Serrano) is much better than the paper's formula on spreading:**

| Network | `ρ_t` paper formula | `ρ_t` normalised | `ξ⁻¹` paper formula | `ξ⁻¹` normalised |
| --- | --- | --- | --- | --- |
| hospital | 0.781 | 0.919 | 0.672 | 0.807 |
| workplace1 | 0.881 | 0.968 | 0.777 | 0.880 |

The paper does not say which version it used, so the claim that DispFilt is poor holds for one version only.

## Diagnosis: where and why backbones fail (`results/<network>/diagnose/`)
"Bridge" = an edge between two Louvain communities of the original graph (10 seeds). 20 budgets with X > 0.05 per network.

- **Structure and function disagree, in both directions.** DispFilt keeps `ρ_C ≥ 0.9` while `ρ_t < 0.8` at 40% of budgets on hospital and 25% on
  workplace1. The reverse also happens: DBS and SmD have `ρ_t ≥ 0.9` with `ρ_C < 0.8` at 20% of budgets on hospital (0% on workplace1), for example
  hospital X = 0.2: DBS `ρ_C` 0.69, `ρ_t` 0.95. So `ρ_C` is a poor guide to spreading fidelity.
- **Fragmentation tracks the spreading failure.** At the same X, across the five methods, the number of connected components is strongly negatively
  related to `ρ_t` (mean Spearman across budgets: −0.82 hospital, −0.79 workplace1) while `ρ_C` is not penalised by it (+0.72, +0.59). Eigenvector
  centrality sits on the dense core, so it does not see the lost connections that stop an epidemic. At hospital X = 0.2, W and DispFilt have 3 and
  9 components, DBS, SmD and wEffRes have one.
- **Bridge retention barely separates the methods** at equal X (hospital X = 0.2: DBS 0.18, DispFilt 0.16, W 0.15), although `ρ_t` differs by 0.2 to
  0.3; only on workplace1 at X = 0.5 is the gap sizeable (0.49 against 0.36). The matched-X correlation with `ρ_t` is moderate (0.50, 0.66) and
  confounded with connectivity.
- **Sparse regime (X below the metric-backbone size, not covered by the paper's AUC).** All methods degrade. At X = 0.05 DBS is best on `ρ_t` (hospital
  0.76, workplace1 0.86); at the spanning tree every method is poor (hospital `ρ_t` 0.10 to 0.35; DBS `ρ_C` 0.06).
- Exploratory, not in `run.py`: above the metric backbone the edges only DBS keeps (not W+B) are mostly between-community (82 to 94% against 34 to
  59%) and weaker; the causal bridge test below shows this does not explain the gap.
- The matched-X correlations use five methods per budget, so they are descriptive.

## Extension: why do W and DispFilt lose spreading fidelity, and what do distance backbones add? (`results/<network>/extension/`)
Three tests on the same budgets, seeds, β and reference runs as the replication. M = the method as in the paper; **M+T** = a spanning tree built
along M's own ranking first, then M's top edges up to the budget; **M+R** = the same with a random spanning tree (control); **M+B** = the metric
backbone (connected, every shortest path preserved) first, then M's order; **M+X** = the same number of random edges instead (control); and three
fills that keep the metric backbone and fill the rest of the budget from M's ranking with between-community, within-community or random edges first.
Controls are the mean of 3 random draws; the bridge fills use 3 Louvain partitions. A variant "closes" the share of the gap to DBS
`(AUC(variant) − AUC(start)) / (AUC(DBS) − AUC(start))`; the verdict rules were fixed before the runs and use a noise band of 0.02 (the largest spread of the
random controls on `ξ⁻¹`): "supported" needs at least half of the gap closed on `ξ⁻¹` and a margin over the control larger than 0.02.

AUC from the metric-backbone size (DBS: hospital `ρ_C` 0.911, `ρ_t` 0.970, `ξ⁻¹` 0.862; workplace1 0.967, 0.971, 0.896):

| Variant | hospital `ρ_C` | `ρ_t` | `ξ⁻¹` | workplace1 `ρ_C` | `ρ_t` | `ξ⁻¹` |
| --- | --- | --- | --- | --- | --- | --- |
| W | 0.946 | 0.897 | 0.782 | 0.955 | 0.953 | 0.870 |
| W+T (spanning tree) | 0.946 | 0.953 | 0.811 | 0.955 | 0.963 | 0.871 |
| W+R (random tree) | 0.938 | 0.959 | 0.785 | 0.940 | 0.956 | 0.841 |
| W+B (metric backbone first) | 0.934 | 0.961 | 0.817 | 0.953 | 0.965 | 0.880 |
| W+X (random edges) | 0.924 | 0.947 | 0.776 | 0.930 | 0.958 | 0.842 |
| W, between-community fill | 0.864 | 0.951 | 0.774 | 0.944 | 0.961 | 0.859 |
| W, within-community fill | 0.917 | 0.952 | 0.805 | 0.929 | 0.951 | 0.839 |
| W, random fill | 0.830 | 0.954 | 0.759 | 0.943 | 0.958 | 0.813 |
| DispFilt | 0.954 | 0.781 | 0.672 | 0.964 | 0.881 | 0.777 |
| DispFilt+T | 0.952 | 0.948 | 0.751 | 0.964 | 0.958 | 0.825 |
| DispFilt+R | 0.944 | 0.949 | 0.693 | 0.951 | 0.948 | 0.797 |
| DispFilt+B | 0.936 | 0.959 | 0.766 | 0.959 | 0.961 | 0.855 |
| DispFilt+X | 0.920 | 0.951 | 0.750 | 0.930 | 0.949 | 0.798 |
| DispFilt, between-community fill | 0.869 | 0.950 | 0.740 | 0.945 | 0.960 | 0.844 |
| DispFilt, within-community fill | 0.917 | 0.950 | 0.779 | 0.932 | 0.949 | 0.821 |
| DispFilt, random fill | 0.833 | 0.958 | 0.762 | 0.941 | 0.957 | 0.815 |

Share of the gap to DBS closed on `ξ⁻¹` (metric limit) and verdict:

| Test (start → variant) | hospital W | hospital DispFilt | workplace1 W | workplace1 DispFilt |
| --- | --- | --- | --- | --- |
| Connectivity fix (M → M+T) | 36%, partial | 41%, partial | 5%, not supported | 40%, partial |
| Shortest paths (M+T → M+B) | 12%, not supported | 13%, not supported | 34%, not supported | 42%, partial |
| Bridges (M+B → between-community fill) | negative, not supported | negative, not supported | undefined, not supported | negative, not supported |

On `ρ_t` the spanning tree closes 77% (W) and 88% (DispFilt) of the gap to DBS on hospital and 86% (DispFilt) on workplace1; W on workplace1 has almost no
`ρ_t` gap to close (0.953 against 0.971, gain 0.010). The tree costs no `ρ_C`. Down to the spanning tree (lower limit X = 0) the pattern is the same
(`results/<network>/extension/gap_closed.csv`, `verdicts.csv`).

- **Connectivity explains most of the infection-order failure.** A spanning tree alone lifts DispFilt's `ρ_t` from 0.781 to 0.948 on hospital and
  from 0.881 to 0.958 on workplace1, at no cost to `ρ_C`; a random tree does nearly as well on `ρ_t`, so for the infection order it is connectivity itself.
- **It does not recover the timescale.** `ξ⁻¹` closes about 40% at most (W on workplace1: 5%, with almost no gap). The guided tree beats the random one on `ξ⁻¹`
  on both networks (W +0.026 hospital, +0.030 workplace1; DispFilt +0.058, +0.028; random-tree std 0.003 to 0.026), so which tree matters for the timescale,
  but the margins are about the size of the noise.
- **Shortest-path preservation explains part of the remaining gap** (12 to 13% on hospital, 34 to 42% on workplace1), never half. Forcing metric edges beats
  forcing the same number of random edges (W hospital 0.817 against 0.776; workplace1 0.880 against 0.842), but on hospital it adds only 0.006 (W) and 0.015 (DispFilt) on top of the spanning tree.
- **Between-community ("bridge") edges do not explain it.** No fill beats the plain metric-backbone fill (W: 0.817 hospital, 0.880 workplace1). On hospital
  within-community fill is better than between-community (0.805 against 0.774), on workplace1 the order flips (0.839 against 0.859), so the sign of any
  bridge effect depends on the network; between-community fills also cost `ρ_C` on hospital (W 0.864 against 0.917).

## Conclusion
**Question.** Why do distance backbones (DBS, SmD) preserve spreading better than weight thresholding (W) and the disparity filter (DispFilt), and does a
centrality-rank measure (`ρ_C`) capture that? The extension is explanatory: it decomposes the advantage into parts and tests each; the spanning-tree repair is a
by-product, not a new method.

**Headline.** On the two contact networks tested, the advantage of distance backbones over W and DispFilt splits into (i) a connectivity part, which accounts for
most of the infection-order advantage and is recovered by guaranteeing a spanning tree, and (ii) a timescale part, which shortest-path preservation explains only in
part (12 to 42% depending on the network and method) and between-community edges do not explain. The rest is left to DBS's λ-based choice of added edges, which was not
isolated.

| # | Claim | Evidence | Networks |
| --- | --- | --- | --- |
| C1 | The paper's spreading result replicates: DBS, SmD, wEffRes beat W and DispFilt on `ρ_t` and `ξ⁻¹`; the ranking is stable under β, SI-run count and tie-breaking | AUC tables, sensitivity | hospital (verified), workplace1 |
| C2 | The paper's centrality claim is mixed: fails on hospital, a near-tie on workplace1; the weakness of DispFilt depends on its weight normalisation | `ρ_C` AUC; sensitivity | 2 |
| C3 | Centrality-rank fidelity (`ρ_C`) does not predict spreading fidelity, in either direction, and cannot see fragmentation | dissociation table; matched-X correlation of components with `ρ_t` about −0.8 against `ρ_C` not penalised | 2 |
| C4 | Fragmentation explains most of the infection-order failure of W and DispFilt: a spanning tree closes 77 to 88% of the `ρ_t` gap (86% for DispFilt on workplace1; W has little gap there) at no cost to `ρ_C` | connectivity fix with random-tree control | 2 |
| C5 | The timescale (`ξ⁻¹`) is only partly recovered by connectivity (about 40% at most), and the choice of tree matters a little | same | 2 |
| C6 | Preserving every shortest path (metric backbone first) explains part of the remaining `ξ⁻¹` gap: 12 to 13% on hospital, 34 to 42% on workplace1 | `+B` test | 2 |
| C7 | Between-community edges do not explain it: forcing them never beats the plain metric-backbone fill, and the effect has opposite sign on the two networks | causal bridge test | 2 |

**What is not claimed.** That the spanning-tree repair beats or replaces DBS or SmD (it does not: hospital `ξ⁻¹` 0.811 against 0.870 for SmD), that it is cheaper (not timed),
that the results hold beyond contact networks, or any significance level.

**Practical reading.** Judge a backbone on connectivity and on a dynamical measure, not on `ρ_C` alone; for weight thresholding and the disparity filter, a spanning-tree
guarantee removes most of the infection-order penalty.

## Limits
- Two networks, one fully verified (hospital); no significance test is meaningful.
- workplace1 differs from the paper's data (above), so it is a robustness check, not a replication.
- Only SI spreading and eigenvector centrality are tested, as in the paper; one β target and one SI model.
- Verdict thresholds (half of the gap, noise band 0.02) are our choices, fixed before the runs; several outcomes sit just below the half line (34 to 42%).
- The λ-fit and the exploratory edge comparison above are one-off checks, not produced by `run.py`.

## Reproduce
```
python -m pip install -r requirements.txt
python -m pytest tests -q          # unit tests
python run.py                      # everything from scratch for both networks (about 30 min) and the dashboard data
python run.py --quick              # same code, 10 SI runs per seed (a few minutes), written to results_quick/
```
Outputs: `results/<network>/` (replication CSVs and `figures/`, `sensitivity/`, `diagnose/`, `extension/`), `results/summary/` (Table 1 check, pooled AUC tables,
sensitivity summaries, extension verdicts) and `ui/data.js` for the dashboard (open `ui/index.html`). `results/` is committed with the summary tables, figures and verdicts only; the raw sweeps are git-ignored and `python run.py` regenerates them.
