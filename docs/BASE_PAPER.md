# Base paper: Pereira, Costa and Rocha (2026)

**Distance backbones optimize spreading dynamics and centrality ranks in the sparsification of complex networks.**
arXiv:2603.14564v2, 19 March 2026, preprint, 33 pages (file: `papers/2603.14564v2.pdf`).

This document records what the paper does, with page numbers and short quotes, so that the project can replicate it
and extend it. Page numbers are PDF pages. I read the main text and the supplement in full from extracted text. The
extractor skipped some figure content, so **numbers that appear only in figures (the boxplots and the annotated
p-values) are not recorded here**. Check the figures yourself before quoting them.

## 1. What the paper is

The paper introduces a **new method**, the *Distance Backbone Synthesis (DBS)*, and shows that it, together with
semi-metric distortion (SmD, the SMDS of this project), preserves node-centrality ranks and SI spreading better than
older methods. The comparison of methods is the evidence for DBS. It is not the question of the paper.

- **DBS** (p. 4 to 5): the path length is generalised from a sum to a family of triangular distance norms
  `g_λ(x, y) = (x^λ + y^λ)^(1/λ)` (Dombi family, p. 5). Each edge gets a value `λ_ij`, the smallest λ at which it is in
  the distance backbone. Removing edges in increasing `λ_ij` gives a nested sequence of backbones from the full graph
  down to the ultrametric backbone, which here is the minimum spanning tree (p. 10, p. 15 to 17).
- **Distance backbone**: edges that satisfy the generalised triangle inequality. The metric backbone is the case
  `g = +` and keeps every shortest path (p. 3).
- **Semi-metric distortion**: `s_ij = d_ij / d^T_ij`, the edge length over the shortest indirect length (p. 3).

## 2. What it claims

| Claim | Where | Quote or detail |
| --- | --- | --- |
| DBS and SmD best preserve eigenvector-centrality rank | p. 6 | "distribution of AUC(ρC) is statistically higher when edge importance is measured by DBS or SmD than in comparison with wEffRes, DispFilt, and W" |
| For infection order, DispFilt is no better than W; the others are higher but statistically similar | p. 6 | "a more informative method such as DispFilt performs just as well as the simple W, while the other methods have higher performance albeit statistically similar" |
| DBS versus wEffRes on infection order is not significant | p. 6 | p = 0.056 |
| For the infection timescale, wEffRes, SmD and DBS beat W and DispFilt | p. 6 | "wEffRes, SmD, and DBS have a superior performance than DispFilt and W" |
| **Lost bridges are given as the reason W and DispFilt fail** | p. 6 | "W and DispFilt performed extremely poorly because they removed bridges very early on in the sparsification process, making the networks disconnected and hindering their spreading ability" |
| On more random networks (69 exhibit networks), SmD and DBS do much better | p. 6, 27 | "W and DispFilt are clearly unsuited to sparsify networks with such a random structure" |
| An optimal backbone lies between the product backbone and the metric backbone; `λ_D ≈ 0.3` gives large sparsification | p. 6 | "all three performance metrics are above 80% of their original value for λD ≈ 0.3" |
| Metric backbone and product backbone are the strict and lenient bounds | p. 6 to 8 | Found by a weighted multi-objective search (section 5 below) |

## 3. The experimental setup

### Networks (Table 1, p. 9)

12 social contact networks. Edges are the number of 20-second proximity intervals between two people (p. 8).

| Network | Nodes | Edges | Density | Metric backbone, adjusted X (%) |
| --- | --- | --- | --- | --- |
| kenya-households | 31 | 216 | 0.465 | 13.98 |
| hospital | 75 | 1139 | 0.410 | 13.43 |
| malawi-village | 84 | 346 | 0.099 | 9.51 |
| workplace1 | 92 | 755 | 0.180 | 20.63 |
| conference | 113 | 2196 | 0.347 | 9.40 |
| workplace2 | 217 | 4274 | 0.182 | 13.04 |
| french-primary-school | 242 | 8317 | 0.285 | 6.80 |
| french-high-school | 327 | 5818 | 0.109 | 5.04 |
| us-elementary-school | 339 | 16546 | 0.289 | 4.87 |
| us-middle-school | 591 | 56867 | 0.326 | 5.21 |
| copenhagen-university | 692 | 79530 | 0.333 | 4.72 |
| us-high-school | 788 | 118291 | 0.381 | 7.22 |

Supplement C.7 (p. 27): 69 daily networks from the Science Gallery exhibit in Dublin, with 135.94 ± 74.97 nodes,
595.78 ± 483.35 edges and density 0.079 ± 0.068. All networks are contact networks. The paper says its approach "is
easily generalizable to networks beyond the social domain" (p. 8) but tests none.

The concept note describes these networks as "75 to 327 nodes; 755 to 8,317 edges". Table 1 actually spans 31 to
788 nodes and 216 to 118,291 edges. The concept note's range matches only the six middle rows.

### Proximity and distance (p. 8)
`p_ij = r_ij / (r_ii + r_jj − r_ij)`, where `r_ij` is the number of intervals between i and j and `r_ii` the total
number of close interactions of i. Distance: `d_ij = 1/p_ij − 1` (p. 3, Eq. 1).

### Methods compared (Table 2, p. 9)

| Label | Method | Edge order |
| --- | --- | --- |
| W | Weight thresholding | remove increasing `p_ij` |
| DispFilt | Disparity filter, `α_ij = 1 − min(F_ki(p_ij), F_kj(p_ij))` | remove decreasing `α_ij` |
| wEffRes | Weighted effective resistance, `p_ij · R_ij` | remove increasing |
| SmD | Semi-metric distortion | remove decreasing `s_ij` |
| DBS | Distance backbone synthesis (new) | remove increasing `λ_ij` |

Supplement B and C.4 add semi-ultrametric distortion and an Aczél-Alsina variant of DBS (p. 18 to 24).
**Not included:** the noise-corrected filter and the high-salience skeleton.

Connectivity (p. 10): W, DispFilt and wEffRes "do not guarantee connectivity preservation". SmD guarantees it only
down to the metric backbone size. DBS keeps it down to the minimum spanning tree.

### Spreading simulation (p. 10)
- SI model. A single seed starts infected. Each step, an infected node infects each susceptible neighbour with
  probability `β · p_ij`. Runs end when all nodes are infected.
- "15% of the number of nodes chosen as seed nodes and 100 simulations for each seed node" (p. 10).
- Seed rule (supplement C.5, p. 25): "5% of the most central, 5% of the least central, and the other 5% of the median
  central, all with respect to their eigenvector centrality in the original network."
- **β is described only as "a global parameter that regulates the timescale of the spreading" (p. 10).** No
  calibration rule is given in the main text or the supplement.
- No run count above 100, no SIR, no SIS. SIS and SIR are listed as future work (p. 8).

### Measures (section 5.3, p. 10 to 12)

| Symbol | Meaning | Detail |
| --- | --- | --- |
| `ρ_C` | Spearman correlation of **eigenvector centrality** between original and sparsified network | Computed on the proximity matrix; deterministic |
| `ρ_t` | Spearman correlation of the mean infection time per node, over 100 runs | One value per seed; mean and standard deviation over seeds |
| `ξ⁻¹_0.5` | `t_0.5 (original) / t_0.5 (sparsified)`, the half-infection time ratio | Median over 100 runs per seed; "bounded and normalized, apart from small random fluctuations" |

### The sparsification axis and summary (p. 11 to 12)
- Size is the **adjusted relative size** `X = (|G'| − (n − 1)) / (|G| − (n − 1))`, so `X = 0` is the spanning tree and
  `X = 1` the full graph (Eq. 13). It is swept continuously. It is not a fixed grid of shares of edges.
- Each curve is summarised by the **area under the curve** from a limit `X̃` to 1 (Eq. 16). `X̃` is "the smallest
  sparsification which preserves connectivity"; for DBS it is fixed at the metric backbone size.
- Methods are compared with a **Mann-Whitney U test** on the 12 AUC values per method, testing whether DBS is larger
  (p. 5).

### Multi-objective search (section 5.4, p. 12)
The optimal DBS parameter maximises `w_C ρ_C + w_t ρ_t + w_ξ ξ⁻¹ + w_X (1 − X)` with weights summing to 1, over 16
weight scenarios (Table 3). This is a weighted-utility choice of a backbone, like the advisor in this project.

## 4. What the paper does not do

Each point says how I know. "Not found" means a search of the extracted text, not proof of absence.

| Gap | Evidence | Use for this project |
| --- | --- | --- |
| **The bridge explanation is stated, not measured.** No bridge-retention measure or test appears | Not found in the text; the quote on p. 6 is the whole argument | Test it: bridge-retention ratio on real networks, and a controlled manipulation on synthetic networks |
| **No community, shortest-path or structure measure.** It evaluates only centrality rank and SI dynamics, although the discussion says it wants backbones that are "dynamically and structurally faithful" (p. 7) | Section 3.2 and 5.3 | Add `F_comm`, `F_path`, `F_struct`, and test whether structure predicts function |
| **No random-removal baseline** | Not found in the text | Add R-U and R-W with skill scores. Hamann et al. found random deletion "performs surprisingly well" |
| **Only contact networks** | Section 5.1, Table 1, C.7 | Add one social or economic and one biological network |
| **Only SI** | p. 8 lists SIS and SIR as future work | Run SIR (already in E5) |
| **No NC or HSS** | Table 2 | Both are in the project |
| **It admits the network properties effect is not understood** | p. 8: "Despite still not being completely understood how each of those network properties affects sparsification methods..." | Directly supports the mechanism and descriptor analysis |
| **Statistics are unpaired** (Mann-Whitney on 12 AUC values) | p. 5 | Paired comparison per network and a cluster bootstrap |
| **No data or code statement** for the experiments | Searched for "available", "github", "zenodo" | Their `DISTANCECLOSURE` package is cited (p. 15, ref. 7) for the backbone computation only |

## 5. How this project's code compares

| Item | Paper | This project | Status |
| --- | --- | --- | --- |
| Proximity, distance | `p = r_ij / (r_ii + r_jj − r_ij)`, `d = 1/p − 1` | `p = w / (s_i + s_j − w)`, `d = 1/p − 1` | Same |
| SI seeds | 15% of nodes: top, median, bottom 5% by eigenvector centrality | Same | Same |
| Runs per seed | 100 | 100 (20 for random baselines) | Same |
| Order measure | Spearman of mean infection times | Same, clipped at 0 | Same |
| Timescale measure | Median half-time ratio | Same, clipped to [0, 1] | Same |
| Rank measure | Eigenvector centrality only | Mean of strength, eigenvector and betweenness | Ours is wider. Report the eigenvector part separately for replication |
| β | Not stated | Chosen so the median half-infection time is about 20 steps, runs capped at 10× | **Our own choice.** Justify it, and test other targets (E5 does 10 and 40) |
| Sparsification axis | Adjusted size `X`, continuous | Share of edges `q`, 8 values (5% to 70%) | **Different.** Convert, or add `X` |
| Summary | AUC of the curve above `X̃` | Per-budget scores | **Not implemented.** Needed for a like-for-like comparison |
| Test | Mann-Whitney on 12 AUC values | Paired Wilcoxon and cluster bootstrap | Different, deliberately |
| DispFilt | `α_ij = 1 − min(F_ki(p), F_kj(p))` | `1 − min(α_i, α_j)` with `α = (1 − p)^(k − 1)` | Equivalent form |
| wEffRes | `p_ij · R_ij`; Laplacian weights not stated in the text | Proximity as conductance | **Check** against Mercier et al. |
| SmD / SMDS | Remove decreasing `s_ij` | Keep smallest `s_ij` first | Same order, opposite direction of description |
| DBS | The paper's method | Not implemented | Optional. Their own text says SmD performs comparably (p. 8) |
| Networks | 12 contact networks, plus 69 exhibit networks | 1 (hospital) | See below |

**A check that already passes.** For hospital, Table 1 gives a metric backbone of 13.43% adjusted size. This
project's code gives 217 of 1,139 edges, which is exactly 13.43% adjusted size (`X = (217 − 74) / (1139 − 74)`).
That confirms the data, the proximity formula and the SMDS code against the paper.

**Network overlap.** By size, four of this project's earlier datasets match the paper: hospital (75 / 1139),
workplace 2013 (92 / 755, the paper's workplace1), workplace 2015 (217 / 4274, workplace2) and the primary school
(242 / 8317). The conference and high-school files used earlier had other sizes (403 / 9565 and 180 / 2220 against
the paper's 113 / 2196 and 327 / 5818), so they are different datasets. Only hospital is in the repository now.

## 6. Plan: replicate, then extend

**Stage A: replicate the paper's claims on the networks you share with it.**
1. Add workplace1, workplace2 and the primary school (same sizes as Table 1). Verify each network's metric backbone size
   against Table 1 before running anything else (the check above).
2. Add an AUC summary over the adjusted size `X` from `X̃` to 1, for `ρ_C`, `ρ_t` and `ξ⁻¹_0.5`.
3. Reproduce the qualitative results on those networks: SmD best on `ρ_C` and `ρ_t`; W and DispFilt similar on `ρ_t`;
   wEffRes, SmD and DBS above W and DispFilt on `ξ⁻¹_0.5`.
4. State plainly which results replicate. With 4 networks the Mann-Whitney test has little power, so also report the
   per-network differences.

**Stage B: extend, in order of value.**
1. Structure versus function: agreement and failure atlas, on the same methods and networks.
2. Bridge mechanism: bridge-retention ratio on real networks, then controlled bridge manipulation on synthetic networks.
3. Add communities and shortest paths as functions, and NC and HSS as methods.
4. Add random baselines and skill scores.
5. Add one non-contact network from each of two other domains.
6. Robustness under SIR and other settings.
7. The advisor, as a generalisation of the paper's weighted objective (section 5.4).

## 7. Open questions to settle

1. What β calibration did the authors use? Not in this paper. Check the Soriano-Paños et al. paper (my keyword search
   found nothing), or ask the authors.
2. Which Laplacian weights does their wEffRes use? Check Mercier et al.
3. Do you want to implement DBS? It is the paper's contribution, and costs an exhaustive search per edge (p. 8).
4. Does the guide accept a replicate-and-extend project built on a preprint? Cite the version used (v2, March 2026).

## 8. Limits of this reading

- Text extraction skipped some content in figures (the tool reported "Exceeded 5000 form XObject invocations"). Figure
  values and the annotated p-values in Fig. 2 are not recorded.
- Supplement figures 4 to 16 were not seen, only their captions.
- "Not found" statements come from keyword search and a full read of the extracted text, not from the figures.
