# Progress: base-paper replication (hospital network only)

Base paper: Pereira, Costa & Rocha (2026), *Distance backbones optimize spreading dynamics and centrality ranks in the
sparsification of complex networks*, arXiv:2603.14564v2 (`papers/2603.14564v2.pdf`).
Scope: replicate the paper on **one** network (hospital, 75 nodes, 1,139 edges). Improvements come after the replication.
Method: one step at a time, each step confirmed before the next.

Run all tests with `python -m pytest tests -q` (27 pass, about 35 s). Full pipeline: `python run.py` (cached) or `python run.py --force` (about 2.5 min).

| Step | What | Status |
| --- | --- | --- |
| 1 | Data ingestion | Done |
| 2 | Backbones: distortion, metric, ultrametric, DBS | Done |
| 3 | Baselines W, DispFilt, wEffRes and progressive sparsification | Done |
| 4 | Sweep over adjusted size X and eigenvector centrality `ρ_C` | Done |
| 5 | SI simulation: `ρ_t` and `ξ⁻¹_0.5` | Done |
| 6 | AUC summary, plots, optimal-λ search (Eq. 17), `run.py` | Done |

---

## Step 1: Data ingestion

**Code:** `backbone/data.py`, `tests/test_data.py`.
**Input:** `data/detailed_list_of_contacts_Hospital.dat` (SocioPatterns hospital ward, Vanhems et al. 2013). Tab-separated,
no header, columns `t, i, j, Ai, Aj`, 32,424 rows, 20 s steps, about 4 days.

**What it does**
- `load_contacts` reads the file and drops self-loops and repeated rows (none found).
- `edge_table` counts contact intervals per pair as `r_ij`, then computes proximity
  `p_ij = r_ij / (r_ii + r_jj − r_ij)` and distance `d_ij = 1/p_ij − 1` (paper Eq. 12 and Eq. 1).
- `build_graph` / `load_graph` return a networkx graph keyed by the original person IDs; edges carry `r, proximity, distance`.
- `python -m backbone.data` writes `data/processed/hospital_edges.csv`.

**Check against the paper (Table 1, hospital row)**

| Quantity | Paper | Ours |
| --- | --- | --- |
| Nodes | 75 | 75 |
| Edges | 1,139 | 1,139 |
| Metric backbone, adjusted size X | 13.43% | 13.427% (217 edges) |

Proximities range from 0.00016 to 0.476. The graph is connected. Node IDs are not 0..74 (e.g. 1157, 1232).

---

## Step 2: Distance backbones

**Code:** `backbone/backbones.py`, `tests/test_backbones.py`.

**What it does**
- `semi_metric_distortion`: `s_ij = d_ij / d^{T,sum}_ij` (1 means a metric edge).
- `ultrametric_mask`: minimax-path closure (vectorised Floyd-Warshall); an edge is ultrametric if `d_ij` equals the minimax distance.
- `dombi_lambda`: DBS importance `λ_ij` for each edge.
  - Dombi path length `(Σ d^λ)^(1/λ)` is minimised by a shortest path on weights `d^λ`.
  - An edge is in `B_λ` if no indirect path is shorter than the edge.
  - Backbones are nested, so `λ_ij` is found by bisection in log λ (range 1e-3 to 200, 40 steps).
  - Ultrametric edges get `λ = inf`.
- `edge_table(G)` returns one row per edge: `u, v, proximity, distance, distortion, lam`.
- Helpers: `metric_edges`, `ultrametric_edges`, `backbone_at_lambda(table, lam)`, `adjusted_size` (Eq. 13).

**Convention:** `λ_ij` is the largest λ at which the edge is still in the backbone. The paper's wording about which edges
are kept is inconsistent; I followed its nestedness result (bigger λ gives a smaller backbone, λ=1 is the metric backbone,
λ→∞ is the ultrametric one). The sparsified graph at threshold `λ_D` keeps edges with `λ_ij ≥ λ_D`, and DBS removes in
increasing `λ_ij`.

**Results on hospital**
- 217 metric edges (matches Table 1) and 74 ultrametric edges = n−1 (the paper says these networks have a unique MST).
- `λ ≥ 1` holds exactly for the metric edges.
- Finite `λ_ij`: median 0.376, range 0.13 to 16.3. The paper reports a log-normal with mode about 0.3 ± 0.1.
- Edges kept by λ: 1,124 at 0.15; 766 at 0.3; 695 at 1/3; 460 at 0.5; 217 at 1; 130 at 2.
- The tests include a hand-solved triangle (`λ = ln 2 / ln 3`) and a check that the bisection agrees with a direct `B_λ` computation.

**Derived file:** `data/processed/hospital_backbone_scores.csv` holds the 1,139-row edge table. It was written by a
one-off command, not by a module. It can be regenerated with `edge_table(G)`.

**Open point:** at λ = 1/3 (the paper's optimal `g`, `(∛x + ∛y)³`) we remove 39% of the edges, while the abstract says
"more than half". The abstract covers all 12 networks, so this may be network-dependent. Not yet resolved.

**Speed:** the exhaustive λ search takes about 5 s here. The paper notes this cost as a drawback for large networks.

---

## Step 3: Baselines and progressive sparsification

**Code:** `backbone/sparsify.py`, `tests/test_sparsify.py`.

**What it does.** Each of the five methods in Table 2 becomes an importance score (higher means keep longer), so removal is
always lowest importance first.

| Method | Importance | Notes |
| --- | --- | --- |
| W | `p_ij` | weight thresholding |
| DispFilt | `−α_ij` | `α = 1 − min(F_ki(p), F_kj(p))`, `F_k(p) = 1 − (1−p)^(k−1)`; a degree-1 node has `F = 1` |
| wEffRes | `p_ij · R_ij` | `R` from the Laplacian pseudo-inverse, proximity as conductance |
| SmD | `−s_ij` | reuses `distortion` from step 2 |
| DBS | `λ_ij` | reuses `lam` from step 2 |

- `ranking(table, method, seed)` orders edges most to least important. Ties are broken by a seeded random order, which
  matters for SmD (217 edges tied at `s = 1`).
- `sparsify(table, method, X)` keeps the top edges for a target adjusted size X (`edges_to_keep` inverts Eq. 13). All methods
  keep the same number of edges at the same X.
- `to_graph` builds a graph on all original nodes (isolated nodes stay), so centrality ranks stay comparable.
  `is_connected` and `realised_X` are helpers.

**Tests:** toy-graph orderings, a hand-computed disparity filter (including the degree-1 case), wEffRes properties (scores
sum to n−1, a bridge scores exactly 1), equal edge counts across methods, and connectivity on hospital.

**Connected components on hospital (1 = connected)**

| X | W | DispFilt | wEffRes | SmD | DBS |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 22 | 34 | 9 | 13 | 1 |
| 0.05 | 11 | 25 | 2 | 4 | 1 |
| 0.134 | 7 | 15 | 1 | 1 | 1 |
| 0.4 | 2 | 5 | 1 | 1 | 1 |
| 0.7 | 1 | 3 | 1 | 1 | 1 |

This agrees qualitatively with the paper: W and DispFilt disconnect the graph early (the paper blames lost bridges), SmD
stays connected down to the metric-backbone size, and DBS stays connected down to the spanning tree.

**Decisions to check**
1. **DispFilt `p`:** I used the proximity `p_ij` directly, following the paper's formula. Serrano's original filter
   normalises weights per node (`w/strength`). This may explain why DispFilt is still disconnected at X = 0.7. Decide
   whether to switch.
2. **wEffRes conductances:** the paper does not state them. I used proximity (the standard choice).
3. **Tie-breaking:** random with a fixed seed. Below X = 0.134, SmD results depend on the seed.

---

## Step 4: Sweep over X and eigenvector centrality `ρ_C`

**Code:** `backbone/sweep.py`, `tests/test_sweep.py`.

**What it does**
- `Network(table)` holds the edge table plus index arrays, so every subgraph is a cheap slice of the edge list.
  `adjacency(keep)` builds the dense proximity matrix of a subgraph (removed edges get 0, kept edges keep their proximity).
- `eigenvector_centrality(A)` takes the leading eigenvector of the symmetric proximity matrix (`numpy.linalg.eigh`),
  takes absolute values, and sets entries below 1e-12 to 0. On a disconnected subgraph the leading eigenvector sits on one
  component and the rest are numerically zero, so they are tied at 0 (Spearman uses average ranks).
- `rho_C(c_full, c_sub)` is the Spearman rank correlation (Eq. 14 and Sec. 5.3.1).
- `sweep(net)` ranks each method's edges once, then keeps the top `k` edges for every `k` from `n−1` to `|G|`. That is
  **X swept one edge at a time** (1,066 points per method, 5,330 rows in total), the finest resolution possible.
  Each row has `method, n_edges, X, rho_C, components`.
- `python -m backbone.sweep` writes `data/processed/sweep_rhoC.csv` (about 40 s).

**Tests:** our eigenvector centrality equals networkx's `eigenvector_centrality_numpy` on a toy graph and on hospital;
`ρ_C` edge cases (identical, reversed, monotone transform); sweep endpoints (`X = 0` has `n−1` edges, `X = 1` is the full
graph and gives `ρ_C = 1` for every method); DBS stays connected at every X on hospital.

**First results on hospital (`ρ_C` at selected X, seed 0)**

| X | DBS | DispFilt | SmD | W | wEffRes |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 0.057 | 0.279 | 0.461 | 0.318 | 0.232 |
| 0.050 | 0.634 | 0.318 | 0.150 | 0.599 | 0.395 |
| 0.134 | 0.669 | 0.798 | 0.669 | 0.799 | 0.622 |
| 0.200 | 0.692 | 0.880 | 0.700 | 0.850 | 0.663 |
| 0.400 | 0.895 | 0.943 | 0.906 | 0.933 | 0.876 |
| 0.600 | 0.976 | 0.990 | 0.980 | 0.982 | 0.963 |
| 0.800 | 0.998 | 0.999 | 0.998 | 0.998 | 0.996 |

A rough area under the curve from X = 0.134 to 1 (plain trapezoid, the same lower limit for every method, **not** the paper's
rule): DBS 0.911, SmD 0.919, DispFilt 0.954, W 0.946, wEffRes 0.896.

**This does not reproduce the paper's ranking on this network.** The paper says DBS and SmD preserve `ρ_C` best
(Fig. 2, left), but on hospital W and DispFilt score higher above X = 0.134. I have not tuned anything to change this.
Possible reasons, none tested yet:
1. The claim is about the distribution over 12 networks; hospital may be one of the exceptions. With one network we cannot
   tell.
2. Eigenvector centrality of the proximity matrix is dominated by the strongest edges, which W and DispFilt keep.
3. The paper's AUC uses a method-specific lower limit `X̃` (the smallest X that stays connected; `|B_m|` for DBS). W and
   DispFilt break connectivity earlier, so their `X̃` is larger. That changes the comparison and is part of step 6.
4. Our DispFilt `p` choice (see step 3) and the random tie-breaking for SmD could matter. SmD is very noisy below
   `X = 0.134` (0.461 at X = 0, 0.150 at X = 0.05).

Below the metric backbone size, DBS is connected but its `ρ_C` drops sharply near `X = 0`
(0.057 at the spanning tree). A tree keeps few strong edges, so the eigenvector is not preserved.

---

## Step 5: SI spreading, `ρ_t` and `ξ⁻¹_0.5`

**Code:** `backbone/spread.py`, `tests/test_spread.py`.

**Model (paper Sec. 5.3.2).** One seed starts infected. Each step, every infected node infects each susceptible neighbour with
probability `β · p_ij` (synchronous update). Infection times are recorded (seed = 0). Implemented vectorised over the 100 runs:
the chance that a susceptible node escapes all infected neighbours is `exp(I @ log(1 − βP))`.

**What it does**
- `choose_seeds`: 15% of nodes (11 of 75) from the original eigenvector-centrality ranking: 4 top, 3 median, 4 bottom
  (the paper's 5% / 5% / 5%, Supplement C.5).
- `simulate(A, seed, beta, runs, rng)`: infection times of shape (runs, n).
- `half_time`: per-run `t_0.5`, the first step at which `ceil(n/2)` nodes are infected.
- `calibrate_beta`: bisection so that the mean over seeds of the median `t_0.5` on the full graph equals a target (20 steps).
- `reference_runs`: original-network runs, computed once and reused for every sparsified graph.
- `spreading_fidelity`: per seed, `ρ_t` (Spearman of the mean infection time per node over 100 runs) and `ξ⁻¹_0.5`
  (median over runs of `t_0.5(original) / t_0.5(sparsified)`).
- `si_sweep`: mean and standard deviation over seeds, for each method and 22 sparsification levels
  (X = 0, 0.05, ..., 1 plus the metric-backbone size).
  `python -m backbone.spread` writes `data/processed/sweep_si.csv` (110 rows; about 1 minute).

**Decisions where the paper is silent**
1. **β:** the paper only says it "regulates the timescale". I calibrate it so the full graph has a median half-infection time of
   20 steps. This gives **β = 1.51** on hospital (largest valid β is `1/p_max = 2.10`). The target of 20 is my choice and
   needs a sensitivity check (10 and 40 are natural alternatives).
2. **Never-infected nodes** (other component) get time `T_MAX = 1000`. A run stops once the seed's component is fully
   infected or at `T_MAX`. If half of the nodes are never reached, `t_0.5 = T_MAX`, so `ξ⁻¹` is near 0.
3. **`ρ_t` excludes the seed** (time 0 in every network, no information). If one side is constant (the seed reaches
   nothing), `ρ_t` is set to 0.
4. **`ξ⁻¹`** is formed per run (run r of the original over run r of the sparsified graph), then the median over the 100 runs.
   The paper says "median over 100 simulations" without saying how runs are paired.
5. **Independent random streams** for the original and each sparsified graph (seeded, so the results are reproducible).

**Tests:** seed groups; deterministic infection time on a path graph (`βp = 1` gives hop distance); unreachable nodes get `T_MAX`;
`half_time`; identical graph with the same random stream gives `ρ_t = ξ⁻¹ = 1`; β calibration hits its target on a ring;
a hospital smoke test (calibrated half-time near 20, and the full graph against itself gives `ρ_t > 0.9`, `ξ⁻¹` near 1).

**Results on hospital (mean over 11 seeds)**

`ρ_t`:

| X | DBS | DispFilt | SmD | W | wEffRes |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 0.261 | 0.333 | 0.259 | 0.351 | 0.099 |
| 0.050 | 0.757 | 0.458 | 0.555 | 0.563 | 0.460 |
| 0.134 | 0.895 | 0.620 | 0.902 | 0.628 | 0.816 |
| 0.300 | 0.971 | 0.655 | 0.963 | 0.742 | 0.951 |
| 0.500 | 0.976 | 0.748 | 0.975 | 0.935 | 0.974 |
| 0.800 | 0.979 | 0.950 | 0.980 | 0.978 | 0.978 |
| 1.000 | 0.977 | 0.980 | 0.979 | 0.981 | 0.980 |

`ξ⁻¹_0.5`:

| X | DBS | DispFilt | SmD | W | wEffRes |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 0.135 | 0.020 | 0.053 | 0.061 | 0.109 |
| 0.050 | 0.317 | 0.186 | 0.220 | 0.223 | 0.300 |
| 0.134 | 0.504 | 0.333 | 0.502 | 0.362 | 0.529 |
| 0.300 | 0.739 | 0.511 | 0.753 | 0.590 | 0.759 |
| 0.500 | 0.849 | 0.638 | 0.866 | 0.767 | 0.872 |
| 0.800 | 0.968 | 0.840 | 0.973 | 0.943 | 0.933 |
| 1.000 | 1.001 | 0.981 | 0.978 | 1.003 | 0.996 |

**Reading the results**
- Qualitatively this **matches the paper**: DBS and SmD preserve infection order and timescale best, W and DispFilt are
  clearly worse, and wEffRes is competitive on the timescale (paper, Sec. 3.2). W and DispFilt lose connectivity early
  (step 3), which is the paper's explanation.
- This differs from `ρ_C` in step 4, where W and DispFilt looked better on hospital. So on this one network, the centrality
  claim and the spreading claim disagree.
- **Noise floor:** at X = 1 the `ρ_t` values are 0.98, not 1, because the original and sparsified runs are independent
  with 100 runs each. Differences of about 0.01 between methods are not meaningful. `ξ⁻¹` values slightly above 1 (e.g. 1.010)
  are also noise.
- At X = 0 (spanning tree) everything is poor. DBS is still connected there, so it has the best `ξ⁻¹` (0.135), but
  `ρ_t` is only 0.26.
- The standard deviation over seeds is large where graphs are disconnected (for example `ρ_t` std 0.40 for W at the metric
  backbone size), because the result depends on which component the seed is in.

---

## Step 6: AUC summary, optimal λ, figures and `run.py`

**Code:** `backbone/summary.py`, `backbone/plots.py`, `run.py`, `tests/test_summary.py`.
`backbone/spread.py` gained `si_setup` (seeds, β and reference runs, so the λ study reuses them).

**What it does**
- `connectivity_limit` / `x_tilde`: the AUC lower limit `X̃`, taken from the full-resolution `ρ_C` sweep. The connectivity
  limit is the smallest X from which a method stays connected for every larger X. `X̃ = max(connectivity limit, metric
  backbone size)`. The paper fixes `X̃ = |B_m|` for DBS and uses the connectivity limit otherwise; I apply the same floor to
  all methods, which is my reading of it.
- `auc`: trapezoid rule over `[X̃, 1]`, divided by `1 − X̃`, interpolating the value at `X̃` (Eq. 16).
- `auc_table`: AUC of `ρ_C`, `ρ_t` and `ξ⁻¹_0.5` per method. Computed two ways: each method's own `X̃` (the paper's rule), and
  the same limit `X̃ = |B_m|` for everybody (a sensitivity check).
- `lambda_sweep`: metrics of the DBS backbone `B_λ` for 51 values of λ (log-spaced from 0.1 to 20, plus 0.3, 1/3, 1 and
  λ = ∞ = the spanning tree). Same seeds, β and reference runs as step 5.
- `optimal_lambda`: for each scenario of Table 3, the λ maximising `w_C ρ_C + w_t ρ_t + w_ξ ξ⁻¹ + w_X (1 − X)` (Eq. 17).
  Table 3 has **17** rows (the paper text says 16 scenarios elsewhere); all 17 are used.
- `plots.py`: four PNGs in `figures/`. First five slots of a validated categorical palette (lightness, chroma, CVD and
  normal-vision checks pass). Aqua, yellow and magenta are below 3:1 contrast on the light surface, so every series also has
  its own marker and line style, and each chart has a legend. Light surface only.
  - `fidelity_curves.png`: `ρ_C`, `ρ_t`, `ξ⁻¹_0.5` against X (cf. paper Fig. 4).
  - `auc_xtilde.png` and `auc_fixed_limit.png`: AUC bars (cf. paper Fig. 2, without boxplots).
  - `dbs_lambda.png`: X and the three fidelities against λ, and the optimal λ per scenario (cf. Fig. 1 and Fig. 3).
- `run.py`: data, backbones, both sweeps, AUC, λ study and figures in one command. Results are cached as CSV in
  `data/processed/` (`auc_xtilde.csv`, `auc_fixed_limit.csv`, `lambda_sweep.csv`, `optimal_lambda.csv`, `sweep_si.csv` now on a
  41-point X grid plus the metric size, and the earlier files).

**Tests:** all 17 scenario weight vectors sum to 1; AUC on known curves (constant, `y = X`, interpolated limit); connectivity
limit and the metric-size floor; optimal λ follows the weights on a hand-made table.

**AUC on hospital**

Each method's own `X̃` (paper rule):

| Method | X̃ | AUC `ρ_C` | AUC `ρ_t` | AUC `ξ⁻¹` |
| --- | --- | --- | --- | --- |
| W | 0.474 | 0.990 | 0.974 | 0.911 |
| DispFilt | 0.800 | 1.000 | 0.974 | 0.905 |
| wEffRes | 0.134 | 0.896 | 0.957 | 0.874 |
| SmD | 0.134 | 0.920 | 0.970 | 0.869 |
| DBS | 0.134 | 0.911 | 0.971 | 0.863 |

Same limit `X̃ = 0.134` for all methods:

| Method | AUC `ρ_C` | AUC `ρ_t` | AUC `ξ⁻¹` |
| --- | --- | --- | --- |
| W | 0.946 | 0.898 | 0.783 |
| DispFilt | 0.954 | 0.777 | 0.669 |
| wEffRes | 0.896 | 0.957 | 0.874 |
| SmD | 0.920 | 0.970 | 0.869 |
| DBS | 0.911 | 0.971 | 0.863 |

**How to read this honestly**
- With the **paper's own `X̃` rule**, W and DispFilt have the *highest* AUC, because their `X̃` is large (0.47 and 0.80): they are only
  evaluated on the range where they are still connected and easy. The AUC does not penalise them for the sparse range they cannot
  reach. With one network and this rule, the paper's claim that DBS and SmD win is **not** reproduced.
- With the **same limit for all methods**, the paper's pattern appears for the spreading measures: DBS and SmD are best on
  `ρ_t` (0.971, 0.970) and wEffRes, SmD and DBS are well ahead of W and DispFilt on `ξ⁻¹` (0.87 against 0.78 and 0.67).
  On `ρ_C`, W and DispFilt are still ahead (0.946 and 0.954 against 0.911 for DBS), as in step 4.
- DBS and SmD are very close on every measure, which the paper also says (Sec. 4).
- The paper's Mann-Whitney test across 12 networks cannot be done with one network.

**Optimal λ_D (Eq. 17), hospital**
- Balanced weights: **λ = 0.346** (X = 0.558; `ρ_C` 0.965, `ρ_t` 0.977, `ξ⁻¹` 0.903). The paper reports about 0.3 for a
  balanced optimum and a log-normal mode of 0.3 ± 0.1, so this agrees.
- Most scenarios give λ between 0.22 and 0.6: Slight Preference for each of `ξ⁻¹`, `ρ_t` and `ρ_C` gives 0.33 to 0.35. Strong
  Preference for `ρ_C`: 0.30. Mainly `ρ_C`: 0.22.
- Scenarios that favour size give much larger λ (Strong Preference for X: 2.6; Mainly X: 11.4). Scenarios with no weight on size
  (Balanced Except X, Mainly `ξ⁻¹`) pick the smallest λ in the grid (0.1, the full graph).
- At λ = 1/3 (the paper's optimal `g`): 695 edges, X = 0.583, `ρ_C` 0.971, `ρ_t` 0.976, `ξ⁻¹` 0.916. All three are above 80%, as the
  paper reports for λ ≈ 0.3 (its Supplementary Fig. 16).
- At the metric backbone (λ = 1): X = 0.134, `ρ_C` 0.669, `ρ_t` 0.901, `ξ⁻¹` 0.495. The metric backbone is stricter than λ ≈ 0.3, as
  the paper says.
- The optimum is read off a coarse λ grid and a single stochastic simulation per point (100 runs per seed), so differences of a few
  hundredths in λ are not meaningful. The curves in `dbs_lambda.png` are not smooth for the same reason.

**Open points**
1. The `X̃` rule is my reading of the paper (floor at the metric size for every method). The ranking in the AUC depends on it, as
   shown above. Worth checking against Supplementary Fig. 9 and the paper's Fig. 2 values, which the text extraction could not
   read.
2. The earlier "39% removed at λ = 1/3" versus the abstract's "more than half" is still unresolved. At λ = 1/3 X = 0.583 counts the
   spanning tree; 695 of 1,139 edges is 61% kept.
3. The β target (20 steps) and DispFilt's `p` choice are still unchecked sensitivities.

---

## Comparison with the paper's figures (checked by eye)

I rendered the paper's Fig. 2 (page 5) and Fig. 4 (page 11) and compared. Values are **read off the boxplots, so about ±0.03**.
Fig. 2 shows the distribution over the 12 networks (box = quartiles, line = median). Hospital is one dot among 12 and cannot be
picked out. Fig. 4 is the French high school, not hospital. So this compares our one network with the paper's medians.

| AUC | Paper median W | DispFilt | wEffRes | SmD | DBS | Ours (own `X̃`): W | DispFilt | wEffRes | SmD | DBS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ρ_C` | 0.68 | 0.75 | 0.88 | 0.92 | 0.93 | 0.990 | 1.000 | 0.896 | 0.920 | 0.911 |
| `ρ_t` | 0.70 | 0.80 | 0.94 | 0.95 | 0.96 | 0.974 | 0.974 | 0.957 | 0.970 | 0.971 |
| `ξ⁻¹` | 0.67 | 0.73 | 0.87 | 0.89 | 0.88 | 0.911 | 0.905 | 0.874 | 0.869 | 0.863 |

**What agrees**
- **wEffRes, SmD and DBS agree with the paper within about 0.02 on all three measures**, which supports the implementation of the
  backbones, the sweep, `ρ_C`, the SI model and the AUC.
- The three are close to each other in the paper too, and DBS is not clearly above SmD there either.
- Fig. 4 `ξ⁻¹` curves: at the spanning tree DBS is 0.12 in the paper and 0.135 here; at X = 0.2 about 0.65 in the paper and 0.62 here.
  The shape (steep rise, saturating near X = 0.5) is the same.
- Fig. 4 `ρ_t`: rises to about 0.93 to 0.95 by X = 0.1 to 0.2 and then sits near 0.97 to 0.98, close to the noise floor we see.
  Our curve rises somewhat later (0.83 at X = 0.1, 0.95 at 0.2); the networks differ.
- The paper marks `X̃` with a dot on each curve and starts each method's AUC there, as I did.

**What does not agree**
- **W and DispFilt.** The paper's medians are much lower (0.67 to 0.80) than ours with each method's own `X̃` (0.90 to 1.00). With the same
  limit for everyone (0.134), our `ρ_t` and `ξ⁻¹` for W and DispFilt (0.90 / 0.78 and 0.78 / 0.67) fall in the paper's range. Our `ρ_C`
  for them stays high (0.95) while the paper's medians are 0.68 and 0.75.
- In Fig. 4 (high school) DBS has the highest `ρ_C` curve and W and DispFilt the lowest. On hospital we see the reverse.
- The paper's W and DispFilt boxes are wide (about 0.4 to 0.95) with outliers near 0.1 to 0.15, so large network-to-network variation is
  expected. Hospital may sit in the upper part of that range, but I cannot confirm it from the figure.

**Conclusion.** The top three methods reproduce the paper's numbers closely. W and DispFilt do not, mainly on `ρ_C`, and the paper's
exact AUC limit for them is unclear. The next test is a second network (workplace1 or the primary school) to see whether hospital is
simply a high point for W and DispFilt.

(`pip install pymupdf` was needed to render the PDF; it was installed into the Python environment on this machine's PATH, not into the project.)

---

## Still to do
- **Sensitivity checks** on the replication: β target (10, 20, 40), DispFilt with per-node normalised weights, other random seeds
  for tie-breaking, more SI runs (reduces the noise floor seen at X = 1).
- **Compare against the paper's figures** (Fig. 2, Fig. 4 and the supplementary figures) by eye. The text extraction skipped the
  figure values.
- **More networks.** The paper's statistics need 12; the data are public SocioPatterns and related contact datasets.
- **Improvements** (the point of the project) once the replication is accepted.
- **Limitation:** one network, so the Mann-Whitney test and any claim about distributions over networks cannot be reproduced.
