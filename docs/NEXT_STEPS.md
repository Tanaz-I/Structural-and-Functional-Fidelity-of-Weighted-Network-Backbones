# Next steps

A step-by-step guide for finishing the project as **"replicate Pereira et al. (2026), then extend it"**.
Work through the steps in order and tick the boxes. Each step says why it matters, what to do (commands and the files
to change), and how you know it is done.

Read first: [DOCUMENTATION.md](DOCUMENTATION.md) (what exists) and [BASE_PAPER.md](BASE_PAPER.md) (what the base
paper does, with page quotes). The full list of possible extensions, with the reason, hypothesis and
code for each, is in [EXTENSIONS.md](EXTENSIONS.md) (it is organised by extension, X1 to X14, not by the step numbers below).

## The project in one paragraph

Pereira et al. introduce a new backbone method and show that path-based methods (SMDS and their DBS) preserve
eigenvector-centrality rank and SI spreading on contact networks better than weight thresholding, the disparity
filter and effective resistance. They explain the failures of the weaker methods by lost bridges but never measure
that, they use only contact networks and SI, and they have no random baseline and no community or path measure.
This project replicates their result on networks you share with them (**Stage A**), then adds what they lack
(**Stage B**): does structural fidelity predict functional fidelity, is the bridge explanation true when tested by
manipulation, and does it hold beyond contact networks.

## Work on one dataset first (hospital), then add more

The project currently has **one real dataset: hospital** (75 nodes, 1,139 edges). Work in two parts. Part 1 needs
nothing but hospital. Part 2 adds datasets, one at a time, and is only needed for claims about networks in general.

**What one dataset can and cannot give you**

| With hospital alone you can | You cannot, because the test pools across networks |
| --- | --- |
| Check the pipeline, the metric-backbone size (13.43%, matches the paper) and every score | Say whether a result holds for contact networks in general |
| Compare the six methods on the paper's three measures, per budget and as an AUC | Run the paper's Mann-Whitney test, or any cluster bootstrap over networks |
| See structure and function disagree for a method and budget (the failure atlas, descriptively) | Get a verdict on H1, H2, RQ1 or RQ4 (they stay "inconclusive" below 5 networks) |
| Compare against random baselines (skill scores) | Test dependence on network type, or regress on network properties |
| Test the bridge explanation by swapping bridge edges at a fixed budget (Step 6.2) | Test the causal factors of the synthetic design (that is a separate dataset) |
| Run the robustness variants (SIR, β target, proximity map and others) | Validate the advisor on unseen networks |
| Implement DBS and check it against the paper's hospital numbers (Step 8.1) | |

Hospital is also the best first network: it is the smallest of the paper's contact networks, so every run is fast, and
its metric backbone is already confirmed against the paper.

## Order of work

```
PART 1: hospital only
  Step 0  decide and finish reading            (no data)
  Step 1  fix known problems                   (no data)
  Step 3  replication code                     (tested on hospital)
  Step 4A replicate on hospital                (per-network comparison with the paper)
  Step 6.2 bridge swap on hospital             (pilot of the extension)
  Step 7.2-7.4 random baselines, communities and paths, robustness on hospital
  Step 8.1 optional: DBS checked against the paper's hospital numbers
  ---> Gate A1: a complete single-network study, written up

PART 2: add datasets, one at a time
  Step 2  add the paper's other contact networks (workplace1, workplace2, primary school)
  Step 4B pooled replication (needs 4 networks)    ---> Gate A
  Step 5  structure vs function (needs 5+ networks)
  Step 6.1, 6.3, 6.4 bridge and descriptor analyses across networks
  Step 7.1 non-contact networks
  Step 9  synthetic sweep, freeze, write the paper   ---> Gates B and C
```

Gates: **A1** single-network study done (Part 1). **A** pooled replication report done. **B** structure-versus-function
and bridge results on 5+ networks. **C** protocol frozen and final run reproduces. Stopping after any gate still leaves
a defensible project.

Each step below is tagged **[hospital]** if it works on one dataset, or **[more datasets]** if it needs several.

---

## Step 0. Decide and finish reading

- [ ] **0.1 Read the other four papers in full** (`papers/`): Soriano-Paños 2025, Correia 2023, Hamann 2016, Yan 2018.
  A first pass is below. For each, answer: *did they test whether structural fidelity predicts functional fidelity
  across methods, networks and budgets?* Record page quotes in a file like BASE_PAPER.md.
- [ ] **0.2 Settle the labels** for hypotheses and research questions (see 1.1). Suggested: use the concept note's H1 to
  H3 and RQ1 to RQ4 exactly as written, and rename the code's tests to match.
- [ ] **0.3 Decide the headline functions.** The concept note puts spreading and communities first. Suggested: report
  those two as the main results and paths and rankings as supporting evidence.
- [ ] **0.4 Confirm with your guide** that a replicate-and-extend project built on a March 2026 preprint is acceptable.
  Cite the version you use (arXiv:2603.14564v2).

**First-pass reading of the other four papers** (abstracts, conclusions and keyword searches only; not a full read):

| Paper | What it does | Consequence |
| --- | --- | --- |
| Hamann et al. 2016 | Compares sparsification methods on social networks and synthetic planted-community networks, on diameter, components, communities, centralities and epidemics. Different methods suit different properties; random deletion "performs surprisingly well"; preserved communities are "not necessarily the same as the one the Louvain algorithm finds" | "No single method wins on every function" is **already known**. Keep random baselines |
| Yan et al. 2018 | Local and global features are quickly lost under weight thresholding, but **group structure is robust**, because large weights sit on edges of high-degree nodes | The concept note says Yan found properties "disrupted early". In the paper that sentence describes an earlier brain-network study. Correct the note. Expect W to do well on communities |
| Soriano-Paños et al. 2025 | Introduces semi-metric distortion sparsification; recovers epidemic dynamics; keeps all shortest paths. No discussion of bridges or communities | Source of SMDS. No support for the community claim in H2 |
| Correia et al. 2023 | Nine contact networks. The metric backbone is small (6% to 20% of edges for eight, 49% for one) and preserves the distribution of shortest paths, including between and within communities | Source for H2 |

---

## Step 1. Fix known problems (do before adding data)

Each was found in the code and checked. Do them in this order; each is small.

- [ ] **1.1 Labels disagree.** The concept note's RQ1 is "how often structure passes while function fails", RQ3 "which
  properties explain it", RQ4 "can a method be recommended". The code uses RQ1 for rank agreement, RQ3 for weight spread
  and coupling, RQ4 for "no single method wins". `protocol.yaml` still lists the design paper's H1 to H5.
  *Do:* rename the verdict ids and statements once (decision 0.2) in `backbone/report.py`, `backbone/studies.py`,
  `protocol.yaml`, `ui/app.js` and the docs. *Done when* the report, protocol, UI and docs show the same labels.
- [ ] **1.2 Inconclusive reads as a finding.** With one network the report says "mixed evidence" and "SMDS beats the
  weight-based filters on some, not all, measures". The real reason is too few networks.
  *Do:* in `report.py` (`_v`, `overall_summary`) separate "inconclusive: fewer than 5 networks" from "inconclusive: mixed
  evidence". *Done when* a one-network run says "not enough networks" everywhere.
- [ ] **1.3 `analyze` prints "SUPPORTED (4/4)" with one network**, while `conclusions` says inconclusive.
  *Do:* apply the same minimum-networks rule in `studies.run_e1_e2`.
- [ ] **1.4 A missing `--domain` silently drops a network from the contact-only tests (H1, H2).**
  *Do:* make `--domain` required in `run.py` (`prepare-data`) and in `network.prepare_dataset`.
- [ ] **1.5 A partial synthetic run crashes the analysis.** `mechanisms --synthetic` raises `FileNotFoundError` for
  `syn40.parquet`. *Do:* in `analysis.load_results`, skip missing files and report "n of 81 used".
- [ ] **1.6 UI text is corrupted.** `ui/app.js` has double-encoded characters in 7 places (verdict icons, "·"
  separators). *Do:* re-save the file as UTF-8 with the intended characters. *Done when* `python run.py ui-check`
  screenshots show them correctly.
- [ ] **1.7 Ties favour the alphabet.** `analysis.winners()` gives a tied cell to the first method alphabetically,
  inflating the "different winners" share. *Do:* count a cell as "different" only when the sets of winners do not overlap.
- [ ] **1.8 Add tests for the generic dataset code** (`prepare_dataset`, `load_edgelist`, `read_contacts`, the unknown-name
  error) with a tiny temporary file, in `tests/test_core.py`.
- [ ] **1.9 Stale pointers.** `backbone/export.py` says its data contract is in `docs/PLAN.md` (now a stub). `export`
  reads the atlas from the label `real_dev` while `analyze --networks X` writes to `custom`. *Do:* fix the docstring and
  make the label explicit.
- [ ] **1.10 Add "advisor not validated on unseen networks" to the limits** in `report.overall_summary`.
- [ ] **1.11 Commit.** Only the initial commit exists. Commit in steps: library, tests, UI, docs, manifest.
  Keep `data/raw/` and `results/` ignored. *Done when* `git status` is clean.

*Gate for Step 1:* `python -m pytest -q` passes (37 now, more after 1.8).

---

## Step 2. Data for the replication  **[more datasets, Part 2]**

**Skip this step during Part 1.** Hospital is already prepared and checked (n = 75, m = 1139, metric backbone 13.43%,
equal to Table 1). Come back here when Part 1 is done.

The paper's Table 1 has 12 contact networks. Four match datasets you already worked with by size; add them first.
The raw files were deleted during cleanup, so download them again.

**Networks to add first** (sizes from the paper's Table 1; "metric X" is the metric-backbone size as the paper's
adjusted size `X`):

| Name in the paper | Nodes | Edges | Metric X (%) | Source file (from the earlier manifest) |
| --- | --- | --- | --- | --- |
| hospital | 75 | 1139 | 13.43 | `hospital_lyon_contacts.dat.gz` (already in `data/raw/`) |
| workplace1 | 92 | 755 | 20.63 | `http://www.sociopatterns.org/assets/data/tij_InVS.dat` |
| workplace2 | 217 | 4274 | 13.04 | `http://www.sociopatterns.org/assets/data/workplace_InVS15_tij.dat.gz` |
| french-primary-school | 242 | 8317 | 6.80 | `http://www.sociopatterns.org/assets/data/primaryschool.csv.gz` |

- [ ] **2.1 Download** the three missing files into `data/raw/`. If a URL no longer works, find the dataset on
  sociopatterns.org by name and check it by node and edge count.
- [ ] **2.2 Prepare each** (replace `--licence` text as needed):
  ```
  python run.py prepare-data --raw data/raw/tij_InVS.dat --name workplace1 --format contacts --domain contact --source http://www.sociopatterns.org/assets/data/tij_InVS.dat --licence "CC BY-NC-SA 4.0 (SocioPatterns)"
  python run.py prepare-data --raw data/raw/workplace_InVS15_tij.dat.gz --name workplace2 --format contacts --domain contact --source http://www.sociopatterns.org/assets/data/workplace_InVS15_tij.dat.gz --licence "CC BY-NC-SA 4.0 (SocioPatterns)"
  python run.py prepare-data --raw data/raw/primaryschool.csv.gz --name primary_school --format contacts --domain contact --source http://www.sociopatterns.org/assets/data/primaryschool.csv.gz --licence "CC BY-NC-SA 4.0 (SocioPatterns)"
  ```
  The `contacts` format reads columns 2 and 3 (`i`, `j`) of a `t i j ...` file and counts contact snapshots as weights.
- [ ] **2.3 Check each network against Table 1** before running anything. The printed `n=` and `m=` must equal the
  table. Then check the metric backbone:
  ```
  .venv\Scripts\python.exe -c "from backbone.network import load_network; from backbone.filters import metric_backbone_mask; n=load_network('NAME'); b=int(metric_backbone_mask(n).sum()); print(n.n, n.m, b, round(100*(b-(n.n-1))/(n.m-(n.n-1)),2))"
  ```
  Expected last number: 13.43 (hospital, already confirmed), 20.63, 13.04, 6.80. *Done when* all four match to two
  decimals. A mismatch means a data or proximity problem; stop and resolve it before continuing.
- [ ] **2.4 Add the names to `real_networks` in `protocol.yaml`.**
- [ ] **2.5 Optional, more of Table 1** (conference 113 / 2196, french-high-school 327 / 5818, kenya-households 31 / 216,
  malawi-village 84 / 346). The earlier conference (403 / 9565) and high-school (180 / 2220) files were **different**
  datasets, so check sizes before using any file. The larger US school and Copenhagen networks (up to 118,291 edges)
  are expensive; leave them for last.

---

## Step 3. Replication code  **[hospital]**

The paper's measures and summary differ slightly from this project's. These changes make a like-for-like comparison
possible. Keep the existing measures; add the paper's alongside. Develop and test everything on hospital.

- [ ] **3.1 Store the paper's three measures separately.**
  - `ρ_C` (eigenvector-centrality rank): `metrics.f_rank` already computes it as `parts["eigenvector"]`, but
    `pipeline.score_backbone` throws the parts away (`out["rank"], _ = ...`).
  - `ρ_t` and `ξ⁻¹_0.5`: `metrics.f_spread` returns `F_order` and `F_time`, but `pipeline.score_backbone` stores only their
    mean.
  - *Do:* in `pipeline.score_backbone`, add `rank_eigenvector`, `spread_order` and `spread_time` (the mean over seeds)
    to `out["_detail"]`; add the three names to `EXTRA` in `analysis.py` so `wide()` keeps them.
  - *Note:* the paper's `ρ_t` is a Spearman correlation without clipping. This project clips it at 0. Add a flag
    `clip=False` to `f_spread` for the replication run, and say which one you used.
- [ ] **3.2 Add the paper's sparsification axis.** The paper uses the adjusted size
  `X = (k − (n − 1)) / (m − (n − 1))`, where `k` is the number of edges kept, `X = 0` is a spanning tree and `X = 1`
  the full graph. This project uses a share of edges `q = k / m` with 8 fixed values.
  - *Do:* in `backbone/filters.py` add `adjusted_size(k, n, m)` and `q_from_adjusted(X, n, m)`
    (`q = ((n − 1) + X (m − n + 1)) / m`). Add `EvalConfig.grid="adjusted"`, which uses
    `X` in `{0.05, 0.10, ..., 1.00}` (20 values) converted to per-network `q`. Add `--grid adjusted` to `eval` in `run.py`
    and a `--tag _adj` so results do not overwrite the standard ones.
  - Per-network `q` values differ, so compare methods within a network, not across networks, until the AUC step.
  - For hospital alone this is simply a finer, evenly spaced budget grid for one network.
- [ ] **3.3 Add the AUC summary** (the paper's Eq. 16): for each network and method, the area under each measure's curve
  from `X̃` to 1, divided by `1 − X̃`, using the trapezoid rule on the `X` grid.
  - `X̃` is "the smallest sparsification which preserves connectivity" (p. 12). The paper does not say whether this is
    per method or per network, so compute **both**: per method (reuse `filters.connectivity_floor`) and one common
    value per network (the metric backbone size, which the paper uses for DBS).
  - *Do:* add `auc_table(w, measures, floor)` to `backbone/analysis.py`. *Test:* a constant curve at 0.8 gives AUC 0.8;
    a method that is disconnected at every `X < 1` raises a clear error, not a silent number.
- [ ] **3.4 Add the paper's test** **[more datasets]**: Mann-Whitney U on the per-network AUC values
  (`scipy.stats.mannwhitneyu`, `alternative="greater"`, SMDS against each other method). Also output the **paired
  per-network difference**, because with 4 networks the test has little power. With one network, print the per-method
  AUC values and the ranking only, and have the function say "needs at least 2 networks" instead of a p-value.
- [ ] **3.5 Add a `replicate` command** in `run.py`: evaluates the chosen networks on the adjusted grid with W, DF, wEffRes
  and SMDS (plus NC and HSS), writes AUC tables to `results/analysis/replication/` and a one-page table of claim versus
  result. *Done when* `python run.py replicate --networks hospital` runs (Part 1), and later
  `python run.py replicate --networks hospital workplace1 workplace2 primary_school` (Part 2).
- [ ] **3.6 Tests** in `tests/test_core.py`: `adjusted_size` of the full graph is 1 and of a spanning tree is 0;
  `q_from_adjusted` is its inverse; the AUC of a constant curve; hospital's metric backbone is 13.43% (a regression test
  for the check in 2.3).

---

## Step 4. Run the replication and compare with the paper

### Step 4A. On hospital alone  **[hospital]**  (Part 1)

- [ ] **4A.1 Run** (full settings, 100 runs per seed; hospital is the fastest network):
  ```
  python run.py replicate --networks hospital
  ```
- [ ] **4A.2 Compare with the paper qualitatively.** Hospital is in the paper's Table 1 and in its per-network
  supplement figures (4 to 6, pp. 21 to 23, whose values I could not extract), so compare the **ordering of the methods**
  on each measure, not exact numbers. For hospital, check:
  1. SMDS has the highest or near-highest AUC of `ρ_C`, above W, DF and wEffRes.
  2. SMDS is above W and DF on the AUC of `ρ_t`; W and DF are close.
  3. wEffRes and SMDS are above W and DF on the AUC of `ξ⁻¹_0.5`.
  4. W and DF disconnect the network earlier than SMDS (compare their `X̃`).
- [ ] **4A.3 Write `docs/REPLICATION.md`** with one row per claim: the paper's page, your hospital value, and
  "consistent / not consistent". State clearly that this is **one network** and cannot show statistical significance.
- [ ] **4A.4 Show that β does not drive the result.** Run `python run.py robustness --networks hospital` and report the
  10, 20 and 40 step variants next to the replication.
- [ ] **4A.5 Resolve differences.** For every claim that is not consistent, list candidate causes (β, seed set, clipping,
  `X̃` definition, wEffRes Laplacian weights) and test the cheapest one.

### Step 4B. Pooled replication  **[more datasets]**  (Part 2, Gate A)

- [ ] **4B.1 Run** on the four shared networks (8 to 40 minutes each):
  ```
  python run.py replicate --networks hospital workplace1 workplace2 primary_school --jobs 4
  ```
- [ ] **4B.2 Extend `docs/REPLICATION.md`** with the Mann-Whitney p-values and the paired per-network differences.
  Claims to check (from the paper, p. 6): in a new `docs/REPLICATION.md`: one row per claim of the paper, the paper's page,
  your result (with the per-network numbers), and "replicates / partly / does not".
  Claims to check (from the paper, p. 6):
  1. SMDS has a higher AUC of `ρ_C` than W, DF and wEffRes.
  2. SMDS has a higher AUC of `ρ_t` than W and DF; W and DF are about equal.
  3. For `ξ⁻¹_0.5`, wEffRes and SMDS are above W and DF.
  4. W and DF disconnect the network early (compare their `X̃` with SMDS).
- [ ] **4B.3 Resolve differences.** For every claim that does not replicate on the pooled set, list possible causes
  (network version, `X̃` definition, wEffRes Laplacian weights) and test the cheapest one. Do not hide a failure to
  replicate: it is a result.

*Gate A1 (Part 1):* `docs/REPLICATION.md` exists with every claim marked for hospital.
*Gate A (Part 2):* the pooled version of the same table exists for four networks.

---

## Step 5. Extension: does structure predict function?  **[more datasets, Part 2; 5.5 works on hospital]**

This is the project's main question. Everything is already implemented (E1, E2); the work is getting enough networks
and reading the result correctly.

- [ ] **5.1 Reach at least 5 real networks over at least 3 domains.** After Step 2 you have 4 contact networks. Add one
  non-contact network (Step 7.1), or more of Table 1. Below 5 networks every pooled verdict stays inconclusive by design.
- [ ] **5.2 Make the rule match your decision 0.3.** The current RQ1 rule is "weak agreement for at least 2 of 4
  functions", which paths and rankings alone can satisfy. *Do:* require spreading or communities, or report primary and
  supporting functions separately (`analysis.agreement_summary`, `report.rq1`).
- [ ] **5.3 Report usable cells.** `analysis.kendall_per_cell` drops cells where all methods have the same structural
  score, so the mean tau comes mostly from low budgets. *Do:* show `n_cells` used per function next to each mean.
- [ ] **5.4 Run:** `python run.py analyze`, `python run.py mechanisms`, `python run.py conclusions`.
- [ ] **5.5 Read the failure atlas** **[hospital]** (`results/analysis/real_dev/e2_atlas.csv`): which method, budget and function fail
  while structure looks fine, and the reverse. Quote the rates with the number of networks.
- [ ] **5.6 Replace the optimistic test.** H1 uses a Wilcoxon test over (network, budget) cells, which treats a network's
  budgets as independent. *Do:* use a cluster bootstrap over networks or a mixed model in
  `analysis.pathbased_vs_weight_filters`.

---

## Step 6. Extension: test the bridge explanation  **[6.1 and 6.2 on hospital; 6.3 and 6.4 need more data]**

The paper says W and DF fail "because they removed bridges very early" (p. 6). This project can test it.

- [ ] **6.1 Measure it on real networks** **[hospital]**. The bridge-retention ratio (`pipeline.mechanism`) is already computed. Problem:
  on real networks it uses Louvain seed 0 of the original network, which is one of the 20 partitions `F_comm` compares
  against. *Do:* compute the ratio from an independent partition (a different Louvain seed set), or report both.
- [ ] **6.2 Add the controlled manipulation** **[hospital]** (the paper's explanation becomes a tested cause). On hospital,
  with its Louvain partition defining within- and between-community edges (later also on synthetic networks, where the
  planted partition gives an exact definition), for a given method and budget `k`:
  1. take the method's backbone;
  2. **swap**: remove `s` of the lowest-ranked within-community edges that are kept and add the `s` highest-ranked
     between-community edges that are not, so the backbone stays at `k` edges;
  3. do the reverse (remove `s` kept bridges, add `s` within-community edges);
  4. re-measure `F_spread` and `F_comm` with the same random numbers (the simulator already shares them);
  5. repeat for `s` = 0, 2%, 5%, 10%, 20% of `k`, for W, DF and SMDS.
  - *Files:* new `run_bridge_intervention` in `backbone/studies.py`; reuse `pipeline.build_gcache` and
    `pipeline.score_backbone`; command `python run.py bridges`.
  - *Output:* change in `F_spread` and `F_comm` per bridge added or removed, by method, with a bootstrap interval.
  - *Done when:* a table shows whether adding bridges to W or DF raises fidelity at fixed budget. If it does not, the
    explanation is not supported, and that is a finding.
- [ ] **6.3 Use the network properties** **[more datasets]**. `network.describe()` computes density, weight spread, coupling, clustering,
  modularity and the semi-metric share, but nothing calls it. *Do:* regress the gap on these across the real and synthetic
  networks (`analysis.py`, `studies.py`). The paper itself says this is "not completely understood" (p. 8).
- [ ] **6.4 Include mixing in the causal verdict** **[synthetic networks, Part 2]**. The synthetic model fits `wb:mixing`, but `report.rq3_synthetic` judges
  only spread and coupling.

---

## Step 7. Extension: other domains, random baselines, robustness  **[7.2 to 7.4 on hospital; 7.1 needs more data]**

- [ ] **7.1 One non-contact network per domain** **[more datasets]**. Netzschleuder datasets used earlier: residence_hall (social, 217 nodes,
  1,839 edges), migration (economic, 232 / 7,895) and foodweb_baywet (biological, 128 / 2,075), from
  `https://networks.skewed.de/net/<name>/`. Directed pairs are merged by summing both directions. **Check the weight
  column:** `prepare-data --format edgelist` uses the third column as the weight, and some Netzschleuder files have
  several columns, so extract the weight column you want into a three-column file first.
- [ ] **7.2 Random baselines and skill scores** **[hospital]** are already in `eval` (R-U and R-W). Report skill next to each result.
  The paper has none, and Hamann found random deletion strong, so this is a real addition.
- [ ] **7.3 Communities and paths** **[hospital]**. Already measured. Report `F_comm` and `F_path` for the replicated networks.
  Expect W to do well on communities (Yan et al.); say so if it does.
- [ ] **7.4 Robustness** **[hospital]**. `python run.py robustness` (SIR, β target, proximity map, epsilon, Louvain resolution, seed
  rule). The paper lists SIR as future work, so SIR results are new. Re-run after the final network set is fixed.

---

## Step 8. Optional  **[8.1 and 8.3 on hospital; 8.2 needs more data]**

- [ ] **8.1 Implement DBS** **[hospital]** (the paper's own method). Only worth it if you want to compare against it directly.
  The method: for an edge, the distance is `d'_ij = (1/p_ij − 1)^λ` and the edge is in the backbone at λ if it is the
  shortest path between its endpoints under summed distances. `λ_ij` is the **largest** λ at which the edge is in the
  backbone; find it by binary search on `[1e-4, 1e3]` with 50 iterations (paper Algorithms 1 and 2, p. 16 to 17).
  Edges are removed in increasing `λ_ij`.
  - *Tests from the paper:* at `λ = 1` the backbone equals the metric backbone (`filters.metric_backbone_mask`). For
    hospital, the paper's Table 2 (p. 18) gives min `λ_ij` = 0.133, max = 16.276 and mode 0.337; your values should agree
    to about three decimals.
  - *Cost:* about `m × 50` shortest-path runs per network; fine for hospital, slow for the largest networks.
- [ ] **8.2 Advisor validation** **[more datasets]**. Either state in the paper that the advisor is not validated on unseen networks, or add
  leave-one-network-out validation once you have at least 5 networks (`advisor.py`, `studies.py`). The paper's own
  weighted objective (section 5.4, p. 12) is the natural starting point.
- [ ] **8.3 Decide the budget range.** The concept note says sparse to moderate; the grid goes to 70%. Keep and label
  50% to 70% as dense, or cut it.

---

## Step 9. Synthetic sweep, freeze, write  **[Part 2]**

- [ ] **9.1 Synthetic sweep.** `python run.py eval --network syn --reduced --jobs 16` evaluates 81 networks (3 per design
  cell; 12 are done and all lie in one corner of the design, so a partial run cannot test the causal factors).
  About 6.5 minutes per network, 69 remaining, so about 7.5 CPU-hours: roughly 30 to 60 minutes with 16 parallel jobs.
  Where the time goes: betweenness about half, Louvain about 40%, the epidemic about 7%. If too slow: fewer random
  draws, or 2 instances per cell (54 networks). Keep the six methods and the budgets. Then
  `python run.py mechanisms --synthetic` and `python run.py conclusions --synthetic-tag ""`.
  The design paper asks for at least 10 instances per cell; say in the paper that you used 3.
- [ ] **9.2 Re-run everything on the final network set:** `eval`, `samples`, `analyze`, `mechanisms`, `robustness`,
  `conclusions`, `export`. `robustness` results and `results/metric_share.json` go stale when networks change.
- [ ] **9.3 Review the UI:** lens, theatre, dark theme, narrow width, colour-blind check; build a demo script from the
  actual results. If a hypothesis is rejected, show that case.
- [ ] **9.4 Freeze the protocol** (Gate C): align the hypothesis text in `protocol.yaml`, set `status: frozen`, commit, tag
  `protocol-v1`. Anything run after the tag is labelled exploratory.
- [ ] **9.5 Reproduce:** regenerate everything from the stored configs with the one global seed and check that results do
  not change.
- [ ] **9.6 Update the docs** (DOCUMENTATION.md status and deviations, README).
- [ ] **9.7 Write the paper.** Suggested outline:
  1. Introduction: backbones, structural checks, the structure-versus-function question.
  2. Related work and the remaining gap (use BASE_PAPER.md and the table in Step 0).
  3. Methods: the six filters, matched budgets, the fidelity scores, and the adjusted-size AUC used for replication.
  4. Replication of Pereira et al.: the table from `docs/REPLICATION.md`.
  5. Extension results: agreement and failure atlas; bridge manipulation; other domains; robustness.
  6. The advisor and its limit.
  7. Discussion and limitations (few networks, static aggregation, SI and SIR only, one weight-to-probability map).
  Print the number of networks next to every claim. Do not claim to be the first comparison, or that bridges explain
  failures unless Step 6 supports it.

---

## Gates and risks

| Gate | Passes when |
| --- | --- |
| Step 1 | Tests pass, the code is committed |
| **A** | Replication table done for the four shared networks |
| **B** | Agreement, failure atlas and bridge manipulation done on 5+ networks |
| **C** | Protocol frozen and tagged; the final run reproduces |

| Risk | Fallback |
| --- | --- |
| The results do not replicate | Report it, with the candidate causes you tested |
| A paper already tests structure against function | Narrow to what remains: the bridge manipulation and the other domains |
| Structure predicts function well | Report it. That is a finding, and structural checks are then adequate |
| Too few networks | Say so beside each claim; use the synthetic design for the causal claims |
| Compute takes too long | Cut random draws, runs or instances first; never the methods |
| You run out of time | Stop after Gate A or B; either is a complete project |

## Open decisions

1. Labels for the hypotheses and research questions (0.2).
2. Which functions are headline (0.3).
3. Does your guide accept a replicate-and-extend project (0.4)?
4. Per-method or per-network `X̃` for the AUC (3.3): compute both and report both.
5. Whether to implement DBS (8.1).
6. Which non-contact networks (7.1).
