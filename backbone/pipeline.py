"""Evaluate all methods and null baselines on one network at every budget (long-format output)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from . import metrics
from .analysis import FUNCS
from .filters import BUDGETS, FILTERS, NULLS, PROXIMITY_FILTERS, backbone_flags, edge_order, k_of, tie_permutation, tie_share, top_k_mask
from .network import Network

@dataclass
class EvalConfig:
    budgets: tuple = BUDGETS
    methods: tuple = tuple(FILTERS)
    seed: int = 0
    n_runs: int = 100          # SI runs per seed on method backbones
    n_runs_null: int = 20      # SI runs per seed on null backbones
    n_null_draws: int = 20
    louvain_seeds: int = 20
    target_half_time: float = 20.0
    proximity: str = "jaccard"
    eps: float = 0.1
    seed_frac: float = 0.05    # share of nodes per centrality tier used as SI seeds
    seed_mode: str = "pereira"  # or "uniform" (robustness): random 15% of nodes
    louvain_resolution: float = 1.0
    with_sir: bool = False      # also compute F_SIR (robustness check)
    sir_runs: int = 60


@dataclass
class GCache:
    """Everything about the original network that is reused for every backbone."""
    seeds: np.ndarray
    beta: float
    TG: dict            # n_runs -> infection times
    PG: list
    ceiling: float
    DG_src: np.ndarray
    DG: np.ndarray
    cG: dict
    partition: np.ndarray   # partition used for bridge ratio
    sir_g: np.ndarray | None = None


def build_gcache(net: Network, cfg: EvalConfig) -> GCache:
    seeds = (metrics.choose_seeds_uniform(net, 3 * cfg.seed_frac, cfg.seed) if cfg.seed_mode == "uniform"
             else metrics.choose_seeds(net, cfg.proximity, cfg.seed_frac))
    beta = metrics.calibrate_beta(net, seeds, cfg.target_half_time, cfg.seed, cfg.proximity)
    steps = int(10 * cfg.target_half_time)
    TG = {r: metrics.simulate(net, None, seeds, beta, r, steps, cfg.seed, cfg.proximity)
          for r in {cfg.n_runs, cfg.n_runs_null}}
    PG = metrics.louvain_partitions(net, None, range(cfg.louvain_seeds), cfg.louvain_resolution)
    sir_g = (metrics.sir_profile(net, None, beta, cfg.sir_runs, cfg.seed, cfg.proximity) if cfg.with_sir else None)
    src = metrics._sources(net, cfg.seed)
    return GCache(seeds, beta, TG, PG, metrics.self_agreement(PG), src,
                  metrics.path_distances(net, None, src, cfg.proximity),
                  metrics.centralities(net, None, cfg.proximity),
                  net.communities if net.communities is not None else PG[0], sir_g)


def score_backbone(net: Network, mask: np.ndarray, g: GCache, cfg: EvalConfig, null: bool = False) -> dict:
    runs = cfg.n_runs_null if null else cfg.n_runs
    out = {}
    out["struct"], sc = metrics.f_struct(net, mask)
    out["spread"], sp = metrics.f_spread(net, mask, g.seeds, g.beta, runs, cfg.target_half_time,
                                          cfg.seed, cfg.proximity, TG=g.TG[runs])
    out["comm"], ci = metrics.f_comm(net, mask, g.PG, g.ceiling, cfg.louvain_resolution)
    if cfg.with_sir:
        out["sir"], _ = metrics.f_sir(net, mask, g.beta, cfg.sir_runs, cfg.seed, cfg.proximity, g.sir_g)
    out["path"], pi = metrics.f_path(net, mask, cfg.eps, cfg.proximity, cfg.seed, DG=g.DG)
    out["rank"], _ = metrics.f_rank(net, mask, cfg.proximity, cG=g.cG)
    out["_samples"] = {"spread": (sp["F_order"] + sp["F_time"]) / 2, "comm": ci["samples"], "path": pi["samples"]}
    out["_detail"] = {**{f"struct_{k}": v for k, v in sc.items()}, "median_stretch": pi["median_stretch"],
                      "disconnected_pair_share": pi["disconnected_share"], "nmi_raw": ci["nmi_raw"]}
    return out


def mechanism(net: Network, mask: np.ndarray, partition: np.ndarray) -> dict:
    inter = partition[net.u] != partition[net.v]
    intra = ~inter
    r_inter = (mask & inter).sum() / max(inter.sum(), 1)
    r_intra = (mask & intra).sum() / max(intra.sum(), 1)
    s = net.strength()
    hubs = s >= np.quantile(s, 0.9)
    touch = hubs[net.u] | hubs[net.v]
    kept_w = net.w[mask].sum()
    return {"bridge_ratio": float(r_inter / r_intra) if r_intra > 0 else float("nan"),
            "hub_concentration": float(net.w[mask & touch].sum() / kept_w) if kept_w > 0 else 0.0}


def evaluate_network(net: Network, cfg: EvalConfig | None = None, progress=None, with_nulls: bool = True,
                     collect: bool = False) -> pd.DataFrame:
    """Rows: network, kind (method|null), name, q, function, score + flags and mechanism columns."""
    cfg = cfg or EvalConfig()
    g = build_gcache(net, cfg)
    perm = tie_permutation(net.m, cfg.seed)
    rows = []
    samples: dict = {}

    def add(kind, name, q, mask, scores, extra):
        det = scores.pop("_detail")
        smp = scores.pop("_samples")
        if collect and kind == "method":
            for f, arr in smp.items():
                samples[f"{name}|{q:.2f}|{f}"] = np.asarray(arr, dtype=np.float32)
        mech = mechanism(net, mask, g.partition)
        for f in ("struct", *FUNCS) + (("sir",) if "sir" in scores else ()):
            rows.append({"network": net.name, "kind": kind, "name": name, "q": q, "function": f,
                         "score": scores[f], "beta": g.beta, **extra, **mech, **det})

    for name in cfg.methods:
        score = FILTERS[name](net, cfg.proximity) if name in PROXIMITY_FILTERS else FILTERS[name](net)
        order = edge_order(score, net.w, perm)
        for q in cfg.budgets:
            k = k_of(q, net.m)
            mask = top_k_mask(order, k)
            extra = {**backbone_flags(net, mask), "tie_share": tie_share(score, order, k), "draw": -1}
            add("method", name, q, mask, score_backbone(net, mask, g, cfg), extra)
            if progress:
                progress(name, q)

    rng_root = np.random.SeedSequence(cfg.seed + 7919)
    for nname, fn in (NULLS.items() if with_nulls else ()):
        for d, child in enumerate(rng_root.spawn(cfg.n_null_draws)):
            order = fn(net.w, np.random.default_rng(child))
            for q in cfg.budgets:
                mask = top_k_mask(order, k_of(q, net.m))
                extra = {**backbone_flags(net, mask), "tie_share": 0.0, "draw": d}
                add("null", nname, q, mask, score_backbone(net, mask, g, cfg, null=True), extra)
        if progress:
            progress(nname, None)
    df = pd.DataFrame(rows)
    df.attrs["samples"] = samples
    df.attrs["seeds"] = g.seeds
    return df
