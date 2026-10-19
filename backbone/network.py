"""Network container, real-network loaders, synthetic generator and descriptors."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import itertools

from scipy.stats import spearmanr
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------- network
@dataclass
class Network:
    name: str
    n: int
    u: np.ndarray  # int64, u < v
    v: np.ndarray
    w: np.ndarray  # float64 > 0
    meta: dict = field(default_factory=dict)
    communities: np.ndarray | None = None  # planted partition if known

    @property
    def m(self) -> int:
        return len(self.w)

    # ---- construction -------------------------------------------------
    @staticmethod
    def from_edges(name, u, v, w, meta=None, communities=None, lcc=True) -> "Network":
        """Clean raw edges: drop loops, merge parallel edges, keep the largest component."""
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        w = np.asarray(w, dtype=np.float64)
        keep = (u != v) & (w > 0)
        u, v, w = u[keep], v[keep], w[keep]
        a, b = np.minimum(u, v), np.maximum(u, v)
        n0 = int(max(a.max(), b.max())) + 1
        key = a * n0 + b
        uk, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=w)
        a, b = uk // n0, uk % n0
        if lcc:
            g = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n0, n0))
            _, lab = sp.csgraph.connected_components(g, directed=False)
            big = np.bincount(lab).argmax()
            nodes = np.flatnonzero(lab == big)
        else:
            nodes = np.unique(np.concatenate([a, b]))
        remap = -np.ones(n0, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        m = (remap[a] >= 0) & (remap[b] >= 0)
        a, b, w = remap[a[m]], remap[b[m]], w[m]
        if communities is not None:
            communities = np.asarray(communities)[nodes]
        order = np.lexsort((b, a))
        return Network(name, len(nodes), a[order], b[order], w[order], dict(meta or {}), communities)

    @staticmethod
    def from_networkx(name, g: nx.Graph, meta=None, weight="weight") -> "Network":
        ids = {x: i for i, x in enumerate(g.nodes())}
        u, v, w = [], [], []
        for a, b, d in g.edges(data=True):
            u.append(ids[a]); v.append(ids[b]); w.append(d.get(weight, 1.0))
        return Network.from_edges(name, u, v, w, meta)

    def to_networkx(self, mask: np.ndarray | None = None) -> nx.Graph:
        """Graph on the full node set; `mask` selects kept edges (isolates stay)."""
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        idx = np.arange(self.m) if mask is None else np.flatnonzero(mask)
        g.add_weighted_edges_from(zip(self.u[idx].tolist(), self.v[idx].tolist(), self.w[idx].tolist()))
        return g

    # ---- derived quantities ---------------------------------------------
    def strength(self, mask: np.ndarray | None = None) -> np.ndarray:
        idx = slice(None) if mask is None else mask
        return (np.bincount(self.u[idx], self.w[idx], self.n)
                + np.bincount(self.v[idx], self.w[idx], self.n))

    def degree(self, mask: np.ndarray | None = None) -> np.ndarray:
        idx = slice(None) if mask is None else mask
        return np.bincount(self.u[idx], minlength=self.n) + np.bincount(self.v[idx], minlength=self.n)

    def proximity(self, kind: str = "jaccard") -> np.ndarray:
        """p_ij in [0,1]. 'jaccard': w/(s_i+s_j-w) (default); 'max': w/w_max."""
        if kind == "max":
            return self.w / self.w.max()
        s = self.strength()
        return self.w / (s[self.u] + s[self.v] - self.w)

    def distance(self, kind: str = "jaccard") -> np.ndarray:
        if kind == "inverse":  # d = 1/w, the convention NetBone uses (validation only)
            return 1.0 / self.w
        return 1.0 / self.proximity(kind) - 1.0

    def csr(self, values: np.ndarray, mask: np.ndarray | None = None) -> sp.csr_matrix:
        """Symmetric sparse matrix of `values` over kept edges (n x n)."""
        idx = np.arange(self.m) if mask is None else np.flatnonzero(mask)
        r = np.concatenate([self.u[idx], self.v[idx]])
        c = np.concatenate([self.v[idx], self.u[idx]])
        d = np.concatenate([values[idx], values[idx]])
        return sp.csr_matrix((d, (r, c)), shape=(self.n, self.n))

    def n_components(self, mask: np.ndarray | None = None) -> int:
        idx = np.arange(self.m) if mask is None else np.flatnonzero(mask)
        g = sp.coo_matrix((np.ones(len(idx)), (self.u[idx], self.v[idx])), shape=(self.n, self.n))
        return int(sp.csgraph.connected_components(g, directed=False)[0])


# ---------------------------------------------------------------- loaders
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "MANIFEST.csv"
MANIFEST_COLS = ["name", "domain", "n", "m", "source", "note", "access_date", "raw_sha256", "licence"]


def load_edgelist(path, name: str, domain: str = "", sep=None, source: str = "") -> Network:
    """Whitespace/CSV edge list with columns u v [w]; node labels may be strings."""
    df = pd.read_csv(path, sep=sep, engine="python", comment="#", header=None)
    if df.shape[1] > 2 and pd.to_numeric(df.iloc[0, 2], errors="coerce") != pd.to_numeric(df.iloc[0, 2], errors="coerce"):
        df = df.iloc[1:].reset_index(drop=True)  # first row is a header such as u,v,weight
    labels = pd.factorize(pd.concat([df.iloc[:, 0], df.iloc[:, 1]]))[0]
    u, v = labels[: len(df)], labels[len(df):]
    w = df.iloc[:, 2].to_numpy(float) if df.shape[1] > 2 else np.ones(len(df))
    return Network.from_edges(name, u, v, w, {"domain": domain, "source": source})


def read_contacts(path, name: str, domain: str = "contact", source: str = "", cols=(1, 2)) -> Network:
    """Temporal contact list (t i j ...), e.g. SocioPatterns: weight = number of snapshots a pair was in contact."""
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=list(cols), engine="python", compression="infer")
    a, b = df[cols[0]], df[cols[1]]
    pair = pd.DataFrame({"u": np.minimum(a, b), "v": np.maximum(a, b)})
    agg = pair.groupby(["u", "v"]).size().reset_index(name="w")
    ids = pd.factorize(pd.concat([agg.u, agg.v]))[0]
    return Network.from_edges(name, ids[: len(agg)], ids[len(agg):], agg.w.to_numpy(float),
                              {"domain": domain, "source": source})


def prepare_dataset(raw_path, name: str, fmt: str = "edgelist", domain: str = "", source: str = "",
                    note: str = "", licence: str = "") -> Network:
    """Turn a raw file into the standard form data/processed/<name>.csv (u,v,w; largest component, no loops,
    parallel edges merged) and record it in data/MANIFEST.csv. fmt: 'edgelist' (u v [w]) or 'contacts' (t i j ...)."""
    import datetime
    import hashlib
    raw_path = Path(raw_path)
    reader = {"edgelist": load_edgelist, "contacts": read_contacts}[fmt]
    net = reader(raw_path, name, domain=domain, source=source)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"u": net.u, "v": net.v, "w": net.w}).to_csv(PROCESSED / f"{name}.csv", index=False)
    row = {"name": name, "domain": domain, "n": net.n, "m": net.m, "source": source, "note": note,
           "access_date": datetime.date.today().isoformat(), "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
           "licence": licence}
    old = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame(columns=MANIFEST_COLS)
    pd.concat([old[old.name != name], pd.DataFrame([row])], ignore_index=True)[MANIFEST_COLS].to_csv(MANIFEST, index=False)
    return net


def dataset_info(name: str) -> dict:
    """Manifest row of a prepared dataset (empty dict if it has none)."""
    if not MANIFEST.exists():
        return {}
    df = pd.read_csv(MANIFEST).fillna("")
    row = df[df.name == name]
    return row.iloc[0].to_dict() if len(row) else {}


def available_datasets() -> list[str]:
    return sorted(p.stem for p in PROCESSED.glob("*.csv")) if PROCESSED.exists() else []


_CACHE: dict[str, Network] = {}


def _synthetic(name: str) -> Network | None:
    """'pilot<i>' = i-th pilot network, 'syn<i>' = i-th network of the 270-network factorial grid."""
    for prefix, grid in (("pilot", pilot_grid()), ("syn", factorial_grid())):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            i = int(name[len(prefix):])
            m, sg, c, k, seed = list(grid)[i]
            return make_synthetic(m, sg, c, seed, name=name)
    return None


def load_network(name: str) -> Network:
    """A prepared dataset (data/processed/<name>.csv), a synthetic network ('syn<i>', 'pilot<i>'),
    or the path of an edge-list file."""
    if name in _CACHE:
        return _CACHE[name]
    net = _synthetic(name)
    if net is None:
        f = PROCESSED / f"{name}.csv"
        if f.exists():
            df = pd.read_csv(f)
            net = Network.from_edges(name, df.u, df.v, df.w, {"domain": dataset_info(name).get("domain", ""),
                                                              "source": dataset_info(name).get("source", "")})
        elif Path(name).is_file():
            net = load_edgelist(name, Path(name).stem)
        else:
            raise KeyError(f"unknown network {name!r}; prepared datasets: {available_datasets()} "
                           "(add one with: python run.py prepare-data --raw FILE --name NAME)")
    _CACHE[name] = net
    return net


# ---------------------------------------------------------------- synthetic
MIXING = (0.1, 0.3, 0.5)
SIGMA = (0.5, 1.0, 1.5)
COUPLING = (0.0, 0.5, 1.0)
N_NODES, N_COMM, MEAN_DEGREE = 300, 6, 20


def make_synthetic(mixing: float, sigma: float, coupling: float, seed: int,
                   n: int = N_NODES, n_comm: int = N_COMM, mean_degree: float = MEAN_DEGREE,
                   name: str | None = None) -> Network:
    rng = np.random.default_rng(seed)
    comm = np.repeat(np.arange(n_comm), n // n_comm)
    size = n // n_comm
    # expected inter-community share of each node's edges equals `mixing`
    p_in = mean_degree * (1 - mixing) / (size - 1)
    p_out = mean_degree * mixing / (n - size)
    iu, ju = np.triu_indices(n, 1)
    p = np.where(comm[iu] == comm[ju], p_in, p_out)
    keep = rng.random(len(p)) < p
    u, v = iu[keep], ju[keep]
    deg = np.bincount(u, minlength=n) + np.bincount(v, minlength=n)
    w = rng.lognormal(0.0, sigma, len(u))
    w = w * (deg[u].astype(float) * deg[v]) ** coupling
    w = w / w.mean()  # keep the mean weight fixed across coupling levels
    meta = {"domain": "synthetic", "mixing": mixing, "sigma": sigma, "coupling": coupling, "seed": seed}
    return Network.from_edges(name or f"syn_m{mixing}_s{sigma}_c{coupling}_{seed}", u, v, w, meta, communities=comm)


def factorial_grid(instances: int = 10, base_seed: int = 1000):
    """Yield (mixing, sigma, coupling, instance, seed) for the 27-cell design."""
    for ci, (m, s, c) in enumerate(itertools.product(MIXING, SIGMA, COUPLING)):
        for k in range(instances):
            yield m, s, c, k, base_seed + ci * 100 + k


SYN_INSTANCES = 3  # instances per cell used in the study (names syn<i>, i = cell * 10 + k, k < SYN_INSTANCES)


def syn_names(instances: int = SYN_INSTANCES) -> list[str]:
    """Synthetic networks used in the study; indices stay those of the full 270-network grid so cached results remain valid."""
    return [f"syn{i}" for i in range(270) if i % 10 < instances]


def pilot_grid(base_seed: int = 90000):
    """Separate pilot set (disjoint seeds) used only to set tau_s, tau_f and beta."""
    cells = [(0.1, 0.5, 0.0), (0.3, 1.0, 0.5), (0.5, 1.5, 1.0), (0.1, 1.5, 1.0), (0.5, 0.5, 0.0)]
    for ci, (m, s, c) in enumerate(cells):
        for k in range(3):
            yield m, s, c, k, base_seed + ci * 10 + k


# ---------------------------------------------------------------- descriptors
def describe(net: Network, partition: np.ndarray | None = None) -> dict:
    from .filters import metric_backbone_mask  # local import: filters depends on this module
    deg = net.degree()
    g = net.to_networkx()
    if partition is None:
        comms = nx.community.louvain_communities(g, weight="weight", seed=0)
        partition = np.empty(net.n, dtype=int)
        for c, nodes in enumerate(comms):
            partition[list(nodes)] = c
    modularity = nx.community.modularity(
        g, [set(np.flatnonzero(partition == c).tolist()) for c in np.unique(partition)], weight="weight")
    prod = deg[net.u].astype(float) * deg[net.v]
    return {
        "name": net.name, "n": net.n, "m": net.m,
        "density": 2 * net.m / (net.n * (net.n - 1)),
        "weight_cv": float(net.w.std() / net.w.mean()),
        "weight_degree_coupling": float(spearmanr(net.w, prod).correlation),
        "clustering": float(nx.average_clustering(g)),
        "modularity": float(modularity),
        "semi_metric_share": float(1.0 - metric_backbone_mask(net).sum() / net.m),
        "domain": net.meta.get("domain", ""),
    }
