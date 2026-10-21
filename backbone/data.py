"""Data ingestion: SocioPatterns contact lists -> weighted proximity/distance graphs.

Follows the base paper (Pereira, Costa & Rocha 2026), Sec. 5.1 and Eq. 1:
    p_ij = r_ij / (r_ii + r_jj - r_ij)      proximity in (0, 1]
    d_ij = 1 / p_ij - 1                     distance in [0, inf)
r_ij is the number of 20 s contact intervals between i and j, and r_ii the total
number of contact intervals of i.

Contact files have rows `t i j [extra columns]`, separated by tabs or spaces; only the first three
columns are used (the extra ones are wards / classes).
"""
from pathlib import Path

import networkx as nx
import pandas as pd

DATA = Path("data")
RAW_HOSPITAL = DATA / "detailed_list_of_contacts_Hospital.dat"

# name -> (file, nodes, edges, metric-backbone adjusted size X in %) as in Table 1 of the paper
# NETWORKS are the four used by default in run.py; EXTRA_NETWORKS are available but slow (hundreds of nodes,
# thousands of edges) and are only run when asked for with `python run.py --networks ...`.
NETWORKS = {
    "hospital": (DATA / "detailed_list_of_contacts_Hospital.dat", 75, 1139, 13.43),
    "workplace1": (DATA / "tij_InVS.dat", 92, 755, 20.63),
    "conference": (DATA / "ht09_contact_list.dat", 113, 2196, 9.40),
    "workplace2": (DATA / "tij_InVS15.dat", 217, 4274, 13.04),
}
EXTRA_NETWORKS = {
    "french-high-school": (DATA / "High-School_data_2013.csv", 327, 5818, 5.04),
    "french-primary-school": (DATA / "primaryschool.csv", 242, 8317, 6.80),
}
ALL_NETWORKS = {**NETWORKS, **EXTRA_NETWORKS}


def load_contacts(path=RAW_HOSPITAL):
    """Raw contact events (t, i, j): one row per 20 s interval, whitespace separated, extra columns ignored."""
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1, 2], names=["t", "i", "j"])
    if df.isna().any().any():
        raise ValueError("contact file has missing values")
    return df[df.i != df.j].drop_duplicates()  # no self-loops / repeated rows


def edge_table(contacts):
    """Undirected edge list with columns u < v, r, proximity, distance."""
    u = contacts[["i", "j"]].min(axis=1)
    v = contacts[["i", "j"]].max(axis=1)
    edges = contacts.assign(u=u, v=v).groupby(["u", "v"]).size().rename("r").reset_index()
    r_ii = pd.concat([contacts["i"], contacts["j"]]).value_counts()  # total interactions per node
    denom = edges["u"].map(r_ii) + edges["v"].map(r_ii) - edges["r"]
    edges["proximity"] = edges["r"] / denom
    edges["distance"] = 1.0 / edges["proximity"] - 1.0
    return edges


def build_graph(edges):
    """networkx Graph keyed by the original person IDs; edges carry r, proximity, distance."""
    G = nx.Graph()
    for row in edges.itertuples(index=False):
        G.add_edge(int(row.u), int(row.v), r=int(row.r), proximity=row.proximity, distance=row.distance)
    return G


def load_graph(path=RAW_HOSPITAL, save_to=None):
    """Raw contact file -> graph. Optionally writes the processed edge table as CSV."""
    edges = edge_table(load_contacts(path))
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        edges.to_csv(save_to, index=False)
    return build_graph(edges)


def load_network(name, save_to=None):
    """Graph of one of the paper's networks by name (see ALL_NETWORKS)."""
    return load_graph(ALL_NETWORKS[name][0], save_to=save_to)


if __name__ == "__main__":
    for name, (path, n_paper, m_paper, _) in ALL_NETWORKS.items():
        if not path.exists():
            print(f"{name:22} missing file {path}")
            continue
        G = load_network(name)
        print(f"{name:22} nodes={G.number_of_nodes():4} (paper {n_paper})  edges={G.number_of_edges():5} "
              f"(paper {m_paper})  connected={nx.is_connected(G)}")
