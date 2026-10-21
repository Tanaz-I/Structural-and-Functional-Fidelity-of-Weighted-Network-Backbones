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

# name -> (file, nodes, edges, metric-backbone adjusted size X in %) as in Table 1 of the paper.
# hospital reproduces Table 1 exactly. workplace1 has the paper's nodes and edges but a smaller metric backbone
# (13.25% against 20.63%): the edge weights differ for a reason we could not find (the file is the official SocioPatterns
# one), so it is used as a second network, not as a verified replication.
NETWORKS = {
    "hospital": (DATA / "detailed_list_of_contacts_Hospital.dat", 75, 1139, 13.43),
    "workplace1": (DATA / "tij_InVS.dat", 92, 755, 20.63),
}


def load_contacts(path):
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


def load_graph(path):
    """Raw contact file -> graph."""
    return build_graph(edge_table(load_contacts(path)))


def load_network(name):
    """Graph of one of the networks by name (see NETWORKS)."""
    return load_graph(NETWORKS[name][0])
