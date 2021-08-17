"""Visualise merge tree of condensation process."""

import argparse
import sys

import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

from utilities import make_tensor
from utilities import parse_keys
from utilities import UnionFind


def make_tree(edges, n_vertices):
    """Create tree from edges."""
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    uf = UnionFind(n_vertices)

    for t, u, v in edges:
        G.add_edge(u, v, time=t)

        uf.merge(u, v)

    # TODO: do something with `uf.roots()`?
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    parser.add_argument(
        '-i', '--interval', default=200, type=int,
        help='Update interval'
    )

    parser.add_argument(
        '-r', '--repeat',
        action='store_true',
        help='Indicates whether animation should loop'
    )

    args = parser.parse_args()

    if args.INPUT == '-':
        args.INPUT = sys.stdin

    # Load data and check whether all keys are available. We require
    # only the diffusion homology pairs and the data set here.
    data = np.load(args.INPUT)
    parsed_keys = parse_keys(data)

    n_vertices = make_tensor(data, parsed_keys['data']).shape[0]

    assert 'diffusion_homology_edges' in parsed_keys, \
        'Require "diffusion_homology_edges" key'

    edges = data['diffusion_homology_edges']
    tree = make_tree(edges, n_vertices)
    pos = nx.nx_agraph.graphviz_layout(
        tree,
        prog='dot',
        root=0,
    )

    nx.draw(tree, pos=pos, with_labels=True)

    plt.tight_layout()
    plt.show()
