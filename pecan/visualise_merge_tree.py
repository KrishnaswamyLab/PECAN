"""Visualise merge tree of condensation process."""

import argparse
import sys

import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

from scipy.cluster.hierarchy import dendrogram

from utilities import parse_keys


def make_tree(edges):
    """Create tree from edges."""
    G = nx.Graph()

    for t, u, v in edges:
        G.add_edge(u, v, time=t)

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

    assert 'diffusion_homology_edges' in parsed_keys, \
        'Require "diffusion_homology_edges" key'

    edges = data['diffusion_homology_edges']
    tree = make_tree(edges)
    pos = nx.nx_pydot.graphviz_layout(tree, prog='dot')

    nx.draw(tree, pos=pos)

    plt.tight_layout()
    plt.show()
