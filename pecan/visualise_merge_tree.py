"""Visualise merge tree of condensation process."""

import argparse
import sys

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from scipy.cluster.hierarchy import dendrogram

from utilities import make_tensor
from utilities import parse_keys


def make_linkage_matrix(edges, n_vertices):
    """Create linkage matrix from edges."""
    # Store representatives and cluster sizes of each point. This is
    # required because the linkage matrix does not permit us to reuse
    # the indices of the original observations more than once.
    reps = {}

    # Prepare linkage matrix; this follows the number of merge steps of
    # the algorithm.
    Z = np.zeros((len(edges), 4))

    for i, (t, u, v) in enumerate(edges):
        rep_u = reps.get(u, (u, 1))
        rep_v = reps.get(v, (v, 1))

        Z[i, 0] = rep_u[0]
        Z[i, 1] = rep_v[0]
        Z[i, 2] = t

        n_observations = rep_u[1] + rep_v[1]
        Z[i, 3] = n_observations

        # The representation for both of the nodes now has to be
        # updated.
        reps[u] = (i + n_vertices, n_observations)
        reps[v] = (i + n_vertices, n_observations)

    return Z


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

    plt.switch_backend('pgf')

    import matplotlib

    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    fig, ax = plt.subplots(figsize=(4, 0.8))

    edges = data['diffusion_homology_edges']
    Z = make_linkage_matrix(edges, n_vertices)

    dendrogram(Z, no_labels=True)

    sns.despine(bottom=True, offset=5)

    plt.ylabel('$t$')
    plt.tight_layout()

    plt.savefig('/tmp/Dendrogram.pgf')
