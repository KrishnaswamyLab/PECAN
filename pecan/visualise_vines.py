"""Visualise time-varying topological features."""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import pairwise_distances

from utilities import parse_keys
from utilities import make_tensor
from utilities import get_limits


def make_3d_vine_plot(persistence_pairs, persistence_points):
    """Create 3D vine plot."""
    # Check whether the number of features coincides for both diagram
    # vectors.
    assert len(persistence_pairs) == len(persistence_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cm = matplotlib.cm.get_cmap('Spectral')
    T = len(persistence_pairs)

    ax.set_ylabel('$t$')
    ax.set_xlabel('Creation')
    ax.set_zlabel('Destruction')

    # Will contain all persistence points as an $(n, 3)$ matrix, where
    # $n$ is the total number of persistence features over *all* times
    # and dimensions 0/1 correspond to creation/destruction.
    persistence_diagram_3d = []

    # TODO: make configurable?
    dimension = 1

    for t, points in enumerate(persistence_points):
        points = points[points[:, 2] == dimension]

        x = points[:, 0]
        y = points[:, 1]
        z = [t] * len(x)

        ax.scatter(
            x, z, y,
            color=cm(t / (T - 1)),
            edgecolors='black'
        )

        persistence_diagram_3d.append(
            np.asarray([x, y, z]).T
        )

    # This is now an (n x 3) matrix, where n is the number of
    # topological features. Columns 1 and 2 correspond to the
    # spatial positions of data, whereas column 3 indicates a
    # time step for a feature.
    persistence_diagram_3d = np.concatenate(persistence_diagram_3d)

    # Try to track persistence features. This is kind of like trying to
    # generate a persistence vineyard construction and hoping that all
    # creators remain the same.

    pairs = []

    # FIXME: this is clunky; the issue is that I need to convert the
    # persistence pairs, which are stored as lists of varying length,
    # to a proper pairing.
    for all_pairs in persistence_pairs:
        pairs_ = []
        for sigma, tau in all_pairs:
            if len(sigma) == dimension + 1:
                pairs_.append((sigma, tau))

        pairs.append(pairs_)

    # `Pairs` will now contain persistence pairs for each time step, and
    # we can try matching them.
    #
    # TODO: this is *not* a proper replacement for the vineyards!
    global_index = 0
    for index, (first, second) in enumerate(zip(pairs, pairs[1:])):
        for i1, (c1, d1) in enumerate(first):
            for i2, (c2, d2) in enumerate(second):
                if c1 == c2 or d1 == d2:
                    print(index, '--', index + 1, c1, d1, c2, d2)

                    x1, y1, z1 = persistence_diagram_3d[global_index + i1]
                    x2, y2, z2 = persistence_diagram_3d[
                        global_index + len(first) + i2
                    ]

                    ax.plot(
                        [x1, x2],
                        [z1, z2],
                        [y1, y2],
                        c='k'
                    )

        global_index += len(first)


def make_2d_vine_plot(
    persistence_pairs,
    persistence_points,
    diffusion_homology_persistence_pairs
):
    """Create 2D vine plot."""
    # Check whether the number of features coincides for both diagram
    # vectors.
    assert len(persistence_pairs) == len(persistence_points)

    fig = plt.figure()
    ax = fig.subplots()

    cm = matplotlib.cm.get_cmap('Spectral')
    T = len(persistence_pairs)

    ax.set_xlabel('$t$')
    ax.set_ylabel('Persistence')

    # Will contain all persistence values as an $(n, 2)$ matrix, where
    # $n$ is the total number of persistence features over *all* times
    # and dimensions 0 corresponds to the persistence.
    persistence_values = []

    # TODO: make configurable?
    dimension = 1

    for t, points in enumerate(persistence_points):
        points = points[points[:, 2] == dimension]

        y = np.abs(points[:, 1] - points[:, 0])
        x = [t] * len(y)

        ax.scatter(
            x, y,
            color=cm(t / (T - 1)),
            edgecolors='black'
        )

        persistence_values.append(
            np.asarray([x, y]).T
        )

    persistence_values = np.concatenate(persistence_values)

    merge_events = np.unique(diffusion_homology_persistence_pairs[:, 1])
    for merge_event in merge_events:
        ax.axvline(merge_event)

    # Try to track persistence features but only based on their
    # persistence value; the resulting vines are much easier to
    # interpret. Again, this is kind of like trying to generate
    # a persistence vineyard...
    pairs = []

    # FIXME: this is clunky; the issue is that I need to convert the
    # persistence pairs, which are stored as lists of varying length,
    # to a proper pairing.
    for all_pairs in persistence_pairs:
        pairs_ = []
        for sigma, tau in all_pairs:
            if len(sigma) == dimension + 1:
                pairs_.append((sigma, tau))

        pairs.append(pairs_)

    # `Pairs` will now contain persistence pairs for each time step, and
    # we can try matching them.
    #
    # TODO: this is *not* a proper replacement for the vineyards!
    global_index = 0
    for index, (first, second) in enumerate(zip(pairs, pairs[1:])):
        for i1, (c1, d1) in enumerate(first):
            for i2, (c2, d2) in enumerate(second):
                if c1 == c2 or d1 == d2:
                    x1, y1 = persistence_values[global_index + i1]
                    x2, y2 = persistence_values[
                        global_index + len(first) + i2
                    ]

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        c='k'
                    )

        global_index += len(first)


def make_2d_simplex_plot(X):
    """Create 2D simplex transposition plot."""
    # Turn `D` into a matrix of shape `(T, M)`, where `T` is the total
    # number of time steps, and `M` is the total number of edges, i.e.
    # the number of points squared.
    D = np.array([
            pairwise_distances(X_, metric='euclidean').flatten()
            for X_ in np.rollaxis(X, axis=2)
        ]
    )

    fig, ax = plt.subplots()

    for row in D.T:
        ax.plot(row, c='k')

    ax.set_xlabel('$t$')


if __name__ == '__main__':

    matplotlib.rcParams['lines.linewidth'] = 0.75

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

    parser.add_argument(
        '-t', '--show-transpositions',
        action='store_true',
        help='If set, show tranposition plot'
    )

    args = parser.parse_args()

    # Check whether all keys are available. We require persistence
    # points, persistence pairs, and obviously a data set.
    data = np.load(args.INPUT, allow_pickle=True)
    parsed_keys = parse_keys(data)

    assert 'data' in parsed_keys, 'Require "data" key'

    assert 'persistence_points' in parsed_keys, \
        'Require "persistence_points" key'

    assert 'persistence_pairs' in parsed_keys, \
        'Require "persistence_pairs" key'

    assert 'diffusion_homology_persistence_pairs' in parsed_keys, \
        'Require "diffusion_homology_persistence_pairs" key'

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]

    persistence_points = [
        data[key] for key, _ in parsed_keys['persistence_points']
    ]

    persistence_pairs = [
        data[key] for key, _ in parsed_keys['persistence_pairs']
    ]

    diffusion_homology_persistence_pairs = \
        data['diffusion_homology_persistence_pairs']

    make_3d_vine_plot(persistence_pairs, persistence_points)
    make_2d_vine_plot(
        persistence_pairs,
        persistence_points,
        diffusion_homology_persistence_pairs
    )

    if args.show_transpositions:
        make_2d_simplex_plot(X)

    plt.show()
