"""Analyse (total persistence over time."""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_distances

from utilities import parse_keys
from utilities import make_tensor


def diameter(X, metric='euclidean'):
    """Calculate diameter of a point cloud."""
    distances = pairwise_distances(X, metric=metric)
    return np.max(distances)


def hausdorff_distance(X, Y, metric='euclidean'):
    """Calculate Hausdorff distance between point clouds.

    Calculates the Hausdorff distance between two finite metric spaces,
    i.e. two finite point clouds.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    distances = pairwise_distances(X=X, Y=Y)

    d_XY = np.max(np.min(distances, axis=1))
    d_YX = np.max(np.min(distances, axis=0))

    return max(d_XY, d_YX)


def process_file(filename, args):
    """Process individual file and return Hausdorff distance curve."""
    data = np.load(filename, allow_pickle=True)
    parsed_keys = parse_keys(data)

    if 'data' not in parsed_keys:
        return None

    X = make_tensor(data, parsed_keys['data'])

    values = []

    for i, X_ in enumerate(np.rollaxis(X, axis=2)):
        if i + 1 < X.shape[2]:
            if args.origin:
                values.append(
                    hausdorff_distance(X_, X[:, :, 0])
                )
            else:
                values.append(
                    hausdorff_distance(X_, X[:, :, i + 1])
                )

    return np.asarray(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', help='Input file(s)')

    parser.add_argument(
        '-o', '--origin',
        action='store_true',
        help='If set, calculates distances from origin instead of using '
             'consecutive time steps.'
    )

    args = parser.parse_args()

    data = []

    for filename in args.INPUT:
        basename = os.path.basename(filename)
        parts = basename.split('_')

        # TODO: make this smarter...
        n_points = parts[1]

        if len(parts) >= 3:
            radius = parts[2]

        values = process_file(filename, args)

        # Skip files that we cannot parse for one reason or the other.
        if values is not None:
            plt.plot(
                values
            )

    basenames = [
        os.path.splitext(os.path.basename(filename))[0]
        for filename in args.INPUT
    ]

    plt.legend(basenames)

    plt.xlabel('Diffusion condensation iteration')
    plt.show()
