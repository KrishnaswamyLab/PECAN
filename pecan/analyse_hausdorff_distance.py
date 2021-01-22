"""Analyse (total persistence over time."""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from utilities import parse_keys
from utilities import make_tensor


def hausdorff_distance(X, Y):
    """Calculate Hausdorff distance between point clouds.

    Calculates the Hausdorff distance between two finite metric spaces,
    i.e. two finite point clouds.
    """
    d_x_Y = 0

    for x in X:
        d_x_y = sys.float_info.max
        for y in Y:
            d = np.linalg.norm(x - y, ord=None)
            d_x_y = min(d_x_y, d)

        d_x_Y = max(d_x_Y, d_x_y)

    d_y_X = 0

    for y in Y:
        d_y_x = sys.float_info.max
        for x in X:
            d = np.linalg.norm(x - y, ord=None)
            d_y_x = min(d_y_x, d)

        d_y_X = max(d_y_X, d_y_x)

    return max(d_x_Y, d_y_X)


def process_file(filename):
    """Process individual file and return Hausdorff distance curve."""
    data = np.load(filename, allow_pickle=True)
    parsed_keys = parse_keys(data)

    if 'data' not in parsed_keys:
        return None

    X = make_tensor(data, parsed_keys['data'])

    values = []

    for i, X_ in enumerate(np.rollaxis(X, axis=2)):
        if i + 1 < X.shape[2]:
            values.append(
                hausdorff_distance(X_, X[:, :, i + 1])
            )

    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', help='Input file(s)')

    args = parser.parse_args()

    data = []

    for filename in args.INPUT:
        basename = os.path.basename(filename)
        parts = basename.split('_')

        # TODO: make this smarter...
        n_points = parts[1]

        if len(parts) >= 3:
            radius = parts[2]

        values = process_file(filename)

        # Skip files that we cannot parse for one reason or the other.
        if values is not None:
            plt.plot(values)

    basenames = [
        os.path.splitext(os.path.basename(filename))[0]
        for filename in args.INPUT
    ]

    plt.xlabel('Diffusion condensation iteration')
    plt.show()
