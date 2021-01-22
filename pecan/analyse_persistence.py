"""Analyse (total persistence over time."""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from utilities import parse_keys


def total_persistence(diagram, p=2):
    """Calculate total persistence of a persistence diagram."""
    return np.sum(np.power(np.abs(np.diff(diagram[:, 0:2])), p))


def infinity_norm(diagram, p=2):
    """Calculate infinity norm of a persistence diagram."""
    if len(diagram[:, 0:2]) != 0:
        return np.power(np.max(np.abs(np.diff(diagram[:, 0:2]))), p)
    else:
        return 0.0


def process_file(filename):
    """Process individual file and return its total persistence values."""
    data = np.load(filename, allow_pickle=True)
    parsed_keys = parse_keys(data)

    if 'persistence_points' not in parsed_keys:
        return None

    persistence_diagrams = [
        data[key] for key, _ in parsed_keys['persistence_points']
    ]

    values = np.asarray([
        summary_fn(diagram[diagram[:, 2] == 1])
        for diagram in persistence_diagrams
    ])

    # Ensures comparability of the values.
    values = values / np.max(values)
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', help='Input file(s)')
    parser.add_argument(
        '-s', '--statistic',
        default='total_persistence',
        type=str,
        help='Pick summary statistic to calculate. Can be either one of '
             '[infinity_norm, total_persistence].'
    )

    args = parser.parse_args()

    N = len(args.INPUT)
    M = 0

    data = []

    if args.statistic == 'total_persistence':
        summary_fn = total_persistence
    elif args.statistic == 'infinity_norm':
        summary_fn = infinity_norm

    for filename in args.INPUT:
        basename = os.path.basename(filename)
        parts = basename.split('_')

        # TODO: make this smarter...
        n_points = parts[1]

        if len(parts) >= 3:
            radius = parts[2]

        values = process_file(filename)

        # Skip files that we cannot parse for one reason or the other.
        if values is None:
            continue

        data.append(values)
        M = max(M, len(values))

    matrix = np.zeros((N, M))

    for i, values in enumerate(data):
        matrix[i, :len(values)] = values

    plt.matshow(matrix, aspect='auto')
    plt.show()
