"""Analyse total persistence over time."""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from utilities import parse_keys


def total_persistence(diagram, p=2):
    """Calculate total persistence of a persistence diagram."""
    return np.sum(np.power(np.abs(np.diff(diagram[:, 0:2])), 2))


def process_file(filename):
    """Process individual file and return its total persistence values."""
    with open(filename) as f:
        data = np.load(filename, allow_pickle=True)
        parsed_keys = parse_keys(data)

        persistence_diagrams = [
            data[key] for key, _ in parsed_keys['persistence_points']
        ]

        values = np.asarray([
            total_persistence(diagram[diagram[:, 2] == 1])
            for diagram in persistence_diagrams
        ])

        return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', help='Input file(s)')

    args = parser.parse_args()

    N = len(args.INPUT)
    M = 0

    data = []

    for filename in args.INPUT:
        basename = os.path.basename(filename)
        parts = basename.split('_')

        # TODO: make this smarter...
        n_points = parts[1]
        radius = parts[2]

        values = process_file(filename)
        data.append(values)

        M = max(M, len(values))

    matrix = np.zeros((N, M))

    for i, values in enumerate(data):
        matrix[i, :len(values)] = values

    plt.matshow(matrix, aspect='equal')
    plt.show()
