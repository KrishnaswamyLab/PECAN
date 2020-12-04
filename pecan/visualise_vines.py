"""Visualise time-varying topological features."""

import argparse
import os
import sys

import matplotlib.collections
import matplotlib.colors
import matplotlib.lines
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np

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

    for t, points in enumerate(persistence_points):
        points = points[points[:, 2] == 1]

        x = points[:, 0]
        y = points[:, 1]
        z = [t] * len(x)

        ax.scatter(
            x, z, y,
            color=cm(t / (T - 1)),
            edgecolors='black'
        )

    return

    D3.append(np.asarray([x, y, z]).T)

    # This is now an (n x 3) matrix, where n is the number of
    # topological features. Columns 1 and 2 correspond to the
    # spatial positions of data, whereas column 3 indicates a
    # time step for a feature.
    D3 = np.concatenate(D3)

    global_index = 0
    for index, (first, second) in enumerate(zip(pairs, pairs[1:])):
        for i1, (c1, d1) in enumerate(first):
            for i2, (c2, d2) in enumerate(second):
                if c1 == c2 or  d1 == d2:
                    print(index, '--', index + 1, c1, d1, c2, d2)

                    x1, y1, z1 = D3[global_index + i1]
                    x2, y2, z2 = D3[global_index + len(first) + i2]

                    ax3.plot(
                        [x1, x2],
                        [z1, z2],
                        [y1, y2],
                        c='k'
                    )

        global_index += len(first)



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

    # Check whether all keys are available. We require persistence
    # points, persistence pairs, and obviously a data set.
    data = np.load(args.INPUT, allow_pickle=True)
    parsed_keys = parse_keys(data)

    assert 'data' in parsed_keys, 'Require "data" key'

    assert 'persistence_points' in parsed_keys, \
        'Require "persistence_points" key'

    assert 'persistence_pairs' in parsed_keys, \
        'Require "persistence_pairs" key'

    # Prepare point cloud visualisation

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]

    persistence_points = [
        data[key] for key, _ in parsed_keys['persistence_points']
    ]

    persistence_pairs = [
        data[key] for key, _ in parsed_keys['persistence_pairs']
    ]

    make_3d_vine_plot(persistence_pairs, persistence_points)

    plt.show()
