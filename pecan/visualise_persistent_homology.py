"""Analyse topology of the condensation process."""

import argparse
import sys

import matplotlib.collections
import matplotlib.colors
import matplotlib.lines
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from utilities import parse_keys
from utilities import make_tensor
from utilities import get_limits


def update(i):
    """Update callback for the animation."""
    # Update time-varying point cloud, similarly to the other plotting
    # scripts.
    scatter.set_offsets(X[..., i])
    ax[0].set_title(f'Data (2D) @ $t={i}$')

    persistence_diagram.set_offsets(persistence_diagrams[i][:, 0:2])
    ax[1].set_title(f'Persistence diagram @ $t={i}$')


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

    fig, ax = plt.subplots(ncols=2, figsize=(6, 3))

    x_min, x_max, y_min, y_max = get_limits(X)

    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))

    # Render first time step before the animation starts.
    scatter = ax[0].scatter(X[:, 0, 0], X[:, 1, 0])

    # Show persistence points in all dimensions (collated). To this end,
    # collect all the diagrams in one vector.

    persistence_diagrams = [
        data[key] for key, _ in parsed_keys['persistence_points']
    ]

    y_max = 0.0
    for pd in persistence_diagrams:
        y_max = max(y_max, np.max(pd[:, 1]))

    # TODO: this assumes that we are always visualising zero-dimensional
    # persistent homology. If this is *not* the case, the limits need to
    # be updated.
    ax[1].set_xlim(-0.1, y_max * 1.05)
    ax[1].set_ylim(-0.1, y_max * 1.05)
    ax[1].axline((-0.1, -0.1), slope=1.0, c='k')

    # Show the diagram of the initial point cloud
    persistence_diagram = ax[1].scatter(
        x=persistence_diagrams[0][:, 0],
        y=persistence_diagrams[0][:, 1],
        c=persistence_diagrams[0][:, 2],
        cmap=matplotlib.colors.ListedColormap(['r', 'b'])
    )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        repeat=args.repeat,
        interval=args.interval,
    )

    plt.show()

    sys.exit(0)

    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')

    pairs = []
    points = []

    for key in data.keys():
        if key.startswith('pairs'):
            pairs.append(data[key])
        elif key.startswith('points'):
            points.append(data[key])

    # 3D persistence diagram
    D3 = []

    cm = matplotlib.cm.get_cmap('Spectral')
    max_t = 0

    for index, values in enumerate(points):
        if len(values[:, 0]) > 0:
            max_t += 1

    for index, values in enumerate(points):
        x = values[:, 0]
        y = values[:, 1]
        z = [index] * len(x)

        ax3.scatter(
            x, z, y,
            color=cm(index / (max_t - 1)),
            edgecolors='black'
        )

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

    plt.show()
