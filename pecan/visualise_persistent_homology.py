"""Analyse topology of the condensation process."""

import argparse
import os
import sys

import matplotlib
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


def total_persistence(diagram, p=2):
    """Calculate total persistence of a persistence diagram."""
    return np.sum(np.power(np.abs(np.diff(diagram[:, 0:2])), 2))


def update(i):
    """Update callback for the animation."""
    # Update time-varying point cloud, similarly to the other plotting
    # scripts.
    scatter.set_offsets(X[..., i])
    ax[0].set_title(f'Data (2D) @ $t={i}$')

    persistence_diagram.set_offsets(persistence_diagrams[i][:, 0:2])
    ax[1].set_title(f'Persistence diagram @ $t={i}$')


if __name__ == '__main__':

    matplotlib.rcParams['lines.markersize'] = 4

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    parser.add_argument(
        '-f', '--frame',
        default=None,
        type=int,
        help='Specifies frame to show (instead of animation)'
    )

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

    # Check whether an animation is desired or not. If not, we just show
    # a single frame and do not start the animation later on.
    if args.frame is not None:
        start_frame = args.frame
    else:
        start_frame = 0

    # Prepare point cloud visualisation

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]

    fig, ax = plt.subplots(ncols=4, figsize=(11, 4))
    fig.suptitle(os.path.splitext(os.path.basename(args.INPUT))[0])

    x_min, x_max, y_min, y_max = get_limits(X)

    ax[0].set_title(f'Data (2D) @ $t={start_frame}$')
    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))

    # Render first frame (or desired frame) before (potentially)
    # starting the animation.
    scatter = ax[0].scatter(X[:, 0, start_frame], X[:, 1, start_frame])

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
    ax[1].set_title(f'Persistence diagram @ $t={start_frame}$')

    cm = matplotlib.colors.ListedColormap(['r', 'b'])

    # Show the diagram of the initial point cloud
    persistence_diagram = ax[1].scatter(
        x=persistence_diagrams[start_frame][:, 0],
        y=persistence_diagrams[start_frame][:, 1],
        c=persistence_diagrams[start_frame][:, 2],
        cmap=cm,
    )

    # If a start frame has been selected by the user, we should not
    # start an animation.
    if args.frame is None:
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=T,
            repeat=args.repeat,
            interval=args.interval,
        )

    # Total persistence calculations, aggregated over dimensions. For
    # now, only dimension 0 and dimension 1 are supported.
    for dimension in [0, 1]:
        diagrams = [d[d[:, 2] == dimension] for d in persistence_diagrams]
        values = [total_persistence(d) for d in diagrams]

        ax[2].plot(values, c=cm(dimension))
        ax[2].set_title('Total persistence')
        ax[2].set_xlabel('$t$')

    # Persistence values as time-varying scatterplots. This is kind of
    # a *projection* of all topological features, but at least, we see
    # their behaviour over time (without tracking).
    #
    # TODO: does it make sense to do this for other dimensions that
    # dimension 1?
    for dimension in [1]:
        diagrams = [d[d[:, 2] == dimension] for d in persistence_diagrams]
        values = [np.abs(np.diff(d[:, 0:2])).flatten() for d in diagrams]

        ax[3].set_xlim(-1, len(values))

        for index, value in enumerate(values):
            ax[3].scatter(
                y=value,
                x=[index] * len(value),
                color=cm(dimension),
            )

    ax[3].set_title('Persistence over time')
    ax[3].set_xlabel('$t$')

    plt.tight_layout()
    plt.show()
