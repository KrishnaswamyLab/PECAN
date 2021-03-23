"""Create figures for preprint/proposal etc.

At present, this script only creates some PNG files that can be directly
dropped into a preprint.
"""

import argparse
import os

import matplotlib
import matplotlib.collections
import matplotlib.colors
import matplotlib.lines
import matplotlib.animation as animation
import matplotlib.pyplot as plt

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

    #persistence_diagram.set_offsets(persistence_diagrams[i][:, 0:2])
    #ax[1].set_title(f'Persistence diagram @ $t={i}$')


if __name__ == '__main__':

    matplotlib.rcParams['lines.markersize'] = 4

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    parser.add_argument(
        '-f', '--frame',
        default=0,
        type=int,
        help='Specifies frame',
    )

    parser.add_argument(
        '-a', '--aspect',
        type=float,
        default=1.0,
        help='Aspect ratio',
    )

    parser.add_argument(
        '-b', '--barcode',
        default=False,
        action='store_true',
        help='If set, shows barcode'
    )

    parser.add_argument(
        '-d', '--diagram',
        default=False,
        action='store_true',
        help='If set, shows diagram'
    )

    args = parser.parse_args()

    # Check whether all keys are available. We require persistence
    # points, persistence pairs, and obviously a data set.
    data = np.load(args.INPUT, allow_pickle=True)
    parsed_keys = parse_keys(data)

    print(parsed_keys)

    assert 'data' in parsed_keys, 'Require "data" key'

    if args.barcode:
        assert 'diffusion_homology_persistence_pairs' in parsed_keys, \
            'Require "diffusion_homology_persistence_pairs" key'

    if args.diagram:
        assert 'persistence_points' in parsed_keys, \
            'Require "persistence_points" key'

    # Prepare point cloud visualisation. This corresponds to showing the
    # data at a certain point of the diffusion condensation process.

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]

    n_plots = 1 + args.barcode + args.diagram

    fig, ax = plt.subplots(ncols=n_plots, figsize=(n_plots * 3, 3), squeeze=True)

    # Simplify treatment of multiple axes below.
    if n_plots == 1:
        ax = [ax]

    for axis in ax:
        axis.set_aspect(args.aspect)

    # Will ensure that the right axis is always used for plotting. I am
    # sure this can be solved more elegantly.
    cur_axis = 0

    x_min, x_max, y_min, y_max = get_limits(X)

    ax[cur_axis].set_title(f'Data (2D) @ $t={args.frame}$')
    ax[cur_axis].set_xlim((x_min, x_max))
    ax[cur_axis].set_ylim((y_min, y_max))

    # Render first frame (or desired frame) before (potentially)
    # starting the animation.
    scatter = ax[cur_axis].scatter(X[:, 0, args.frame], X[:, 1, args.frame])

    if args.barcode:
        pass

    if args.diagram:
        cur_axis += 1

        persistence_diagrams = [
            data[key] for key, _ in parsed_keys['persistence_points']
        ]

        persistence_diagram = persistence_diagrams[args.frame]

        x_min = np.min(persistence_diagram[:, 0])
        x_max = np.max(persistence_diagram[:, 0])
        y_min = np.min(persistence_diagram[:, 1])
        y_max = np.max(persistence_diagram[:, 1])

        ax[cur_axis].set_xlim(x_min - 0.01, x_max + 0.01)
        ax[cur_axis].set_ylim(y_min - 0.01, y_max + 0.01)
        ax[cur_axis].axline((-0.1, -0.1), slope=1.0, c='k')
        ax[cur_axis].set_title(f'Persistence diagram @ $t={args.frame}$')

        # Show the diagram of the initial point cloud

        cm = matplotlib.colors.ListedColormap(['r', 'b'])

        persistence_diagram = ax[cur_axis].scatter(
            x=persistence_diagrams[args.frame][:, 0],
            y=persistence_diagrams[args.frame][:, 1],
            c=persistence_diagrams[args.frame][:, 2],
            cmap=cm,
        )

    plt.tight_layout()
    plt.show()
