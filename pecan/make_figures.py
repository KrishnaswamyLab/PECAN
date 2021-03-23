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

    fig, ax = plt.subplots(ncols=2, figsize=(3, 3))

    ax = [ax]
    ax.set_aspect(args.aspect)

    x_min, x_max, y_min, y_max = get_limits(X)

    ax[0].set_title(f'Data (2D) @ $t={start_frame}$')
    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))

    # Render first frame (or desired frame) before (potentially)
    # starting the animation.
    scatter = ax[0].scatter(X[:, 0, start_frame], X[:, 1, start_frame])

    # Show persistence points in all dimensions (collated). To this end,
    # collect all the diagrams in one vector.

   # persistence_diagrams = [
   #     data[key] for key, _ in parsed_keys['persistence_points']
   # ]

   # y_max = 0.0
   # for pd in persistence_diagrams:
   #     y_max = max(y_max, np.max(pd[:, 1]))

    # TODO: this assumes that we are always visualising zero-dimensional
    # persistent homology. If this is *not* the case, the limits need to
    # be updated.
    #ax[1].set_xlim(-0.1, y_max * 1.05)
    #ax[1].set_ylim(-0.1, y_max * 1.05)
    #ax[1].axline((-0.1, -0.1), slope=1.0, c='k')
    #ax[1].set_title(f'Persistence diagram @ $t={start_frame}$')

    cm = matplotlib.colors.ListedColormap(['r', 'b'])

    # Show the diagram of the initial point cloud
    #persistence_diagram = ax[1].scatter(
    #    x=persistence_diagrams[start_frame][:, 0],
    #    y=persistence_diagrams[start_frame][:, 1],
    #    c=persistence_diagrams[start_frame][:, 2],
    #    cmap=cm,
    #)

    plt.tight_layout()
    plt.show()
