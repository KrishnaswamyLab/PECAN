"""Create animations and figures for preprint/proposal etc."""

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


def update_diagram(i):
    """Update callback for animated persistence diagram."""
    persistence_diagram.set_offsets(X[i][:, 0:2])
    ax[0].set_title(f'Persistence diagram @ $t={i}$')


def update_barcode(i):
    """Update callback for animated barcodes."""
    values = [destruction for _, destruction in X if destruction <= i]

    segments = [
        [(0, i), (destruction, i)] for i, destruction in enumerate(values)
    ]

    barcode.set_segments(segments)


def update(i):
    """Update callback for the animation."""
    # Update time-varying point cloud, similarly to the other plotting
    # scripts.
    scatter.set_offsets(X[..., i])
    ax[0].set_title(f'Data (2D) @ $t={i}$')


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

    parser.add_argument(
        '-t', '--type',
        help='Select type of data to visualise. Use "condensation" for '
             'the diffusion condensation process, "barcode" for the '
             'diffusion homology barcode, and "diagram" for the joint '
             'topological features.',
        default='condensation',
    )

    args = parser.parse_args()

    data = np.load(args.INPUT, allow_pickle=True)
    parsed_keys = parse_keys(data)

    if args.type == 'condensation':
        assert 'data' in parsed_keys, 'Require "data" key'
        key = 'data'
        update_fn = update

    elif args.type == 'barcode':
        assert 'diffusion_homology_persistence_pairs' in parsed_keys, \
            'Require "diffusion_homology_persistence_pairs" key'

        update_fn = update_barcode

    elif args.type == 'diagram':
        assert 'persistence_pairs' in parsed_keys, \
            'Require "persistence_pairs" key'

        update_fn = update_diagram
    else:
        raise RuntimeError(f'Type "{args.type}" is unexpected')

    # Check whether an animation is desired or not. If not, we just show
    # a single frame and do not start the animation later on.
    if args.frame is not None:
        start_frame = args.frame
    else:
        start_frame = 0

    # Prepare point cloud visualisation
    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]

    if args.type == 'barcode':
        X = data['diffusion_homology_persistence_pairs']
    elif args.type == 'diagram':
        X = [
            data[key] for key, _ in parsed_keys['persistence_points']
        ]

    fig, ax = plt.subplots(ncols=1, figsize=(4, 4))
    fig.suptitle(os.path.splitext(os.path.basename(args.INPUT))[0])

    ax = [ax]

    if args.type == 'condensation':

        x_min, x_max, y_min, y_max = get_limits(X)

        ax[0].set_title(f'Data (2D) @ $t={start_frame}$')
        ax[0].set_xlim((x_min, x_max))
        ax[0].set_ylim((y_min, y_max))

        # Render first frame (or desired frame) before (potentially)
        # starting the animation.
        scatter = ax[0].scatter(X[:, 0, start_frame], X[:, 1, start_frame])

    elif args.type == 'barcode':

        x_min, x_max, y_min, y_max = get_limits(X)

        ax[0].set_title('Diffusion barcode')
        ax[0].set_xlim(0, np.max(X[:, 1]))     # Length of longest bar
        ax[0].set_ylim(0, len(X[:, 1]))        # How many bars?

        barcode = matplotlib.collections.LineCollection(segments=[])
        ax[0].add_collection(barcode)

        # Populate barcode directly with the first frame.
        update_fn(start_frame)

    elif args.type == 'diagram':

        cm = matplotlib.colors.ListedColormap(['r', 'b'])

        y_max = 0.0
        for pd in X:
            y_max = max(y_max, np.max(pd[:, 1]))

        ax[0].set_xlim(-0.1, y_max * 1.05)
        ax[0].set_ylim(-0.1, y_max * 1.05)
        ax[0].axline((-0.1, -0.1), slope=1.0, c='k')
        ax[0].set_title(f'Persistence diagram @ $t={start_frame}$')

        persistence_diagram = ax[0].scatter(
            x=X[start_frame][:, 0],
            y=X[start_frame][:, 1],
            c=X[start_frame][:, 2],
            cmap=cm,
        )

        update_fn(start_frame)

    # If a start frame has been selected by the user, we should not
    # start an animation.
    if args.frame is None:
        ani = animation.FuncAnimation(
            fig,
            update_fn,
            frames=T,
            repeat=args.repeat,
            interval=args.interval,
        )

        # TODO: make configurable
        ani.save('/tmp/Condensation_%03d.png', writer='imagemagick')

    plt.tight_layout()
    plt.show()
