"""Create figures for preprint/proposal etc.

At present, this script only creates some PNG files that can be directly
dropped into a preprint.
"""

import argparse

import matplotlib
import matplotlib.collections
import matplotlib.colors
import matplotlib.lines
import matplotlib.pyplot as plt

import numpy as np

from utilities import parse_keys
from utilities import make_tensor
from utilities import get_limits


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

    parser.add_argument(
        '-o', '--output',
        help='Specifies an optional output file'
    )

    args = parser.parse_args()

    # Check whether all keys are available. We require persistence
    # points, persistence pairs, and obviously a data set.
    data = np.load(args.INPUT, allow_pickle=True)
    parsed_keys = parse_keys(data)

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

    fig, ax = plt.subplots(
        ncols=n_plots,
        figsize=(n_plots * 3, 3),
        squeeze=True,
    )

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
        cur_axis += 1

        # List of persistence pairs of the form (creation, destruction).
        # They only pertain to diffusion homology.
        pp = data['diffusion_homology_persistence_pairs']

        ax[cur_axis].set_title('Ambient diffusion homology')
        ax[cur_axis].set_xlim(-0.01, np.max(pp[:, 1]) + 0.01)
        ax[cur_axis].set_ylim(-0.01, len(pp[:, 1]) + 0.01)

        pp = [destruction for _, destruction in pp]

        segments = [
            [(0, i), (destruction, i)] for i, destruction in enumerate(pp)
        ]

        barcode = matplotlib.collections.LineCollection(segments=segments)
        ax[cur_axis].add_collection(barcode)
        ax[cur_axis].set_xlabel('$t$')
        ax[cur_axis].set_aspect(1.50)

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

        min_ = min(x_min, y_min)
        max_ = max(x_max, y_max)

        ax[cur_axis].set_xlim(min_ - 0.01, max_ + 0.01)
        ax[cur_axis].set_ylim(min_ - 0.01, max_ + 0.01)
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

        ax[cur_axis].set_aspect(1.0)

    # Ensures that we can see the label. Can't wait to do this with TikZ
    # instead...
    fig.subplots_adjust(bottom=.5)

    plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()
