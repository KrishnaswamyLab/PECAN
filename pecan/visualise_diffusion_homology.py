"""Visualise output of condensation process."""

import argparse
import sys

import matplotlib.collections
import matplotlib.lines
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np

from utilities import parse_keys
from utilities import make_tensor
from utilities import get_limits


def update(i):
    """Update function for animation."""
    scatter.set_offsets(X[..., i][:, 0:2])

    # Figure out all intervals to draw here
    values = [destruction for _, destruction in pd if destruction <= i]

    segments = [
        [(0, i), (destruction, i)] for i, destruction in enumerate(values)
    ]

    barcode.set_segments(segments)


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

    if args.INPUT == '-':
        args.INPUT = sys.stdin

    # Load data and check whether all keys are available. We require
    # only the diffusion homology pairs and the data set here.
    data = np.load(args.INPUT)
    parsed_keys = parse_keys(data)

    assert 'data' in parsed_keys, 'Require "data" key'

    assert 'diffusion_homology_persistence_pairs' in parsed_keys, \
        'Require "diffusion_homology_persistence_pairs" key'

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]

    # Plot dynamic point cloud first. This assumes that only two
    # dimensions are available; higher-dimensional data may need
    # an additional dimensionality reduction step before-hand.
    fig, ax = plt.subplots(ncols=3, figsize=(10, 4))

    x_min, x_max, y_min, y_max = get_limits(X)

    ax[0].set_title('Data (2D)')
    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))

    # Render first time step before the animation starts.
    scatter = ax[0].scatter(X[:, 0, 0], X[:, 1, 0])

    # Set up the diffusion homology pairing. This just amounts to
    # accessing the distribution of pairs and turning them into a
    # barcode visualisation.

    # List of persistence pairs of the form (creation, destruction). An
    # optional third dimension is ignored.
    pd = data['diffusion_homology_persistence_pairs']

    ax[1].set_title('Diffusion barcode')
    ax[1].set_xlim(0, np.max(pd[:, 1]))     # Length of longest bar
    ax[1].set_ylim(0, len(pd[:, 1]))        # How many bars?

    barcode = matplotlib.collections.LineCollection(segments=[])
    ax[1].add_collection(barcode)

    # Basic animation setup; it will loop over all the frames in the
    # data set and depict the corresponding topological features in
    # a barcode plot.

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=args.interval,
        repeat=args.repeat,
    )

    # Plot total persistence for every time step; this is a cumulative
    # sum of the persistence values.

    total_persistence = [
        np.sum(np.diff(pd[pd[:, 1] <= t])) for t in range(T)
    ]

    ax[2].set_title('Total persistence (cumulative)')
    ax[2].plot(total_persistence)
    ax[2].set_xlabel('$t$')

    plt.tight_layout()
    plt.show()
