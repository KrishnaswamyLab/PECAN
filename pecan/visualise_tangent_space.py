"""Visualise tangent space output of condensation process."""

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
    scatter.set_offsets(X[..., i])
    scatter.set_array(C[i])

    min_c, max_c = np.min(C[i]), np.max(C[i])
    range_c = max(np.abs(min_c), np.abs(max_c))
    scatter.set_clim((-range_c, range_c))


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
    assert 'curvature' in parsed_keys, 'Require "curvature" key'

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]
    C = {
        t: data[key] for key, t in parsed_keys['curvature']
    }

    # Plot dynamic point cloud first. This assumes that only two
    # dimensions are available; higher-dimensional data may need
    # an additional dimensionality reduction step before-hand.
    fig, ax = plt.subplots(figsize=(4, 4))

    x_min, x_max, y_min, y_max = get_limits(X)

    ax.set_title('Data (2D)')
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_facecolor('black')

    # Render first time step before the animation starts.
    scatter = ax.scatter(X[:, 0, 0], X[:, 1, 0], c=C[0], cmap='RdYlBu')

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=args.interval,
        repeat=args.repeat,
    )

    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()
