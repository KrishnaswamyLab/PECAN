"""Extract output of condensation process."""

import argparse
import sys

import numpy as np

from utilities import parse_keys
from utilities import make_tensor
from utilities import get_limits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

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

    np.savetxt(
        '/tmp/X.txt',
        X[:, :, 0],
        fmt='%.8f',
        delimiter=',',
        header='x,y',
        comments=''
    )

    raise 'heck'

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
