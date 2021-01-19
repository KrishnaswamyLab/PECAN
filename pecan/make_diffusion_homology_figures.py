"""Create figures for preprint/proposal etc."""

import argparse
import os

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

    args = parser.parse_args()

    # Check whether all keys are available. We require persistence
    # points, persistence pairs, and obviously a data set.
    data = np.load(args.INPUT, allow_pickle=True)
    parsed_keys = parse_keys(data)

    assert 'data' in parsed_keys, 'Require "data" key'

    assert 'diffusion_homology_persistence_pairs' in parsed_keys, \
        'Require "diffusion_homology_persistence_pairs" key'

    fig, ax = plt.subplots(figsize=(3,5))

    # List of persistence pairs of the form (creation, destruction). An
    # optional third dimension is ignored.
    pd = data['diffusion_homology_persistence_pairs']

    ax.set_title('Diffusion barcode')
    ax.set_xlim(0, np.max(pd[:, 1]))     # Length of longest bar
    ax.set_ylim(0, len(pd[:, 1]))        # How many bars?

    segments = [
        [(0, i), (destruction, i)] for i, (_, destruction) in enumerate(pd)
    ]

    print(pd)

    print(segments)

    barcode = matplotlib.collections.LineCollection(segments=segments)
    ax.add_collection(barcode)

    plt.tight_layout()
    plt.show()
