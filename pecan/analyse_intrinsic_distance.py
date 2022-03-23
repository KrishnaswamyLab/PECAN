"""Analyse intrinsic diffusion distances."""

import argparse
import itertools

import numpy as np

from utilities import parse_keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    args = parser.parse_args()

    # Load data and check whether all keys are available. We require
    # diffusion homology distances and pairs.
    data = np.load(args.INPUT)
    parsed_keys = parse_keys(data)

    assert 'diffusion_homology_distances' in parsed_keys, \
        'Require "diffusion_homology_distances" key'

    D = data['diffusion_homology_distances']

    n = len(D)
    indices = np.arange(n)

    n_failed_triples = 0

    for x, y, z in itertools.combinations(indices, 3):
        n_failed_triples += D[x, y] > D[x, z] + D[z, y]

    print(n_failed_triples)
