"""Calculate intrinsic diffusion homology based on distances."""

import argparse

import numpy as np

from gtda.homology import VietorisRipsPersistence

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

    assert 'diffusion_homology_persistence_pairs' in parsed_keys, \
        'Require "diffusion_homology_persistence_pairs" key'

    D = data['diffusion_homology_distances']
    pairs = data['diffusion_homology_persistence_pairs']

    np.savetxt('/tmp/D.txt', D, fmt='%.4f')

    # FIXME: perturbation, to be controlled by the client?
    # n = len(D)
    # D += np.random.default_rng().uniform(low=0.01, high=0.02, size=(n, n))
 
    print(D)

    vr = VietorisRipsPersistence(
        metric='precomputed',
        reduced_homology=True,
        collapse_edges=True,
        homology_dimensions=(0, 1)
    )
    diagram = vr.fit_transform(D[None, :, :])[0]

    persistence_pairs = diagram[diagram[:, 2] == 0][:, :2]
    diff = persistence_pairs - pairs
    diff = diff.sum()

    if diff != 0.0:
        print('Something is wrong: diffusion homology pairs do not match')

    # Only print cycles for now...should generalise, though?
    print(diagram[diagram[:, 2] == 1][:, :2])
    print(np.diff(diagram[diagram[:, 2] == 1][:, :2]).sum())
