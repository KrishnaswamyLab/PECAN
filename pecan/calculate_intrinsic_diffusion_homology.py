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

    vr = VietorisRipsPersistence(
        metric='precomputed',
        reduced_homology=True
    )
    diagram = vr.fit_transform([D])[0]

    persistence_pairs = diagram[diagram[:, 2] == 0][:, :2]
    diff = persistence_pairs - pairs
    diff = diff.sum()

    assert diff == 0.0, \
        'Something is wrong: diffusion homology pairs do not match'
