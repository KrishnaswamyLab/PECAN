"""Analyse contractive behaviour of diffusion condensation."""

import argparse
import os

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances

from utilities import parse_keys
from utilities import make_tensor
from utilities import parse_filename


def process_file(filename):
    """Process individual file."""
    data = np.load(filename, allow_pickle=True)
    parsed_keys = parse_keys(data)

    if 'data' not in parsed_keys:
        return None

    X = make_tensor(data, parsed_keys['data'])

    result = []

    for X_ in np.rollaxis(X, axis=2):
        D = pairwise_distances(X_)
        result.append(D)

    return result


def calculate_contraction_constant(dist):
    """Calculate overall contraction constant over all time steps."""
    all_scale_factors = []
    for i in range(dist[0].shape[0]):
        for j in range(i + 1, dist[0].shape[0]):
            with np.errstate(divide='ignore', invalid='ignore'):
                pairwise_scale_factors = np.asarray([
                    d2[i, j] / d1[i, j] for d1, d2 in zip(dist, dist[1:])
                ])

                pairwise_scale_factors = pairwise_scale_factors[
                    np.isfinite(pairwise_scale_factors)
                ]

            scale_factor = np.nanmax(pairwise_scale_factors)
            all_scale_factors.append(scale_factor)

    return np.nanmax(all_scale_factors)


def calculate_neighbourhood_consistency(dist, tau=1e-8):
    """Calculate neighbourhood consistency over time."""
    neighbourhoods = []
    masks = []
    for D in dist:
        m = D < tau
        neighbourhoods.append(np.sum(m))
        masks.append(m.ravel())

    masks = np.asarray(masks).astype(int)
    return np.min(np.min(np.diff(masks, axis=0), axis=0)) >= 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', help='Input file(s)')

    args = parser.parse_args()

    data = dict()

    for filename in args.INPUT:
        name, tokens = parse_filename(filename)
        distances = process_file(filename)

        data[name] = [
            calculate_contraction_constant(distances),
            calculate_neighbourhood_consistency(distances)
        ]

    data = pd.DataFrame.from_dict(
        data,
        orient='index',
        columns=['L', 'C'],
    )
    print(data)
