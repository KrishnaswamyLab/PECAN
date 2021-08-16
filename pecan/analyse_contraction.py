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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', help='Input file(s)')

    args = parser.parse_args()

    data = dict()

    for filename in args.INPUT:
        name, tokens = parse_filename(filename)
        distances = process_file(filename)

        data[name] = distances
