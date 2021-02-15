"""Analyse metric space along condensation process."""

import argparse
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics.pairwise import pairwise_distances

from utilities import parse_keys
from utilities import make_tensor


def diameter(X, metric='euclidean'):
    """Calculate diameter of a point cloud."""
    distances = pairwise_distances(X, metric=metric)
    return np.max(distances)


def hausdorff_distance(X, Y, metric='euclidean'):
    """Calculate Hausdorff distance between point clouds.

    Calculates the Hausdorff distance between two finite metric spaces,
    i.e. two finite point clouds.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    distances = pairwise_distances(X=X, Y=Y)

    d_XY = np.max(np.min(distances, axis=1))
    d_YX = np.max(np.min(distances, axis=0))

    return max(d_XY, d_YX)


def process_file(filename, args):
    """Process individual file."""
    data = np.load(filename, allow_pickle=True)
    parsed_keys = parse_keys(data)

    if 'data' not in parsed_keys:
        return None

    X = make_tensor(data, parsed_keys['data'])

    hausdorff_distances = []
    diameters = []

    for i, X_ in enumerate(np.rollaxis(X, axis=2)):
        if i + 1 < X.shape[2]:
            if args.origin:
                hausdorff_distances.append(
                    hausdorff_distance(X_, X[:, :, 0])
                )
            else:
                hausdorff_distances.append(
                    hausdorff_distance(X_, X[:, :, i + 1])
                )
        # Ensures that there is always a Hausdorff distance available,
        # so as not to collide with the other statistics.
        else:
            # This is not an accident; most plotting tools will simply
            # ignore such values, thus not misleading us when creating
            # a visualisation.
            hausdorff_distances.append(np.nan)

        diameters.append(diameter(X_))

    n = len(hausdorff_distances)
    factor_t = n - 1 if args.normalise else 1.0

    result = {
        'basename': [os.path.splitext(os.path.basename(filename))[0]] * n,
        'filename': [filename] * n,
        'hausdorff_distance': hausdorff_distances,
        'diameter': diameters,
        't': np.arange(0, n) / factor_t,
    }

    # Add eigenvalue information; this is slightly more tricky than the
    # other statistics because there are as many eigenvalues as samples
    # in the data set.
    if 'P' in parsed_keys:
        P = make_tensor(data, parsed_keys['P'])
        E = []

        for P_ in np.rollaxis(P, axis=2):
            E.append(sorted(np.linalg.eigvalsh(P_), reverse=True)[:10])

        E = np.asarray(E)
        for i, column in enumerate(E.T[:10]):
            result[f'eigenvalues_{i}'] = column

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', help='Input file(s)')

    parser.add_argument(
        '-o', '--origin',
        action='store_true',
        help='If set, calculates distances from origin instead of using '
             'consecutive time steps.'
    )

    parser.add_argument(
        '-n', '--normalise',
        action='store_true',
        help='If set, normalises variables to [0, 1] to simplify their '
             'subseqeuent analysis.'
    )

    parser.add_argument(
        '-p', '--plot',
        type=str,
        help='If set, plots the specified column of the data set.'
    )

    args = parser.parse_args()

    data = []

    for filename in args.INPUT:
        basename = os.path.basename(filename)
        parts = basename.split('_')

        # TODO: make this smarter...
        n_points = parts[1]

        if len(parts) >= 3:
            radius = parts[2]

        row = process_file(filename, args)
        row = pd.DataFrame.from_dict(row)
        data.append(row)

    pd.set_option('precision', 4)

    df = pd.concat(data)
    print(
        df.to_csv(
            sep='\t',
            na_rep='',
            index=False
        )
    )

    if args.plot is not None:

        # Two variables, or so we hope; time for a regression plot.
        if ',' in args.plot:
            x, y = args.plot.split(',')

            sns.lmplot(
                data=df,
                x=x,
                y=y,
                hue='filename',
                legend=False,
            )

        # Ordinary line plot...
        else:
            sns.lineplot(
                data=df,
                x='t',
                y=args.plot,
                hue='filename'
            )

        sns.despine()

        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
        plt.tight_layout()
        plt.show()
