"""Analyse (total) persistence over time."""

import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial

from utilities import parse_keys
from utilities import parse_filename


def total_persistence(diagram, p=2):
    """Calculate total persistence of a persistence diagram."""
    return np.power(np.abs(np.diff(diagram[:, 0:2])), p).sum()


def p_norm(diagram, p=2):
    """Calculate $p$-norm of a persistence diagram (per dimension)."""
    return np.power(
            np.power(np.abs(np.diff(diagram[:, 0:2])), p).sum(),
            1 / p
    )


def infinity_norm(diagram, p=2):
    """Calculate infinity norm of a persistence diagram."""
    if len(diagram[:, 0:2]) != 0:
        return np.power(np.max(np.abs(np.diff(diagram[:, 0:2]))), p)
    else:
        return 0.0


def calculate_persistence_diagram_summaries(diagrams):
    """Calculate set of persistence diagram summaries and return them.

    Parameters
    ----------
    diagrams : list
        List of persistence diagrams in tuple format, i.e. a list of
        arrays of shape `(m, 3)`, where `m` is the number of points,
        and each point is described by a triple `(c, d, dim)`.

    Returns
    -------
    Dictionary with persistence-based summaries; the key is the name of
    the summary statistic, with a potential `_d` suffix to indicate the
    dimension. The values are provided in the order of the input data.
    """
    # Collate *all* dimensions of all diagrams, ensuring that we can
    # always calculate them appropriately.
    dimensions = np.unique(
        list(
            itertools.chain(
                *[
                    np.unique(diagram[:, 2]).tolist()
                    for diagram in diagrams
                ]
            )
        )
    )

    summary_functions = {
        '2_norm': partial(p_norm, p=2),
        '1_norm': partial(p_norm, p=1),
        'infinity_norm_p2': partial(infinity_norm, p=2),
        'infinity_norm_p1': partial(infinity_norm, p=1),
        'total_persistence_p1': partial(total_persistence, p=1),
        'total_persistence_p1_normalized': partial(total_persistence, p=1),
        'total_persistence_p2': partial(total_persistence, p=2),
        'total_persistence_p2_normalized': partial(total_persistence, p=2),
    }

    # Will be filled with the results; to be merged with a larger data
    # frame later on.
    results = {}

    for dimension in dimensions:
        for name, fn in summary_functions.items():
            values = np.asarray([
                fn(diagram[diagram[:, 2] == dimension])
                for diagram in diagrams
            ])

            # Normalize if desired by the user. This changes the
            # interpretation slightly.
            if 'normalized' in name:
                if np.max(values) > 0:
                    values = values / np.max(values)

            results[f'{name}_d{dimension:d}'] = values

    return results


def process_file(filename):
    """Process individual file.

    This function processes a single filename and returns a data frame
    with information about the file. The data frame created here might
    not have the same number of columns available for each file, as an
    individual file might contain different kinds of information.
    """
    data = np.load(filename, allow_pickle=True)
    parsed_keys = parse_keys(data)

    name, tokens = parse_filename(filename)

    # Will become a data frame later on. We just pre-fill it with some
    # information about the calculations later on.
    results = {
        'name': [name],
        't': np.arange(len(parsed_keys['data']))
    }

    results = {**results, **tokens}

    if 'persistence_points' in parsed_keys:
        persistence_diagrams = [
            data[key] for key, _ in parsed_keys['persistence_points']
        ]

        summaries = calculate_persistence_diagram_summaries(
            persistence_diagrams
        )

        results = {**results, **summaries}

    df = pd.DataFrame({k: pd.Series(v) for k, v in results.items()})
    cols = ['name'] + list(tokens.keys())
    df[cols] = df[cols].ffill()

    return df


def aggregate(df):
    """Aggregate a given data frame in terms of mean/std."""
    df = df.groupby(['name', 'n_samples', 't']).agg(
        [np.mean, np.std]).reset_index()

    # Create single columns from the multi-index columns in the data
    # set, resulting in columns of the form `name_mean` for instance.
    df.columns = ['_'.join(col) for col in df.columns]

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', help='Input file(s)')

    parser.add_argument(
        '-s', '--statistic',
        default='infinity_norm_p2',
        type=str,
        help='Statistic to plot',
    )

    parser.add_argument(
        '-d', '--dimension',
        default=1,
        type=int,
        help='Dimension of topological features to plot'
    )

    args = parser.parse_args()

    data = []

    for filename in args.INPUT:
        df = process_file(filename)
        data.append(df)

    df = pd.concat(data)
    df['n_samples'] = pd.to_numeric(df['n_samples'])

    print(
        df.to_csv(sep='\t', na_rep='', index=False)
    )

    sns.lineplot(
        x='t',
        y=f'{args.statistic}_d{args.dimension}',
        hue='n_samples',
        data=df,
        estimator='mean',
        ci='sd',
        palette='husl',
    )

    plt.show()
