"""Extract output of condensation process for publications."""

import argparse
import os
import logging

import numpy as np

from utilities import parse_keys
from utilities import make_tensor


logging.basicConfig(level=logging.DEBUG)


def extract_point_clouds(data, parsed_keys, prefix, out_dir):
    """Extract point clouds and store them as text files."""
    assert 'data' in parsed_keys, 'Require "data" key'

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]

    n_digits = int(np.log10(T) + 1)

    for t in range(T):
        out = os.path.join(
            out_dir, prefix + f'_t{t:0{n_digits}d}.txt'
        )

        logging.info(f'Storing point cloud in {out}...')

        np.savetxt(
            out,
            X[:, :, t],
            fmt='%.8f',
            delimiter='\t',
            header='x\ty',
            comments=''
        )


def extract_diffusion_homology(data, parsed_keys, prefix, out_dir):
    """Extract diffusion homology pairs."""
    assert 'diffusion_homology_persistence_pairs' in parsed_keys, \
        'Require "diffusion_homology_persistence_pairs" key'

    assert 'data' in parsed_keys, 'Require "data" key'

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]
    pd = data['diffusion_homology_persistence_pairs']

    out = os.path.join(
        out_dir, prefix + '_intrinsic_diffusion_homology.txt'
    )

    logging.info(f'Storing intrinsic diffusion homology in {out}...')

    with open(out, 'w') as f:
        for i, (c, d) in enumerate(pd):
            print(f'{c:.8f}\t{i}\n'
                  f'{d:.8f}\t{i}\n', file=f)

    total_persistence = [
        np.sum(np.diff(pd[pd[:, 1] <= t])) for t in range(T)
    ]

    P = np.max(total_persistence)

    out = os.path.join(
        out_dir, prefix + '_total_persistence.txt'
    )

    logging.info(f'Storing total persistence curve in {out}...')

    pd = [
        (t, pers / P) for t, pers in zip(range(T), total_persistence)
    ]

    np.savetxt(
        out,
        pd,
        fmt='%.8f',
        delimiter='\t',
        header='time\ttotal_persistence',
        comments=''
    )

    out = os.path.join(
        out_dir, prefix + '_betti.txt'
    )

    betti = data['diffusion_homology_betti']

    logging.info(f'Storing Betti curve in {out}...')

    np.savetxt(
        out,
        betti,
        fmt='%.8f',
        delimiter='\t',
        header='time\tbetti_number',
        comments=''
    )


def extract_persistence_points(data, parsed_keys, prefix, out_dir):
    """Extract persistence points."""
    assert 'persistence_points' in parsed_keys, \
        'Require "persistence_points" key'

    persistence_diagrams = [
        data[key] for key, _ in parsed_keys['persistence_points']
    ]

    # TODO: restrict to other dimensions?
    persistence_diagrams = [d[d[:, 2] == 1] for d in persistence_diagrams]
    T = len(persistence_diagrams)
    n_digits = int(np.log10(T) + 1)

    for t, pd in zip(range(T), persistence_diagrams):
        out = os.path.join(
            out_dir, prefix + f'_persistence_diagram_t{t:0{n_digits}d}.txt'
        )

        logging.info(f'Storing persistence diagram in {out}...')

        pd = pd[:, 0:2]

        if len(pd) == 0:
            pd = [[0, 0]]

        np.savetxt(
            out,
            pd,
            fmt='%.8f',
            delimiter='\t',
            header='creation\tdestruction',
            comments=''
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    parser.add_argument(
        '-o', '--output',
        default='/tmp',
        type=str,
        help='Output directory'
    )

    args = parser.parse_args()

    # Load data and check whether all keys are available. We require
    # only the diffusion homology pairs and the data set here.
    data = np.load(args.INPUT, allow_pickle=True)
    parsed_keys = parse_keys(data)

    prefix = os.path.basename(args.INPUT)
    prefix = os.path.splitext(prefix)[0]

    extract_point_clouds(data, parsed_keys, prefix, args.output)
    extract_diffusion_homology(data, parsed_keys, prefix, args.output)

    try:
        extract_persistence_points(data, parsed_keys, prefix, args.output)
    except AssertionError:
        pass
