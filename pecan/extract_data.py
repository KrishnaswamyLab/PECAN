"""Extract output of condensation process for publications."""

import argparse
import os
import logging

import numpy as np

from utilities import parse_keys
from utilities import make_tensor
from utilities import get_limits


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

    args = parser.parse_args()

    # Load data and check whether all keys are available. We require
    # only the diffusion homology pairs and the data set here.
    data = np.load(args.INPUT, allow_pickle=True)
    parsed_keys = parse_keys(data)

    prefix = os.path.basename(args.INPUT)
    prefix = os.path.splitext(prefix)[0]

    extract_point_clouds(data, parsed_keys, prefix, '/tmp')
    extract_diffusion_homology(data, parsed_keys, prefix, '/tmp')
    extract_persistence_points(data, parsed_keys, prefix, '/tmp')

    raise 'heck'

    X = make_tensor(data, parsed_keys['data'])
    T = X.shape[-1]

    for t in range(T):
        np.savetxt(
            f'/tmp/Double_annulus_{t:02d}.txt',
            X[:, :, t],
            fmt='%.8f',
            delimiter=',',
            header='x,y',
            comments=''
        )

    pd = data['diffusion_homology_persistence_pairs']

    total_persistence = [
        np.sum(np.diff(pd[pd[:, 1] <= t])) for t in range(T)
    ]

    P = np.max(total_persistence)

    #for t, pers in zip(range(T), total_persistence):
    #    print(f"{t}, {pers / P:.4f}")

    persistence_diagrams = [
        data[key] for key, _ in parsed_keys['persistence_points']
    ]

    persistence_diagrams = [d[d[:, 2] == 1] for d in persistence_diagrams]

    print('time,creation,destruction')
    for t, pd in zip(range(T), persistence_diagrams):
        if len(pd):
            for x, y, d in pd:
                print(f'{t},{x:.4f},{y:.4f}')
        else:
            print(f'{t},0,0')

    #for i, (c, d) in enumerate(pd):
    #    print(f'{c}, {i}')
    #    print(f'{d}, {i}')
    #    print('')

    # Set up the diffusion homology pairing. This just amounts to
    # accessing the distribution of pairs and turning them into a
    # barcode visualisation.

    # List of persistence pairs of the form (creation, destruction). An
    # optional third dimension is ignored.

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
