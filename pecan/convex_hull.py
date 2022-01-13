"""Calculate convex hull of input point cloud.

This script is used to visualise the convex hull of an input point
cloud, subject to a diffusion condensation process. It also provides the
coordinates of the two points responsible for 'creating' the diameter of
the point cloud.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from sklearn.metrics import pairwise_distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', help='Input file(s)', nargs='+')

    args = parser.parse_args()

    for filename in args.INPUT:
        X = pd.read_csv(filename, sep='\t').to_numpy()
        try:
            hull = ConvexHull(X)

        # Don't do anything for these files
        except QhullError:
            continue

        vertices = hull.vertices

        print('\n', filename, file=sys.stderr)

        # Generate lines for the convex hull (in TikZ format)
        output = '\\draw '
        for x, y in zip(vertices, vertices[1:]):
            output += f'({x:.02f},{y:.02f}) -- '

        output += 'cycle;'
        print(output)

        # Find diameter
        dist = pairwise_distances(X)
        index = np.unravel_index(np.argmax(dist), dist.shape)

        a, b = X[index, :]

        diam_line = '\\draw '
        diam_line += f'({a[0]:.02f},{a[1]:.02f}) -- '
        diam_line += f'({b[0]:.02f},{b[1]:.02f});'
        print(diam_line)
