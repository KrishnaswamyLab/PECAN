"""Calculate convex hull of input point cloud.

This script is used to visualise the convex hull of an input point
cloud, subject to a diffusion condensation process. It also provides the
coordinates of the two points responsible for 'creating' the diameter of
the point cloud.
"""

import argparse
import sys

import pandas as pd

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


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

        print(filename, file=sys.stderr)

        # Generate lines for the convex hull (in TikZ format)

        output = '\\draw '
        for x, y in zip(vertices, vertices[1:]):
            output += f'({x:.02f},{y:.02f}) -- '

        output += 'cycle;'
        print(output)
