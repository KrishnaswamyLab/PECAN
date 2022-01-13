"""Calculate convex hull of input point cloud.

This script is used to visualise the convex hull of an input point
cloud, subject to a diffusion condensation process. It also provides the
coordinates of the two points responsible for 'creating' the diameter of
the point cloud.
"""

import argparse

import pandas as pd

from scipy.spatial import ConvexHull


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', help='Input file')

    args = parser.parse_args()

    X = pd.read_csv(args.INPUT, sep='\t').to_numpy()
    hull = ConvexHull(X)
    vertices = hull.vertices

    # Generate lines for the convex hull (in TikZ format)

    output = '\\draw '
    for x, y in zip(vertices, vertices[1:]):
        output += f'({x:.02f},{y:.02f}) -- '

    output += 'cycle;'
    print(output)
