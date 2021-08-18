"""Analyse configurations in general position."""

import itertools

import numpy as np


if __name__ == '__main__':
    points = [
        (0, 0),
        (1, 0),
        (1, 0.5),
        (0, np.sqrt(1.0625))
    ]

    distances = []

    for p, q in itertools.combinations(points, 2):
        p = np.asarray(p)
        q = np.asarray(q)
        distance = np.linalg.norm(p - q, ord=2)

        distances.append(distance)

    print(f'No. distances:  {len(distances)}')
    print(f'No. unique distances: {len(np.unique(distances))}')

    for p, q in itertools.combinations(points, 2):
        p = np.asarray(p)
        q = np.asarray(q)
        midpoint = 0.5 * (p + q)

        print(f'Midpoint: {midpoint}')

        for r in points:
            r = np.asarray(r)

            distance = np.linalg.norm(midpoint - r, ord=2)

            if distance in distances:
                print(f'\tDistance collision: {distance:.2f}')
