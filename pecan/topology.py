"""Analyse topology of the condensation process."""

import argparse

import matplotlib.collections
import matplotlib.lines
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np


# Will be updated later on by the animation. I know that this is
# horrible, but it's the easiest way :)
scatter = None


def get_limits(data):
    x = np.asarray([X[:, 0] for X in data]).flatten()
    y = np.asarray([X[:, 1] for X in data]).flatten()

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    return x_min, x_max, y_min, y_max


def update(i):
    global scatter
    if scatter is None:
        scatter = ax[0].scatter(X[0][:, 0], X[0][:, 1])
    else:
        scatter.set_offsets(X[i])

    ax[0].set_title(f'$t={T[i]}$')

    from gtda.homology import VietorisRipsPersistence

    VR = VietorisRipsPersistence(
            homology_dimensions=[1],
            infinity_values=1.0,
            reduced_homology=False,
    )
    diagrams = VR.fit_transform([X[i]])

    # We only have a single set of homology features anyway, so there's
    # no need to select anything here.
    diagram = diagrams[0][:, 0:2]
    persistence_diagram.set_offsets(diagram)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')
    parser.add_argument('-r', '--repeat', action='store_true')
    parser.add_argument('-i', '--interval', type=int, default=200)

    args = parser.parse_args()

    data = np.load(args.INPUT, allow_pickle=True)

    fig, ax = plt.subplots(ncols=2, figsize=(6,3))

    X = []
    T = []

    for key in data.keys():
        if key.startswith('t'):
            X.append(data[key])
            T.append(int(key.split('_')[1]))

    x_min, x_max, y_min, y_max = get_limits(X)

    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))

    # The persistence diagram is always scaled to [0,1] x [0,1].
    ax[1].set_xlim(-0.1, 1.1)
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].axline((-0.1, -0.1), slope=1.0, c='k')
    persistence_diagram = ax[1].scatter(x=[], y=[])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(X),
        repeat=args.repeat,
        interval=args.interval,
    )

    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')

    pairs = []
    points = []

    for key in data.keys():
        if key.startswith('pairs'):
            pairs.append(data[key])
        elif key.startswith('points'):
            points.append(data[key])

    for index, values in enumerate(points):
        x = values[:, 0]
        y = values[:, 1]
        z = [index] * len(x)

        ax3.scatter(x, z, y)

    for index, (first, second) in enumerate(zip(pairs, pairs[1:])):
        for c1, d1 in first:
            for c2, d2 in second:
                if c1 == c2:
                    print(index, '--', index + 1, 'C', c1, d1, c2, d2)
                elif d1 == d2:
                    print(index, '--', index + 1, 'D', c1, d1, c2, d2)

    plt.show()
