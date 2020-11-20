"""Analyse topology of the condensation process."""

import argparse

import matplotlib.collections
import matplotlib.lines
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np


def get_limits(data):
    x = np.asarray([X[:, 0] for X in data]).flatten()
    y = np.asarray([X[:, 1] for X in data]).flatten()

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    return x_min, x_max, y_min, y_max


def update(i):
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

    data = np.load(args.INPUT)

    fig, ax = plt.subplots(ncols=2)

    X = []
    T = []

    for key in data.keys():
        if key.startswith('t'):
            X.append(data[key])
            T.append(int(key.split('_')[1]))

    x_min, x_max, y_min, y_max = get_limits(X)

    scatter = ax[0].scatter(X[0][:, 0], X[1][:, 1])

    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))

    # The persistence diagram is always scaled to [0,1] x [0,1].
    ax[1].set_xlim(-0.1, 1.1)
    ax[1].set_ylim(-0.1, 1.1)
    persistence_diagram = ax[1].scatter(x=[], y=[])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(X),
        repeat=args.repeat,
        interval=args.interval,
    )

    plt.show()
