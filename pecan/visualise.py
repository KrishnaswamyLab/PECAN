"""Visualise output of condensation process."""

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


    # Figure out all intervals to draw here
    
    values = [destruction for _, destruction in pd if destruction <= T[i]]

    segments = [
        [(0, i), (destruction, i)] for i, destruction in enumerate(values)
    ]

    barcode.set_segments(segments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    args = parser.parse_args()

    data = np.load(args.INPUT)

    fig, ax = plt.subplots(ncols=2)

    # Store all times and all variants of the data set. This makes the
    # visualisation easier.
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

    pd = data['D']

    ax[1].set_xlim(0, np.max(pd[:, 1]))
    ax[1].set_ylim(0, len(pd[:, 1]))

    barcode = matplotlib.collections.LineCollection(segments=[])
    ax[1].add_collection(barcode)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(X),
    )

    plt.show()
