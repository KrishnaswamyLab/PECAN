"""Visualise output of condensation process."""

import argparse

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    args = parser.parse_args()

    data = np.load(args.INPUT)

    fig, ax = plt.subplots(ncols=2)

    X = []
    for key in data.keys():
        if key.startswith('t'):
            X.append(data[key])

    x_min, x_max, y_min, y_max = get_limits(X)

    scatter = ax[0].scatter(X[0][:, 0], X[1][:, 1])
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(X),
    )

    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))

    pd = data['D']

    ax[1].scatter(pd[:, 0], pd[:, 1])

    plt.show()
