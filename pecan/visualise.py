"""Visualise output of condensation process."""

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np


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

    scatter = ax[0].scatter(X[0][:, 0], X[1][:, 1])
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(X),
    )

    ax[0].set_xlim((-1.5, 1.5))
    ax[0].set_ylim((-1.5, 1.5))

    pd = data['D']

    ax[1].scatter(pd[:, 0], pd[:, 1])

    plt.show()
