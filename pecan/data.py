"""Data set generator methods."""

import numpy as np


def make_barbell(N, beta=1):
    """Generate uniformly-sampled 2-D barbelll with colours."""
    X = []
    C = []
    k = 1

    while k <= N:
        x = (2 + beta/2)*np.random.uniform()
        y = (2 + beta/2)*np.random.uniform()

        X.append((x, y))
        k += 1

        if (x - 0.5)**2 + (y - 0.5)**2 <= 0.25:
            C.append(0)

        elif abs(x - 1 - beta/4) < beta/4 and abs(y - 0.5) < 0.125:
            C.append(1)

        elif (x - 1.5 - beta/2)**2 + (y - 0.5)**2 <= 0.25:
            C.append(2)

    return np.asarray(X), np.asarray(C)
