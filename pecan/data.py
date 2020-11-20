"""Data set generator methods."""

import numpy as np


def moons(N, random_state=None, **kwargs):
    """Generate moons data set with labels."""
    from sklearn.datasets import make_moons
    return make_moons(N, random_state=42)


def barbell(N, beta=1, **kwargs):
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


def double_annulus(N):
    """Sample N points from a double annulus."""
    X = []
    for i in range(N):
        while True:
            t = [
                np.random.uniform(-50, 50, 1),
                np.randon.uniform(-50, 140, 1)
            ]

            d = np.sqrt(np.dot(t, t))
            if d <= 50 and d >= 20:
                X.append(t)
                break

            d = np.sqrt(t[0]**2 + (t[1] - 90)**2)
            if d <= 50 and d >= 40:
                X.append(t)
                break

    return np.asarray(X)


def hyperuniform_circle(N):
    """Generate hyperuniformly-Sampled 2-D circle and colours."""
    X = []
    C = np.linspace(0, 1, N)

    theta = np.linspace(0, 2*np.pi, N, endpoint=False)

    for t in theta:
        X.append((np.cos(t), np.sin(t)))

    return np.asarray(X), np.asarray(C)


def hyperuniform_ellipse(N, a=0.5, b=1, **kwargs):
    """Generate hyperuniformly-sampled 2-D ellipse with colours."""
    # a must be the length of minor semi-axis; b major semi-axis
    assert a < b

    X = []
    C = np.linspace(0, 1, N)

    angles = 2 * np.pi * np.arange(N) / N

    if a != b:
        # Given N points, combine scipy elliptic integral + optimize to
        # find N equidistant points along ellipse manifold, then
        # convert to angles.

        from scipy import special
        from scipy import optimize

        e = np.sqrt(1.0 - a**2 / b**2)
        tot_size = special.ellipeinc(2.0*np.pi, e)
        arc_size = tot_size/N
        arcs = np.arange(N) * arc_size
        res = optimize.root(
                lambda x: (special.ellipeinc(x, e) - arcs), angles
        )
        angles = res.x

        arcs = special.ellipeinc(angles, e)

    for t in angles:
        X.append((a*np.cos(t), b*np.sin(t)))

    return np.asarray(X), np.asarray(C)
