from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/PECAN/pecan")
from ripser import Ripser
from kernels import get_kernel_fn, alpha_decaying_kernel
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def knn_kernel(X, k):
    # Create a NearestNeighbors object
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)

    # Get the k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    # Construct the adjacency matrix
    n_samples = len(X)
    adjacency_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        adjacency_matrix[i, indices[i]] = 1
        
    return adjacency_matrix

## DEPRECATED. NOW using appendix A of coifman 2006.
def diffusion_map(P, dim=10, tolerance=1e-10):
    """
    P the diffusion operator (D^{-1}K)
    or L to make sure the eigenvectors are real-valued.
    dim the dim of the diffusion map (number of eigenvectors)
    """
    # Find the 2 largest eigenvalues and their corresponding eigenvectors
    eigenvalues, eigenvectors = eigs(P, k=dim, which='LM')
    return check_real(eigenvalues, tolerance), check_real(eigenvectors, tolerance)
    # return check_real(eigenvalues * eigenvectors, tolerance)

## DEPRECATED.
# def diffusion_map2(P, dim=10):
#     """
#     P the diffusion operator (D^{-1}K)
#     or L to make sure the eigenvectors are real-valued.
#     dim the dim of the diffusion map (number of eigenvectors)
#     """
#     # Find the 2 largest eigenvalues and their corresponding eigenvectors
#     # eigenvalues, eigenvectors = eigs(P, k=dim, which='LM')
#     svd = TruncatedSVD(n_components=dim)
#     svd.fit(P)
#     eigenvalues = svd.singular_values_
#     eigenvectors = svd.components_.T
#     return eigenvalues, eigenvectors
#     return check_real(eigenvalues * eigenvectors, tolerance)

def eigendecompose_diffusion(P, d):
    """
    P the diffusion operator (D^{-1}K)
    d the degrees (D=diag(d))
    see appendix A. in Coifman and Lafon, 2006
    """
    W = np.diag(np.sqrt(d)) @ P @ np.diag(1/np.sqrt(d))
    eigenvalues, eigenvectorsW = eigh(W)
    eigenvectors = np.diag(1/np.sqrt(d)) @ eigenvectorsW
    return eigenvalues, eigenvectors


def check_real(array, tolerance=1e-10):
    """
    Efficiently checks if all elements in a 2D array are real within a specified tolerance.
    Returns a new array with the real parts of the elements if they are real.
    
    Parameters:
        array (numpy.ndarray): 2D array to check.
        tolerance (float): The tolerance within which the imaginary part should be zero.
    
    Returns:
        numpy.ndarray: A new array containing the real parts of the elements.
    
    Raises:
        ValueError: If any element in the array is not real within the specified tolerance.
    """
    
    # Check if all elements are real within the tolerance
    if np.all(np.abs(array.imag) <= tolerance):
        return array.real
    else:
        raise ValueError("Some elements in the array are not real within the specified tolerance.")

def visuallize_diffusion_map(X, P, t, dim=10, tolerance=1e-10, name='Dataset', epsilon=1):
    lam, Xd = diffusion_map(P)
    Xd = lam ** t * Xd
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(Xd)

    # Create a figure with 1 row and 2 columns of subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first dataset on the first subplot
    axs[0].scatter(X[:, 0], X[:, 1])
    axs[0].set_title(name)

    # Plot the second dataset (diffusion map) on the second subplot
    axs[1].scatter(Xpca[:, 0], Xpca[:, 1])
    axs[1].set_title(f'Diffusion Map $\epsilon=${epsilon}')

    plt.show()

"""Data set generator methods."""

import numpy as np


def spiral(N, **kwargs):
    A=1
    omega=1
    phi=0
    t_min=0
    t_max=10
    """
    Generate a dataset of a spiral.

    Parameters:
    - N: Number of points
    - A: Amplitude
    - omega: Angular frequency
    - phi: Phase
    - t_min: Minimum time value
    - t_max: Maximum time value

    Returns:
    - X: 2D array of shape (N, 2), where each row is a point [x(t), y(t)]
    - t: 1D array of time values
    """
    # Generate time values
    t = np.linspace(t_min, t_max, N)
    
    # Generate the spiral points using the parametric equations
    x = A * t * np.cos(omega * t + phi)
    y = A * t * np.sin(omega * t + phi)
    
    # Combine x and y coordinates into a single 2D array
    X = np.column_stack((x, y))
    
    return X, t

def simplex(N, **kwargs):
    """Generate set of equidistant points."""
    X = []
    C = np.linspace(0, 1, N)

    for i in range(N - 1):
        x = np.zeros((N - 1))
        x[i] = 1

        X.append(x)

    # Last point is a little bit icky...
    X.append((np.ones((N - 1)) - np.sqrt(N)) / (N - 1))

    # Make everything uniform and scale distances
    X = np.asarray(X)
    X /= np.sqrt(2)

    return X, C


def blobs(N, **kwargs):
    """Generate set of Gaussian blobs."""
    from sklearn.datasets import make_blobs

    return make_blobs(N, random_state=kwargs["random_state"])


def moons(N, **kwargs):
    """Generate moons data set with labels."""
    from sklearn.datasets import make_moons

    return make_moons(N, random_state=kwargs["random_state"])


def nested_circles(N, **kwargs):
    """Generate nested circles with labels."""
    from sklearn.datasets import make_circles

    return make_circles(N, random_state=kwargs["random_state"])


def barbell(N, beta=1, **kwargs):
    """Generate uniformly-sampled 2-D barbelll with colours."""
    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    X = []
    C = []
    k = 1

    while k <= N:
        x = (2 + beta / 2) * np.random.uniform()
        y = (2 + beta / 2) * np.random.uniform()

        n_prev = len(C)

        if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25:
            C.append(0)

        elif abs(x - 1 - beta / 4) < beta / 4 and abs(y - 0.5) < 0.125:
            C.append(1)

        elif (x - 1.5 - beta / 2) ** 2 + (y - 0.5) ** 2 <= 0.25:
            C.append(2)

        if len(C) > n_prev:
            X.append((x, y))
            k += 1

    return np.asarray(X), np.asarray(C)


def double_annulus(N, **kwargs):
    """Sample N points from a double annulus."""
    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    X = []
    C = []
    for i in range(N):
        while True:
            t = [np.random.uniform(-50, 50, 1)[0], np.random.uniform(-50, 140, 1)[0]]

            d = np.sqrt(np.dot(t, t))
            if d <= 50 and d >= 20:
                X.append(t)
                C.append(0)
                break

            d = np.sqrt(t[0] ** 2 + (t[1] - 90) ** 2)
            if d <= 50 and d >= 40:
                X.append(t)
                C.append(1)
                break

    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.asarray(X), np.asarray(C)


def annulus(N, r, R, **kwargs):
    """Sample points from annulus.

    This function samples `N` points from an annulus with inner radius `r`
    and outer radius `R`.

    Parameters
    ----------
    N : int
        Number of points to sample

    r : float
        Inner radius of annulus

    R : float
        Outer radius of annulus

    **kwargs:
        Optional keyword arguments, such as a fixed random state for the
        pseudo-random number generator.

    Returns
    -------
    Array of (x, y) coordinates.
    """
    if r >= R:
        raise RuntimeError("Inner radius must be less than or equal to outer radius")

    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    thetas = np.random.uniform(0, 2 * np.pi, N)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(np.random.uniform(r ** 2, R ** 2, N))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return X, np.linspace(0, 1, N)


def hyperuniform_circle(N, **kwargs):
    """Generate hyperuniformly-Sampled 2-D circle and colours."""
    X = []
    C = np.linspace(0, 1, N)

    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

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

        e = np.sqrt(1.0 - a ** 2 / b ** 2)
        tot_size = special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / N
        arcs = np.arange(N) * arc_size
        res = optimize.root(lambda x: (special.ellipeinc(x, e) - arcs), angles)
        angles = res.x

        arcs = special.ellipeinc(angles, e)

    for t in angles:
        X.append((a * np.cos(t), b * np.sin(t)))

    return np.asarray(X), np.asarray(C)


def linked_hyperuniform_circles(N, **kwargs):
    """Generate linked hyperuniformly-sampled 2D circles and colours."""
    X = []
    C = np.linspace(0, 1, N)

    theta = np.linspace(0, 2 * np.pi, N // 2, endpoint=False)
    theta = np.concatenate((theta, theta))

    for i, t in enumerate(theta):
        if i < N // 2:
            X.append((np.cos(t), np.sin(t)))
        # Shift the second circle away so that they are only linked at
        # a single point.
        else:
            X.append((2 + np.cos(t), np.sin(t)))

    return np.asarray(X), np.asarray(C)


def petals(N, **kwargs):
    """Generate petal data set."""
    X = []  # points in respective petals
    Y = []  # auxiliary array (points on outer circle)
    C = []

    assert N > 4, "Require more than four data points"

    # Number of 'petals' to point into the data set. This is required to
    # ensure that the full space is used.
    M = int(np.floor(np.sqrt(N)))
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)

    for theta in thetas:
        Y.append(np.asarray([np.cos(theta), np.sin(theta)]))

    # Radius of the smaller cycles is half of the chord distance between
    # two 'consecutive' points on the circle.
    r = 0.5 * np.linalg.norm(Y[0] - Y[1])

    for i, x in enumerate(Y):
        for theta in thetas:
            X.append(np.asarray([r * np.cos(theta) - x[0], r * np.sin(theta) - x[1]]))

            # Indicates that this point belongs to the $i$th circle.
            C.append(i)

    return np.asarray(X), np.asarray(C)


def poisson_process(N, **kwargs):
    """Generate points based on a Poisson process."""
    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    n = np.random.poisson(N)
    X = np.random.rand(n, 2)
    C = np.linspace(0, 1, n)

    return X, C


def const_curvature_disk(N, K, **kwargs):
    """Sample points on manifolds of constant curvature."""

    if np.isclose(K, 0):
        return euclidean_disk(N, **kwargs)
    else:
        return spherical_cap(N, K, **kwargs) if K > 0 else hyperbolic_disk(N, K, **kwargs)


def euclidean_disk(N, **kwargs):
    """Sample points on a unit Euclidean disk."""

    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    u = np.random.uniform(0, 1, N)
    r = np.sqrt(u)
    theta = np.random.uniform(0, 2 * np.pi, N)
    x = np.multiply(r, np.cos(theta))
    y = np.multiply(r, np.sin(theta))
    C = np.linspace(0, 1, N)

    return np.vstack([x, y]).T, C


def spherical_cap(N, K, **kwargs):
    """Sample points from a unit disk on the surface of a sphere with const. positive curvature K."""

    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    assert K > 0

    u = np.random.uniform(0, 1, N)
    r = np.multiply(np.sqrt(u), np.sin(np.sqrt(K) / 2.0))
    r = np.multiply(2.0 / np.sqrt(K), np.arcsin(r))
    theta = np.random.uniform(0, 2 * np.pi, N)
    x = np.multiply(r, np.cos(theta))
    y = np.multiply(r, np.sin(theta))
    C = np.linspace(0, 1, N)

    return np.vstack([x, y]).T, C


def hyperbolic_disk(N, K, **kwargs):
    """Sample points from a hyperbolic disk (using the Poincaré metric) with const. negative curvature K."""

    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    assert K < 0

    u = np.random.uniform(0, 1, N)
    r = np.multiply(np.sqrt(u), np.sinh(np.sqrt(-K) / 2.0))
    r = np.multiply(2.0 / np.sqrt(-K), np.arcsinh(r))
    theta = np.random.uniform(0, 2 * np.pi, N)
    x = np.multiply(r, np.cos(theta))
    y = np.multiply(r, np.sin(theta))
    C = np.linspace(0, 1, N)

    return np.vstack([x, y]).T, C


def torus(N, inner_radius=3, outer_radius=10):
    """Generate n-point hollow torus with one hole."""

    # begin by generating a hollow cylinder
    # from N random uniform points, normalized so that y^2 + z^2 = inner_radius^2
    X = np.random.rand(N, 3) * 2 - 1
    X[:, 1:] /= np.sqrt(np.sum(X[:, 1:] ** 2, axis=1))[:, None] / inner_radius
    x, y, z = X.T
    # Twist the cylinder
    # first, shift the cylinder on the y axis so that 0,0,0 lies at the center of the soon-to-be constructed donut
    rod_length = outer_radius - inner_radius
    y += outer_radius
    # Transform into polar coordinates around Z axis
    r = np.sqrt(x ** 2 + y ** 2)
    theta = (
        x * 2 * np.pi
    )  # we can disregard the previous angle, which didn't mean anything
    # Back to cartesian coords
    x = np.cos(theta) * r
    y = np.sin(theta) * r

    # Recombine the variables
    X = np.column_stack((x, y, z))
    return X, None


def sphere(N, dim=3):
    """Generate N-point sphere with specified dimension. A wrapper for tadasets

    Args:
        N (int): Number of points
        dim (int, optional): Dimension of sphere. Defaults to 3.
    """
    import tadasets

    dsphere = tadasets.dsphere(
        n=N, d=dim - 1, r=1
    )  # TODO: Can also specify the amount of noise to add to the sphere
    return dsphere, None

# Function to generate the Swiss roll
def generate_swiss_roll(N_t, N_w, t_min=0, t_max=10, w_min=-1, w_max=1, omega=1):
    t = np.linspace(t_min, t_max, N_t)
    w = np.linspace(w_min, w_max, N_w)
    t, w = np.meshgrid(t, w)
    t = t.flatten()
    w = w.flatten()
    x = t * np.cos(omega * t)
    y = w
    z = t * np.sin(omega * t)
    X = np.column_stack((x, y, z))
    return X

# Function to generate the Swiss roll
def generate_swiss_roll_rand(N, t_min=0, t_max=10, w_min=-1, w_max=1, omega=1, seed=42):
    np.random.seed(seed)
    # t = np.random.uniform(t_min, t_max, N_t)
    # w = np.random.uniform(w_min, w_max, N_w)
    # t, w = np.meshgrid(t, w)
    # t = t.flatten()
    # w = w.flatten()
    tw = np.random.uniform([0, w_min], [np.square(t_max - t_min), w_max], [N, 2])
    tw[:, 0] = np.sqrt(tw[:, 0]) + t_min
    t, w = tw[:, 0], tw[:, 1]
    x = t * np.cos(omega * t)
    y = w
    z = t * np.sin(omega * t)
    X = np.column_stack((x, y, z))
    return X


def make_hole(X, hole_center, hole_radius):
    # Remove points that are within `hole_radius` of the hole_center
    distances_from_center = np.linalg.norm(X - hole_center, axis=1)
    X = X[distances_from_center > hole_radius]
    return X

def swiss_roll_hole(N, **kwargs):
    X = generate_swiss_roll_rand(N=1000, t_min=0, t_max=4 * np.pi, w_min=-10, w_max=5, omega=1)
    omega=1
    t = 3 * np.pi
    w = -3
    hole_center = [t * np.cos(omega * t), w, t * np.sin(omega * t)]
    X = make_hole(X, hole_center, 3)
    return X, None

def moon_petals(N, degree=np.pi, R=1, **kwargs):
    """Generate moon petal data set."""
    X = []  # points in respective petals
    Y = []  # auxiliary array (points on outer circle)
    C = []

    assert N > 4, "Require more than four data points"

    # Number of 'petals' to point into the data set. This is required to
    # ensure that the full space is used.
    M = int(np.floor(np.sqrt(N)))
    thetas_outer = np.linspace(0, degree, M, endpoint=False)
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)


    for theta in thetas_outer:
        Y.append(np.asarray([np.cos(theta), np.sin(theta)]))

    # Radius of the smaller cycles is half of the chord distance between
    # two 'consecutive' points on the circle.
    r = 0.5 * np.linalg.norm(Y[0] - Y[1])

    for i, x in enumerate(Y):
        for theta in thetas:
            X.append(np.asarray([r * np.cos(theta) - x[0], r * np.sin(theta) - x[1]]))

            # Indicates that this point belongs to the $i$th circle.
            C.append(i)

    return np.asarray(X), np.asarray(C)

def sprial_petals(turns=2, a=0.1, base_delta_theta=0.1 * 2 * np.pi, points_per_unit_radius=100):
    """
    Generate points forming tangent circles along a spiral with even distribution of spiral points.
    
    Parameters:
    - turns: Number of turns for the spiral.
    - a: Parameter controlling the distance between successive turns of the spiral.
    - base_delta_theta: Base angular increment for the spiral points at theta = 2*pi.
    - points_per_unit_radius: Number of points per unit radius for the circle.
    
    Returns:
    - A numpy array containing the points forming the tangent circles.
    """
    
    # Generate points on the spiral with varying delta_theta
    def generate_even_spiral_points(turns, a, base_delta_theta):
        theta = 0.0
        spiral_points = []
        
        while theta < 2 * np.pi * turns:
            x = a * theta * np.cos(theta)
            y = a * theta * np.sin(theta)
            spiral_points.append([x, y])

            # Adjust delta_theta to be inversely proportional to theta, but ensure it's never zero
            delta_theta = base_delta_theta / (1 + theta / (2 * np.pi))
            theta += delta_theta
        
        return np.array(spiral_points)

    # Generate points on a circle
    def generate_circle_points(center, r, num_points):
        thetas = np.linspace(0, 2 * np.pi, num_points)
        x = r * np.cos(thetas) + center[0]
        y = r * np.sin(thetas) + center[1]
        return np.column_stack((x, y))
    
    spiral_points = generate_even_spiral_points(turns, a, base_delta_theta)
    circle_points_list = []

    for i in range(1, len(spiral_points) - 1):
        prev_point = spiral_points[i - 1]
        curr_point = spiral_points[i]
        next_point = spiral_points[i + 1]

        # Calculate radius as half the average distance to adjacent points
        r = 0.25 * (np.linalg.norm(curr_point - prev_point) + np.linalg.norm(next_point - curr_point))

        # Number of points for this circle proportional to its radius
        num_circle_points = int(r * points_per_unit_radius)

        # Generate circle points and append to the list
        circle_points = generate_circle_points(curr_point, r, num_circle_points)
        circle_points_list.append(circle_points)

    # Combine all circle points into a single array
    all_circle_points = np.vstack(circle_points_list)
    
    return all_circle_points, None



def make_diffusion(X, alpha=1, epsilon=1, threshold=None):
    # FIXME let me recheck the distance
    K = alpha_decaying_kernel(X, alpha=alpha, epsilon=epsilon)
    if threshold is not None:
        kthr = K.max() * threshold
        K[K < kthr] = 0.
    Q = np.sum(K, axis=1)
    P = np.diag(1.0 / Q) @ K
    return P


def make_diffusion_d(X, alpha=1, epsilon=1, threshold=None):
    # FIXME let me recheck the distance
    K = alpha_decaying_kernel(X, alpha=alpha, epsilon=epsilon)
    if threshold is not None:
        kthr = K.max() * threshold
        K[K < kthr] = 0.
    Q = np.sum(K, axis=1)
    P = np.diag(1.0 / Q) @ K
    return P, Q

# def make_diffusion_sym(X, alpha=1, epsilon=1, threshold=None):
#     # FIXME let me recheck the distance
#     K = alpha_decaying_kernel(X, alpha=alpha, epsilon=epsilon)
#     if threshold is not None:
#         kthr = K.max() * threshold
#         K[K < kthr] = 0.
#     Q = np.sum(K, axis=1)
#     P = np.diag(1.0 / np.sqrt(Q)) @ K @np.diag(1.0 / np.sqrt(Q))
#     return P

def make_diffusion_box(X, epsilon=1):
    kernel_fn = get_kernel_fn("box")
    K = kernel_fn(X, epsilon=epsilon)
    Q = np.sum(K, axis=1)
    P = np.diag(1.0 / Q) @ K
    return P

def make_diffusion_box_d(X, epsilon=1):
    kernel_fn = get_kernel_fn("box")
    K = kernel_fn(X, epsilon=epsilon)
    Q = np.sum(K, axis=1)
    P = np.diag(1.0 / Q) @ K
    return P, Q

# def make_diffusion_sym_box(X, epsilon=1):
#     kernel_fn = get_kernel_fn("box")
#     K = kernel_fn(X, epsilon=epsilon)
#     Q = np.sum(K, axis=1)
#     P = np.diag(1.0 / np.sqrt(Q)) @ K @np.diag(1.0 / np.sqrt(Q))
#     return P


def make_diffusion_knn(X, k=10):
    K = knn_kernel(X, k=k)
    Q = np.sum(K, axis=1)
    P = np.diag(1.0 / Q) @ K
    return P

def make_diffusion_knn_d(X, k=10):
    K = knn_kernel(X, k=k)
    Q = np.sum(K, axis=1)
    P = np.diag(1.0 / Q) @ K
    return P, Q

# def make_diffusion_sym_knn(X, k=10):
#     K = knn_kernel(X, k=k)
#     Q = np.sum(K, axis=1)
#     P = np.diag(1.0 / np.sqrt(Q)) @ K @ np.diag(1.0 / np.sqrt(Q))
#     return P


def visuallize_diffusion_map_subplot(X, P, ax, dim=10, tolerance=1e-10, name='Dataset', epsilon=1, alpha=1, t=1, filter_func=None):
    lam, Xd = diffusion_map(P)  # Assuming diffusion_map is a function you've defined
    if filter_func is None:
        Xd = lam ** t * Xd
    else:
        Xd = filter_func(lam) * Xd
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(Xd)

    # Plot the diffusion map on the specified subplot
    ax.scatter(Xpca[:, 0], Xpca[:, 1])
    ax.set_title(f'Diffusion Map $\epsilon={np.round(epsilon, 1)}$,$\\alpha={np.round(alpha, 0)}$, t={np.round(t, 0)}')

def visuallize_diffusion_map_subplot_correct(X, P, d, ax, name='Dataset', epsilon=1, alpha=1, t=1, filter_func=None, s=1):
    lam, Xd = eigendecompose_diffusion(P, d)  # Assuming diffusion_map is a function you've defined
    if filter_func is None:
        Xd = lam ** t * Xd
    else:
        Xd = filter_func(lam) * Xd
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(Xd)

    # Plot the diffusion map on the specified subplot
    ax.scatter(Xpca[:, 0], Xpca[:, 1], s=s)
    ax.set_title(f'Diffusion Map $\epsilon={np.round(epsilon, 2)}$,$\\alpha={np.round(alpha, 0)}$, t={np.round(t, 0)}')

from sklearn.metrics.pairwise import euclidean_distances

def calculate_lawvere_homology_logp(P):
    # in the code it is 
    # A_comp = 1 - np.diag(np.sqrt(1/d)) @ K @ np.diag(np.sqrt(1/d))
    A = - np.log(P)
    tuples, points = Ripser(dimension=1)(A)
    # tuples_comp, points_comp = Ripser(dimension=1)(A_comp)
    # Add additional information about the dimension of each
    # topological feature.
    dimension = np.asarray([len(c) - 1 for c, _ in tuples])

    # Adds the dimension as an additional column, turning the 2D
    # points of the diagram into 3D points.
    points = np.column_stack((points, dimension))
    return points

def calculate_lawvere_homology_diffu_dist(P, t=1):
    # in the code it is 
    # A_comp = 1 - np.diag(np.sqrt(1/d)) @ K @ np.diag(np.sqrt(1/d))
    lam, phi = diffusion_map(P)
    Xd = lam ** t * phi
    A = euclidean_distances(Xd)
    tuples, points = Ripser(dimension=1)(A)
    # tuples_comp, points_comp = Ripser(dimension=1)(A_comp)
    # Add additional information about the dimension of each
    # topological feature.
    dimension = np.asarray([len(c) - 1 for c, _ in tuples])

    # Adds the dimension as an additional column, turning the 2D
    # points of the diagram into 3D points.
    points = np.column_stack((points, dimension))
    return points

def calculate_lawvere_homology_pecan(X, epsilon, alpha=2, t=1):
    K = alpha_decaying_kernel(X, alpha=alpha, epsilon=epsilon)
    d = K.sum(axis=1)
    # A = np.diag(np.sqrt(1/d)) @ (1 - K) @ np.diag(np.sqrt(1/d)) # why? also, inconsistent w/ in code wrt normalization.
    # in the code it is 
    A = 1 - np.diag(np.sqrt(1/d)) @ K @ np.diag(np.sqrt(1/d))
    tuples, points = Ripser(dimension=1)(A)
    # tuples_comp, points_comp = Ripser(dimension=1)(A_comp)
    # Add additional information about the dimension of each
    # topological feature.
    dimension = np.asarray([len(c) - 1 for c, _ in tuples])

    # Adds the dimension as an additional column, turning the 2D
    # points of the diagram into 3D points.
    points = np.column_stack((points, dimension))
    return points

def calculate_vr_homology(X):
    A = euclidean_distances(X)
    tuples, points = Ripser(dimension=1)(A)
    # tuples_comp, points_comp = Ripser(dimension=1)(A_comp)
    # Add additional information about the dimension of each
    # topological feature.
    dimension = np.asarray([len(c) - 1 for c, _ in tuples])

    # Adds the dimension as an additional column, turning the 2D
    # points of the diagram into 3D points.
    points = np.column_stack((points, dimension))
    return points

def plot_three_dataset(epsilon, X):
    t = 10.
    # Assuming petals is a function you've defined
    P, d = make_diffusion_d(X, alpha=10, epsilon=epsilon)  # Assuming make_diffusion is a function you've defined

    # Create a new figure with 1 row and 2 columns of subplots
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # First subplot: old right-hand side of visuallize_diffusion_map
    visuallize_diffusion_map_subplot_correct(X, P, d, ax=axs[0], epsilon=epsilon, t=t)

    # Calculate Lawvere homology points (replace with your own function)
    points = calculate_lawvere_homology_logp(P)  # Assuming calculate_lawvere_homology_logp is a function you've defined
    visualize_lawvere_homology(points, axs[1], epsilon=epsilon, name='$-\log p $')

    # Calculate Lawvere homology points (replace with your own function)
    points = calculate_lawvere_homology_diffu_dist(P)  # Assuming calculate_lawvere_homology_logp is a function you've defined
    visualize_lawvere_homology(points, axs[2], epsilon=epsilon, name='diffusion distance')

    points = calculate_lawvere_homology_pecan(X, epsilon=epsilon, alpha=2, t=t)
    visualize_lawvere_homology(points, axs[3], epsilon=epsilon, name='old PECAN')

    plt.tight_layout()
    plt.show()

def visualize_lawvere_homology(points, ax, epsilon=1., name='', s=1):
    ax.set_xlim(points[:, :-1].min() * 0.9, points[:, :-1].max() * 1.1)
    ax.set_ylim(points[:, :-1].min() * 0.9, points[:, :-1].max() * 1.1)
    ax.axline((-0.1, -0.1), slope=1.0, c='k')
    ax.scatter(x=points[:, 0], y=points[:, 1], c=points[:, 2], cmap='coolwarm', s=s)
    ax.set_title(f"Lawvere Homology with {name}")

def plot_diffusions(epsilon, alpha, t, X, s=1):
    # Assuming petals is a function you've defined
    P, d = make_diffusion_d(X, alpha=alpha, epsilon=epsilon)  # Assuming make_diffusion is a function you've defined

    # Create a new figure with 1 row and 2 columns of subplots
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # First subplot: old right-hand side of visuallize_diffusion_map
    visuallize_diffusion_map_subplot_correct(X, P, d, ax=axs[0], epsilon=epsilon, t=t, s=s)

    # Calculate Lawvere homology points (replace with your own function)
    points = calculate_vr_homology(X)  # Assuming calculate_lawvere_homology_logp is a function you've defined
    visualize_lawvere_homology(points, axs[1], epsilon=0., name='vietoris-rips', s=s)

    # Calculate Lawvere homology points (replace with your own function)
    points = calculate_lawvere_homology_diffu_dist(P)  # Assuming calculate_lawvere_homology_logp is a function you've defined
    visualize_lawvere_homology(points, axs[2], epsilon=epsilon, name='diffusion distance', s=s)

    points = calculate_lawvere_homology_pecan(X, epsilon=epsilon, alpha=alpha, t=t)
    visualize_lawvere_homology(points, axs[3], epsilon=epsilon, name='old PECAN', s=s)

    plt.tight_layout()
    plt.show()

def sinewave_petals(length=8, A=2, B=1, C=0, delta_t=0.3, points_per_unit_radius=50):
    """
    Generate points forming tangent circles along a sine wave in the xy-plane with constant delta_t.
    
    Parameters:
    - length: Length of the sine wave along x-axis.
    - A: Amplitude of the sine wave.
    - B: Determines the period of the wave.
    - C: Phase shift of the sine wave.
    - delta_t: Constant increment for t.
    - points_per_unit_radius: Number of points per unit radius for the circle.
    
    Returns:
    - A numpy array containing the points forming the tangent circles.
    """
    
    # Generate points on the sine wave with constant delta_t
    def generate_even_sine_wave_points(length, A, B, C, delta_t):
        t_values = np.arange(0, length, delta_t)
        x_values = t_values
        y_values = A * np.sin(B * t_values + C)
        return np.column_stack((x_values, y_values))

    # Generate points on a circle
    def generate_circle_points(center, r, num_points):
        thetas = np.linspace(0, 2 * np.pi, num_points)
        x = r * np.cos(thetas) + center[0]
        y = r * np.sin(thetas) + center[1]
        return np.column_stack((x, y))
    
    sine_wave_points = generate_even_sine_wave_points(length, A, B, C, delta_t)
    circle_points_list = []

    for i in range(1, len(sine_wave_points) - 1):
        prev_point = sine_wave_points[i - 1]
        curr_point = sine_wave_points[i]
        next_point = sine_wave_points[i + 1]

        # Calculate radius as half the average distance to adjacent points
        r = 0.25 * (np.linalg.norm(curr_point - prev_point) + np.linalg.norm(next_point - curr_point))

        # Number of points for this circle proportional to its radius
        num_circle_points = int(r * points_per_unit_radius)

        # Generate circle points and append to the list
        circle_points = generate_circle_points(curr_point, r, num_circle_points)
        circle_points_list.append(circle_points)

    # Combine all circle points into a single array
    all_circle_points = np.vstack(circle_points_list)
    
    return all_circle_points, None