"""Utility functions and classes."""

import collections
import re

import numpy as np


def get_limits(X):
    """Calculate plotting limits of an input tensor.

    This auxiliary function calculates the 2D plotting limits of an
    input tensor.

    Parameters
    ----------
    X : np.array
        Input tensor. Only the first two dimensions of the tensor will
        be considered.

    Returns
    -------
    Tuple of `x_min`, `x_max`, `y_min`, `y_max`, which can be used to
    restrict the extents of a plot.
    """
    x = np.asarray(X[:, 0, ...]).flatten()
    y = np.asarray(X[:, 1, ...]).flatten()

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    return x_min, x_max, y_min, y_max



def make_tensor(data, parsed_keys):
    """Create a tensor from a time-varying data set.

    This function takes a time-varying data set of the same (!) shape
    and turns it into a tensor whose last axis denotes the time steps
    of the process.

    Parameters
    ----------
    data : `dict` of `np.array`
        A sequence of matrices, typically originating from an `.npz`
        file that was loaded.

    parsed_keys : list of tuples
        List of `(key, t)` tuples, where `key` indicates the
        corresponding key and `t` the time step.

    Returns
    -------
    Tensor comprising all data arrays that match the supplied key, with
    an additional axis (the last one) representing time.
    """
    T = len(parsed_keys)

    if not T:
        return None

    shape = data[parsed_keys[0][0]].shape
    X = np.empty(shape=(*shape, T))

    for key, t in parsed_keys:
        X[..., t] = data[key]

    return X


def parse_keys(data):
    """Extract keys from a set of matrices.

    The goal of this function is to parse the keys of a set of matrices
    and return them for subsequent processing. Keys will be checked for
    being time-varying. If so, the time steps will be extracted as well
    and corresponding tuples will be returned.

    The output of this function is dictionary mapping the name of a key
    to a list of instances of the key, plus optional time steps::

        {
            # Time-varying
            'data': [
                ('data_t_0', 0),
                ('data_t_1', 1),
                ('data_t_2', 2),
            ],

            # Static
            'diffusion_homology_pairs': [
                ('diffusion_homology_pairs', None)
            ]
        }

    Parameters
    ----------
    data : `dict` of `np.array`
        A sequence of matrices, typically originating from an `.npz`
        file that was loaded.

    Returns
    -------
    Dictionary with parsed keys, as described in the example above.
    """
    # Parses a time-varying key. If a string matches this regular
    # expression, it is time-varying.
    re_time = r'(.*)_t_(\d+)$'

    parsed_keys = collections.defaultdict(list)

    for key in data.keys():
        m = re.match(re_time, key)
        if m:
            name = m.group(1)
            time = int(m.group(2))

            parsed_keys[name].append((key, time))
        else:
            parsed_keys[key].append((key, None))

    return parsed_keys


def generate_output_filename(args, suffix='.npz'):
    """Generate output filename string from CLI arguments."""
    result = f'{args.data}_n{args.num_samples}{suffix}'
    return result


class UnionFind:
    """An implementation of a Union--Find class.

    The class performs path compression by default. It uses integers for
    storing one disjoint set, assuming that vertices are zero-indexed.
    """

    def __init__(self, n_vertices):
        """Initialise an empty Union--Find data structure.

        Creates a new Union--Find data structure for a given number of
        vertices. Vertex indices are assumed to range from `0` to
        `n_vertices`.

        Parameters
        ----------
        n_vertices:
            Number of vertices
        """
        self._parent = [x for x in range(n_vertices)]

    def find(self, u):
        """Find and return the parent of `u` with respect to the hierarchy.

        Parameters
        ----------
        u:
            Vertex whose parent is looked up

        Returns
        -------
        Component the vertex belongs to.
        """
        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        """Merge vertex `u` into the component of vertex `v`.

        Performs a `merge()` operation. Note the asymmetry of this
        operation, as vertex `u` will be  merged into the connected
        component of `v`.

        Parameters
        ----------
        u:
            Source connected component

        v:
            Target connected component
        """
        # There is no need to adjust anything if, by some fluke, we
        # merge ourselves into our parent component.
        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        """Generate roots.

        Generator expression for returning roots, i.e. components that
        are their own parents.

        Returns
        -------
        Yields each root vertex.
        """
        # We assume that vertices are numbered contiguously from zero to
        # `n_vertices`. This simplifies identifying a vertex here.
        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex
