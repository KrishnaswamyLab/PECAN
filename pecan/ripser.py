"""Handle communication with Ripser."""

import itertools
import os
import tempfile
import subprocess
import warnings

import numpy as np


# Let's see whether we can find `ripser`. Notice that this does
# not guarantee that subsequent calls work as well, but we want
# to be nice to users and warn them here.
try:
    subprocess.run(
        ['ripser', '--help'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
except FileNotFoundError:
    warnings.warn(
        'The execution of `ripser` failed. Persistent homology '
        'features will be unavailable. Please install `ripser` '
        'on your system.'
    )


class Ripser:
    """Interface class for the `ripser` command-line tool."""

    def __init__(self, dimension):
        """Create interface class."""
        self.dimension = dimension

    def __call__(self, D):
        """Call `ripser` on a provided distance matrix.

        This function uses an existing `ripser` installation to handle
        the calculation of topological features. If no `ripser` binary
        can be found, the function will fail gracefully.

        Parameters
        ----------
        D : `np.array`
            Distance matrix of some metric space whose persistent
            homology should be calculated.

        Returns
        -------
        Tuple consisting of the persistence pairs (i.e. pairs of indices
        corresponding to positive and negative simplices) and points for
        the persistence diagrams (i.e. pairs of distances).

        If `ripser` fails for some reason, the function will just return
        `None, None`. This special value needs to be handled downstream.
        """
        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'w') as tmp:
                np.savetxt(tmp, D)

            try:
                result = subprocess.run(
                    ['ripser', '--dim', str(self.dimension), path],
                    stdout=subprocess.PIPE
                )
            except FileNotFoundError:
                return None, None

        finally:
            os.remove(path)

        tuples = self._parse(result.stdout.decode('utf-8'))
        points = self._make_points(tuples, D)

        return np.asarray(tuples, dtype=object), \
            np.asarray(points, dtype=object)

    def _parse(self, output):
        """Perform output parsing and return persistence pairs."""
        # Will be filled with the persistence pairs, i.e. the vertices
        # of simplices that are paired with each other.
        tuples = []

        for line in output.splitlines():
            if line.startswith('('):
                line = line[1:len(line) - 1]

                creator, destroyer = line.split(',')
                creator = self._parse_simplex(creator)
                destroyer = self._parse_simplex(destroyer)

                tuples.append((creator, destroyer))

        return tuples

    def _parse_simplex(self, simplex):
        simplex = simplex[1:len(simplex) - 1]
        simplex = simplex.split(' ')
        simplex = map(int, simplex)

        return list(simplex)

    def _make_points(self, tuples, D):
        """Create persistence points from tuples and distances."""
        return [
            (self._get_weight(creator, D), self._get_weight(destroyer, D))
            for creator, destroyer in tuples
        ]

    def _get_weight(self, simplex, D):
        """Get weight of simplex based on distance matrix."""
        max_weight = 0.0
        for subset in itertools.combinations(simplex, 2):
            max_weight = max(max_weight, D[subset])

        return max_weight

