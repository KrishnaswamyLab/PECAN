"""Handle communication with Ripser."""

import os
import tempfile
import subprocess

import numpy as np


class Ripser:
    """Interface with the `ripser` command-line tool."""

    def __call__(self, D):

        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'w') as tmp:
                np.savetxt(tmp, D)

            result = subprocess.run(
                ['ripser', path],
                stdout=subprocess.PIPE
            )

        finally:
            os.remove(path)

        tuples = self._parse(result.stdout.decode('utf-8'))
        return tuples

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
