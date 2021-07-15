import numpy as np
from pydiffmap import diffusion_map

def diffusion_distances_to(i,diffusion_coordinates):
    return np.linalg.norm(
        diffusion_coordinates
        - (
            np.ones_like(self.diffusion_coordinates)
            @ np.diag(diffusion_coordinates[i])
        ),
        axis=1,
    )
def diffusion_distance(self, i, j,diffusion_coordinates):
        return np.linalg.norm(
            diffusion_coordinates[i] - diffusion_coordinates[j]
        )