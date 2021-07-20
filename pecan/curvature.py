"""
########..####.########.########.##.....##..######..####..#######..##....##.....######..##.....##.########..##.....##....###....########.##.....##.########..########..######.
##.....##..##..##.......##.......##.....##.##....##..##..##.....##.###...##....##....##.##.....##.##.....##.##.....##...##.##......##....##.....##.##.....##.##.......##....##
##.....##..##..##.......##.......##.....##.##........##..##.....##.####..##....##.......##.....##.##.....##.##.....##..##...##.....##....##.....##.##.....##.##.......##......
##.....##..##..######...######...##.....##..######...##..##.....##.##.##.##....##.......##.....##.########..##.....##.##.....##....##....##.....##.########..######....######.
##.....##..##..##.......##.......##.....##.......##..##..##.....##.##..####....##.......##.....##.##...##....##...##..#########....##....##.....##.##...##...##.............##
##.....##..##..##.......##.......##.....##.##....##..##..##.....##.##...###....##....##.##.....##.##....##....##.##...##.....##....##....##.....##.##....##..##.......##....##
########..####.##.......##........#######...######..####..#######..##....##.....######...#######..##.....##....###....##.....##....##.....#######..##.....##.########..######.
Set of graph-based curvature measures for benchmarking
"""
import numpy as np
from DiffusionEMD import DiffusionCheb
from DiffusionEMD.estimate_utils import l1_distance_matrix
from pecan import data
from DiffusionEMD.estimate_utils import l1_distance_matrix
import graphtools
from scipy.spatial.distance import pdist, squareform
import ot
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from pydiffmap import diffusion_map
from tqdm import trange
from sklearn.metrics.pairwise import pairwise_kernels


class Ollivier_Ricci_Curvature_Regular_OT_with_Diffusion_Distances:
    """
    As input, takes a graphtools graph.
    Calculates the Ollivier-Ricci curvature using diffusion distance as the ground distance and Diffusion EMD as the Wasserstein distance.
    """

    def __init__(self, graph, idleness_parameter):  # TODO: Implement idlenss parameter
        self.name = "Curvature from Regular OT with L2 diffusion distance"
        # Compute symmetric diffusion operator
        self.Ms = graph.diff_aff.toarray()  # TODO: can we keep things sparse for longer
        # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        self.M = graph.P.toarray()
        self.A = graph.K.toarray() - np.eye(len(self.Ms))
        # Create diffusion map
        self.E, self.V = np.linalg.eigh(self.Ms)
        self.diffusion_coordinates = self.V * self.E
        # create distance matrix of diffusion distances
        diffusion_distance_matrix_condensed = pdist(self.diffusion_coordinates)
        self.diffusion_distance_matrix = np.ascontiguousarray(
            squareform(diffusion_distance_matrix_condensed)
        )

    def diffusion_distances(self, i, j):
        return np.linalg.norm(
            self.diffusion_coordinates[i] - self.diffusion_coordinates[j], ord=2
        )  # use L2 norm

    def EMD(self, distributions):
        # takes two distributions as input, returns the DEMD between them.
        d = ot.emd2(
            np.ascontiguousarray(distributions[:, 0]),
            np.ascontiguousarray(distributions[:, 1]),
            self.diffusion_distance_matrix,
        )
        return d

    def curvature_between(self, i, j):
        """
        Returns the Ollivier-Ricci curvature between nodes i and j
        """
        # create a pair of distributions by diffusing two diracs
        distributions = np.zeros((len(self.A), 2))
        distributions[:, 0][i] = 1
        distributions[:, 1][j] = 1
        distributions = (
            distributions.T @ self.M
        )  # TODO: This is inefficient and could be combined with DEMD's diffusions.
        # find demd between distributions
        Wij = self.EMD(distributions.T)
        # find diffusion distance between i and j
        dij = self.diffusion_distances(i, j)
        # convert to Ollivier Ricci Curvature
        Kij = 1 - Wij / dij  # TODO: warn when dij is zero
        # debugging
        print(f"EMD is {Wij}")
        print(f"Diffusion distance is {dij}")
        return Kij


class Ollivier_Ricci_Curvature_DEMD:
    """
    As input, takes a graphtools graph.
    Calculates the Ollivier-Ricci curvature using diffusion distance as the ground distance and Diffusion EMD as the Wasserstein distance.
    """

    def __init__(
        self, graph, idleness_parameter=0.5, alpha=1
    ):  # TODO: Implement idleness parameter
        self.alpha = alpha
        self.name = "curvature from DEMD over l1 diffusion distance"
        # Compute symmetric diffusion operator
        self.Ms = graph.diff_aff.toarray()  # TODO: can we keep things sparse for longer
        # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        self.M = graph.P.toarray()
        self.A = graph.K.toarray() - np.eye(len(self.Ms))
        # Create diffusion map
        self.E, self.V = np.linalg.eigh(self.Ms)
        self.diffusion_coordinates = self.V * self.E
        # initialize diffusion emd operator and fit to graph
        self.demd = DiffusionCheb()
        self.demd.fit(self.A)

    def diffusion_distances(self, i, j):
        # uses the diffusion ground distance defined in Tong et al.
        summed_scales = 0
        for k in range(0, 4):
            scale = 2 ** (-k)
            scaled_M = self.M ** scale
            summed_scales += scale ** self.alpha * np.sum(
                np.abs(scaled_M[i] - scaled_M[j])
            )
        return summed_scales

    def EMD(self, distributions):
        # takes two distributions as input, returns the DEMD between them.
        embeddings = self.demd.transform(distributions)
        d = np.linalg.norm(embeddings[0] - embeddings[1], ord=1)
        return d

    def curvature_between(self, i, j):
        """
        Returns the Ollivier-Ricci-DEMD curvature between nodes i and j
        """
        # create a pair of distributions by diffusing two diracs
        distributions = np.zeros((len(self.A), 2))
        distributions[:, 0][i] = 1
        distributions[:, 1][j] = 1
        distributions = (
            distributions.T @ self.M
        )  # TODO: This is inefficient and could be combined with DEMD's diffusions.
        # find demd between distributions
        Wij = self.EMD(distributions.T)
        # find diffusion distance between i and j
        dij = self.diffusion_distances(i, j)
        # convert to Ollivier Ricci Curvature
        Kij = 1 - Wij / dij  # TODO: warn when dij is zero
        # debugging
        print(f"DEMD is {Wij}")
        print(f"Diffusion distance is {dij}")
        return Kij


class Ollivier_Ricci_Curvature_DEMD_Total:
    """
    As input, takes a graphtools graph.
    Calculates the Ollivier-Ricci curvature using diffusion distance as the ground distance and Diffusion EMD as the Wasserstein distance.
    """

    def __init__(
        self,
        graph,
        idleness_parameter=0.5,
        lp=1,
        n_scales=6,
        alpha=1,
        diffusion_steps_to_ball=1,
    ):
        # TODO: Implement idleness parameter
        self.lp = lp
        self.alpha = alpha
        self.num_scales = n_scales
        self.diffusion_steps_to_ball = diffusion_steps_to_ball
        self.name = "DEMD_Total: curvature from DEMD over DEMD between diracs"
        # Compute symmetric diffusion operator
        self.Ms = (
            graph.diff_aff.toarray()
        )  # .toarray() #TODO: can we keep things sparse for longer
        # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        self.M = graph.P.toarray()
        self.P = graph.P.toarray()  # TODO: Clean up

        self.A = graph.K - np.eye(len(self.Ms))
        # first, calculate all of the powers of P. Store this within the class for easy access.
        self.Pks = [fractional_matrix_power(self.P, self.alpha)]
        for k in range(self.num_scales + self.diffusion_steps_to_ball):
            self.Pks.append(self.Pks[-1] @ self.Pks[-1])
        # Create diffusion map
        # 		self.E, self.V = np.linalg.eigh(self.Ms)
        # 		self.diffusion_coordinates = self.V * self.E
        # initialize diffusion emd operator and fit to graph
        # self.demd = DiffusionCheb(n_scales=n_scales)
        # self.demd.fit(self.A)

    def slow_DEMD_curvature_between(self, i, j):
        # Raises powers on the diffusion operator directly.
        # Prints differences at each scale, for help with theoretical debugging.
        assert i != j
        d = 0

        # next, take the differences between these powers at different scales
        diraci_embeddings = np.zeros(
            (self.num_scales, len(self.P))
        )  # to store the embeddings into L1 diffusion emd space.
        balli_embeddings = np.zeros(
            (self.num_scales, len(self.P))
        )  # each embedding is a row of the diffusion matrix
        # for point j
        diracj_embeddings = np.zeros(
            (self.num_scales, len(self.P))
        )  # to store the embeddings into L1 diffusion emd space.
        ballj_embeddings = np.zeros(
            (self.num_scales, len(self.P))
        )  # each embedding is a row of the diffusion matrix
        for k in range(self.num_scales):
            scaling_factor_dirac = 2 ** (k * self.alpha)
            scaling_factor_ball = scaling_factor_dirac * (
                2 ** (self.alpha * self.diffusion_steps_to_ball)
            )
            # embed the diracs, which start directly from k = 0
            diraci_embeddings[k] = self.Pks[k][i] * scaling_factor_dirac
            diracj_embeddings[k] = self.Pks[k][j] * scaling_factor_dirac
            # embed the balls, which are very similar, but start from one higher diffusion power
            balli_embeddings[k] = (
                self.Pks[k + self.diffusion_steps_to_ball][i] * scaling_factor_ball
            )
            ballj_embeddings[k] = (
                self.Pks[k + self.diffusion_steps_to_ball][j] * scaling_factor_ball
            )

        # calculate L1 distances between diracs and balls
        Wij = np.sum(np.abs(balli_embeddings - ballj_embeddings))
        dij = np.sum(np.abs(diraci_embeddings - diracj_embeddings))
        print(f"EMD between diffused diracs is {Wij}")
        print(f"EMD between diracs is {dij}")
        # print blow-by-blow comparison
        # differences by scale
        diff_by_scale = np.sum(
            np.abs(diraci_embeddings - diracj_embeddings), axis=1
        ) / np.array([2 ** i for i in range(self.num_scales)])
        print(f"Summarιεσ {diff_by_scale}")
        plt.plot(diff_by_scale)
        plt.plot(np.sum(np.abs(diraci_embeddings - diracj_embeddings), axis=1))
        plt.show()
        print(
            f"dirac differences by scale {np.sum(np.abs(diraci_embeddings-diracj_embeddings),axis=1)}"
        )
        print(
            f"Ball differences by scale {np.sum(np.abs(balli_embeddings-ballj_embeddings),axis=1)}"
        )
        Kij = 1 - Wij / dij
        return Kij

    # def diffusion_distances(self, i, j):
    # 	# uses the diffusion ground distance defined in Tong et al.
    # 	summed_scales = 0
    # 	for k in range(0,4):
    # 		scale = 2**(-k)
    # 		scaled_M = (self.M**scale)
    # 		summed_scales += scale**self.alpha * np.sum(np.abs(scaled_M[i] - scaled_M[j]))
    # 	return summed_scales

    def EMD(self, distributions):
        # takes two distributions as input, returns the DEMD between them.
        embeddings = self.demd.transform(distributions)
        d = np.linalg.norm(embeddings[0] - embeddings[1], ord=1)
        return d

    def curvature_between(self, i, j):
        """
        Returns the Ollivier-Ricci-DEMD curvature between nodes i and j
        """
        # create a pair of distributions by diffusing two diracs
        distributions = np.zeros((len(self.A), 2))
        distributions[:, 0][i] = 1
        distributions[:, 1][j] = 1
        distributions_diffused = (
            distributions.T @ self.M
        )  # TODO: This is inefficient and could be combined with DEMD's diffusions.
        # find demd between distributions
        Wij = self.EMD(distributions_diffused.T)
        # find diffusion distance between i and j
        dij = self.EMD(distributions)
        # convert to Ollivier Ricci Curvature
        Kij = 1 - Wij / dij  # TODO: warn when dij is zero
        # debugging
        print(f"EMD between diffused diracs is {Wij}")
        print(f"EMD between diracs is {dij}")
        return Kij


class Ollivier_Ricci_Curvature_DEMD_Total_linear:
    """
    As input, takes a graphtools graph.
    Calculates the Ollivier-Ricci curvature using diffusion distance as the ground distance and Diffusion EMD as the Wasserstein distance.
    """

    def __init__(
        self,
        graph,
        idleness_parameter=0.5,
        lp=1,
        n_scales=6,
        alpha=1,
        separation_factor=1,
    ):  # TODO: Implement idleness parameter
        self.lp = lp
        self.alpha = alpha
        self.num_scales = n_scales
        self.name = "DEMD_Total: curvature from DEMD over DEMD between diracs"
        # Compute symmetric diffusion operator
        self.Ms = (
            graph.diff_aff.toarray()
        )  # .toarray() #TODO: can we keep things sparse for longer
        # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        self.M = graph.P.toarray()
        self.P = graph.P.toarray()  # TODO: Clean up
        self.Pks = [fractional_matrix_power(self.P, self.alpha)]

        self.A = graph.K.toarray() - np.eye(len(self.Ms))
        # first, calculate all of the powers of P. Store this within the class for easy access.
        for k in range(self.num_scales + separation_factor):
            self.Pks.append(self.Pks[0] @ self.Pks[-1])
        # Create diffusion map
        # 		self.E, self.V = np.linalg.eigh(self.Ms)
        # 		self.diffusion_coordinates = self.V * self.E
        # initialize diffusion emd operator and fit to graph
        # self.demd = DiffusionCheb(n_scales=n_scales)
        # self.demd.fit(self.A)

    def slow_DEMD_curvature_between(self, i, j):
        # Raises powers on the diffusion operator directly.
        # Prints differences at each scale, for help with theoretical debugging.
        assert i != j
        d = 0

        # next, take the differences between these powers at different scales
        diraci_embeddings = np.zeros(
            (self.num_scales, len(self.P))
        )  # to store the embeddings into L1 diffusion emd space.
        balli_embeddings = np.zeros(
            (self.num_scales, len(self.P))
        )  # each embedding is a row of the diffusion matrix
        # for point j
        diracj_embeddings = np.zeros(
            (self.num_scales, len(self.P))
        )  # to store the embeddings into L1 diffusion emd space.
        ballj_embeddings = np.zeros(
            (self.num_scales, len(self.P))
        )  # each embedding is a row of the diffusion matrix
        for k in range(self.num_scales):
            scaling_factor = self.alpha * (k + 1)
            # embed the diracs, which start directly from k = 0
            diraci_embeddings[k] = self.Pks[k][i] * scaling_factor
            diracj_embeddings[k] = self.Pks[k][j] * scaling_factor
            # embed the balls, which are very similar, but start from one higher diffusion power
            balli_embeddings[k] = self.Pks[k + 1][i] * (scaling_factor + 1 * self.alpha)
            ballj_embeddings[k] = self.Pks[k + 1][j] * (scaling_factor + 1 * self.alpha)

        # calculate L1 distances between diracs and balls
        Wij = np.sum(np.abs(balli_embeddings - ballj_embeddings))
        dij = np.sum(np.abs(diraci_embeddings - diracj_embeddings))

        # print blow-by-blow comparison
        # differences by scale
        diff_by_scale = np.sum(
            np.abs(diraci_embeddings - diracj_embeddings), axis=1
        ) / np.array([k for i in range(self.num_scales)])
        print(f"Summarιεσ {diff_by_scale}")
        plt.plot(diff_by_scale)
        plt.show
        print(
            f"dirac differences by scale {np.sum(np.abs(diraci_embeddings-diracj_embeddings),axis=1)}"
        )
        print(
            f"Ball differences by scale {np.sum(np.abs(balli_embeddings-ballj_embeddings),axis=1)}"
        )
        print(f"EMD between diffused diracs is {Wij}")
        print(f"EMD between diracs is {dij}")
        Kij = 1 - Wij / dij
        return Kij

    # def diffusion_distances(self, i, j):
    # 	# uses the diffusion ground distance defined in Tong et al.
    # 	summed_scales = 0
    # 	for k in range(0,4):
    # 		scale = 2**(-k)
    # 		scaled_M = (self.M**scale)
    # 		summed_scales += scale**self.alpha * np.sum(np.abs(scaled_M[i] - scaled_M[j]))
    # 	return summed_scales

    def EMD(self, distributions):
        # takes two distributions as input, returns the DEMD between them.
        embeddings = self.demd.transform(distributions)
        d = np.linalg.norm(embeddings[0] - embeddings[1], ord=1)
        return d

    def curvature_between(self, i, j):
        """
        Returns the Ollivier-Ricci-DEMD curvature between nodes i and j
        """
        # create a pair of distributions by diffusing two diracs
        distributions = np.zeros((len(self.A), 2))
        distributions[:, 0][i] = 1
        distributions[:, 1][j] = 1
        distributions_diffused = (
            distributions.T @ self.M
        )  # TODO: This is inefficient and could be combined with DEMD's diffusions.
        # find demd between distributions
        Wij = self.EMD(distributions_diffused.T)
        # find diffusion distance between i and j
        dij = self.EMD(distributions)
        # convert to Ollivier Ricci Curvature
        Kij = 1 - Wij / dij  # TODO: warn when dij is zero
        # debugging
        print(f"EMD between diffused diracs is {Wij}")
        print(f"EMD between diracs is {dij}")
        return Kij


class DiffusionRayCurvature:
    """
    As input, takes raw data points.
    """

    def __init__(self, X, t=1, knn=5, num_steps=20, percent_of_manifold_to_cover=0.3):
        self.points = X
        self.num_points = len(X)
        self.t = t
        self.knn = knn
        self.num_steps = num_steps
        self.percent_of_manifold_to_cover = percent_of_manifold_to_cover
        self.name = "Diffusion Ray Curvature"  # to be programmatically accessed by printing functions, e.g.
        dmap = diffusion_map.DiffusionMap.from_sklearn(
            epsilon=0.15, alpha=0.5, n_evecs=10
        )
        self.diffusion_coordinates = dmap.fit_transform(X)
        # self.A = graph.K - np.eye(len(graph.K))
        # D = np.diag(np.sum(self.A, axis=1)**0.5)
        # # Compute symmetric diffusion operator
        # self.Ms = graph.diff_aff #TODO: can we keep things sparse for longer
        # # eigendecompose # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        # # Create diffusion map and diffusion coordinates (basis of diffusion distance)
        # print("Eigendecomposing diffusion matrix")
        # self.E, self.V = np.linalg.eigh(self.Ms)
        # # correct eigenvecs of Ms to M
        # self.V = D @ self.V
        # print("Building diffusion coordinates")
        # self.diffusion_coordinates = self.V * (self.E**self.t)

    def diffusion_distances_to(self, i):
        return np.linalg.norm(
            self.diffusion_coordinates
            - (
                np.ones_like(self.diffusion_coordinates)
                @ np.diag(self.diffusion_coordinates[i])
            ),
            axis=1,
        )

    def curvature(self, i):
        # Find max diffusion distance from i
        distances_to_i = self.diffusion_distances_to(i)
        max_dist_to_i = np.max(distances_to_i)
        # print("max dist is", max_dist_to_i)
        # Estimate step size from max dist to i
        step_size = max_dist_to_i / (self.num_steps / self.percent_of_manifold_to_cover)
        # print("step size is",step_size)
        # find k nearest neighbors for i.
        nn = np.argsort(distances_to_i)  # sorts the adjacency matrix
        knn = nn[: self.knn]  # takes the k values with highest affinity
        # print("adjacencies",self.A[i][knn])
        # construct rays from each neighbor
        rays = np.zeros((self.knn, self.num_steps, self.diffusion_coordinates.shape[1]))
        ray_coords = np.zeros((self.knn, self.num_steps))  # for debugging
        for h, n in enumerate(knn):
            x = n
            for i in range(self.num_steps):
                # print(i)
                # set new threshold distance
                threshold = distances_to_i[x] + step_size
                # print("threshold",threshold)
                # get distances to current point
                distances_to_x = self.diffusion_distances_to(x)
                # filter based on threshold
                indxs = (distances_to_i > threshold).nonzero()
                # print("choosing between possible indexes that meet distance to i threshold")
                # print(indxs)
                if len(indxs[0]) == 0:
                    break
                distances_filtered = distances_to_x[
                    indxs
                ]  # the nonzero function returns the indices of all nonzero points
                # find point closest to x
                # print("here are the distances to filter",distances_filtered)
                # print("and the min")
                # print(np.min(distances_filtered))
                # print(np.argmin(distances_filtered))
                x = indxs[0][np.argmin(distances_filtered)]
                # print('chose ',x)
                # store this index in rays
                rays[h][i] = self.diffusion_coordinates[x]
                ray_coords[h][i] = x
        # for each array, compute a line of diffusion distances
        distances_between_rays = np.empty((self.knn ** 2, self.num_steps))
        for ray1 in range(self.knn):
            for ray2 in range(self.knn):
                for step in range(self.num_steps):
                    distances_between_rays[ray1 * self.knn + ray2][
                        step
                    ] = np.linalg.norm(rays[ray1][step] - rays[ray2][step])

        # estimate curvature by comparing the integral of the difference growth to the expected linear increase
        # TODO: How best to combine multiple rays into a single curvature measurement?
        deviations_total = 0
        nonzero_deviations = 0
        for dr in distances_between_rays:
            expected = dr[-1] * self.num_steps / 2
            actual = np.sum(dr)
            if actual != 0:
                deviations_total += actual - expected
                nonzero_deviations += 1
        # average deviation by number of rays
        deviations_avg = deviations_total / nonzero_deviations
        return deviations_avg, ray_coords, distances_between_rays

    def pointwise_curvature(self):
        # returns an [n_points] sized array of pointwise curvatures.
        # TODO: How can we speed up redundant distance calculations?
        # TODO: We probably don't need to compute the curvature of every point. Can we sample points, and then average the curvatures around them?
        curvatures = np.empty(self.num_points)
        for i in trange(self.num_points):
            curvatures[i] = self.curvature(i)[0]
        return curvatures


class DiffusionRayCurvatureV2:
    """
    As input, takes raw data points.
    """

    def __init__(
        self,
        graph,
        t=1,
        knn=5,
        num_steps=20,
        percent_of_manifold_to_cover=0.3,
        n_evecs=10,
        radius=0.3,
    ):
        self.t = t
        self.knn = knn
        self.num_steps = num_steps
        self.percent_of_manifold_to_cover = percent_of_manifold_to_cover
        self.name = "Diffusion Ray Curvature"  # to be programmatically accessed by printing functions, e.g.
        self.radius = radius
        # dmap = diffusion_map.DiffusionMap.from_sklearn(
        # epsilon=0.15, alpha=0.5, n_evecs=n_evecs,
        # )
        # self.diffusion_coordinates = dmap.fit_transform(X)

        self.A = graph.K.toarray() - np.eye(len(graph.K.toarray()))
        self.num_points = len(self.A)
        D = np.diag(1 / np.sum(self.A, axis=1) ** 0.5)
        # # Compute symmetric diffusion operator
        self.Ms = graph.diff_aff.toarray()  # TODO: can we keep things sparse for longer
        # # eigendecompose # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        # # Create diffusion map and diffusion coordinates (basis of diffusion distance)
        print("Eigendecomposing diffusion matrix")
        self.E, self.V = np.linalg.eigh(self.Ms)
        # # correct eigenvecs of Ms to M
        self.V = D @ self.V
        # print("Building diffusion coordinates")
        self.diffusion_coordinates = self.V * (self.E ** self.t)

    def diffusion_distances_to(self, i):
        return np.linalg.norm(
            self.diffusion_coordinates
            - (
                np.ones_like(self.diffusion_coordinates)
                @ np.diag(self.diffusion_coordinates[i])
            ),
            axis=1,
        )

    def curvature(self, i, num_steps=20):
        self.num_steps = num_steps
        # Find max diffusion distance from i
        distances_to_i = self.diffusion_distances_to(i)
        max_dist_to_i = np.max(distances_to_i)
        # Convert this distance to a radius of inclusion
        radius = self.radius * max_dist_to_i
        # find k nearest neighbors for i.
        nn = np.argsort(distances_to_i)  # sorts the adjacency matrix
        knn = nn[1 : self.knn + 1]  # takes the k values with highest affinity
        # Simple proof of concept: can be heavily optimized
        # Loop through nearest neighbors and assemble rays for each
        rays = np.zeros((self.knn, self.num_steps, self.diffusion_coordinates.shape[1]))
        ray_coords = np.zeros((self.knn, self.num_steps), dtype=np.int)  # for debugging
        x = self.diffusion_coordinates[i]
        for m, k in enumerate(knn):
            # Loop through all points in the dataset and compute the distance from $p$ to the ray that passes through x and y
            point_dists = []
            y = self.diffusion_coordinates[k]
            normalized_ray_direction = (y - x) / np.linalg.norm(y - x)
            # Only consider coordinates within a specified radius from the central point
            points_within_radius = (distances_to_i <= radius).nonzero()[
                0
            ]  # gives the indices of points within the radius from i
            for n, j in enumerate(points_within_radius):
                p = self.diffusion_coordinates[j]
                # length of the hypotenuse from x to p
                c2 = np.linalg.norm(p - x) ** 2
                # length of ray to p's projection onto y-x
                a2 = (np.dot(normalized_ray_direction, p - x)) ** 2
                # length of shortest side from p to closest point on ray y-x
                b2 = c2 - a2
                point_dists.append(b2)
            # take the closest num_step points
            # print("Sorted point dists",np.sort(point_dists))
            ray_coords[m] = points_within_radius[
                np.argsort(point_dists)[: self.num_steps]
            ]
            # print("have ray coords",ray_coords[m])
        # for each array, compute a line of diffusion distances
        distances_between_rays = np.empty((self.knn ** 2, self.num_steps))
        for ray1 in range(self.knn):
            for ray2 in range(self.knn):
                for step in range(self.num_steps):
                    # print("taking slice ",ray_coords[ray1][step])
                    distances_between_rays[ray1 * self.knn + ray2][
                        step
                    ] = np.linalg.norm(
                        self.diffusion_coordinates[ray_coords[ray1][step]]
                        - self.diffusion_coordinates[ray_coords[ray2][step]]
                    )

        # estimate curvature by comparing the integral of the difference growth to the expected linear increase
        # TODO: How best to combine multiple rays into a single curvature measurement?
        deviations_total = 0
        nonzero_deviations = 0
        for dr in distances_between_rays:
            expected = dr[-1] * self.num_steps / 2
            actual = np.sum(dr)
            if actual != 0:
                deviations_total += actual - expected
                nonzero_deviations += 1
        # average deviation by number of rays
        deviations_avg = deviations_total  # / nonzero_deviations
        return deviations_avg, ray_coords, distances_between_rays

    def pointwise_curvature(self):
        # returns an [n_points] sized array of pointwise curvatures.
        # TODO: How can we speed up redundant distance calculations?
        # TODO: We probably don't need to compute the curvature of every point. Can we sample points, and then average the curvatures around them?
        curvatures = np.empty(self.num_points)
        for i in trange(self.num_points):
            curvatures[i] = self.curvature(i)[0]
        return curvatures


class DiffusionForkCurvature:
    """
    Estimate the curvature from the Alexandrov Triangle condition, by comparing the distance across a triangle to the expected distance in the euclidean setting.
    As input, takes raw data points.
    """

    def __init__(
        self,
        X,
        t=1,
        knn=5,
        num_steps=20,
        percent_of_manifold_to_cover=0.3,
        n_evecs=10,
        radius=0.3,
    ):
        self.points = X
        self.num_points = len(X)
        self.t = t
        self.knn = knn
        self.num_steps = num_steps
        self.percent_of_manifold_to_cover = percent_of_manifold_to_cover
        self.name = "Diffusion Ray Curvature"  # to be programmatically accessed by printing functions, e.g.
        self.radius = radius
        dmap = diffusion_map.DiffusionMap.from_sklearn(
            epsilon=0.15, alpha=0.5, n_evecs=n_evecs
        )
        self.diffusion_coordinates = dmap.fit_transform(X)

    def diffusion_distances_to(self, i):
        return np.linalg.norm(
            self.diffusion_coordinates
            - (
                np.ones_like(self.diffusion_coordinates)
                @ np.diag(self.diffusion_coordinates[i])
            ),
            axis=1,
        )

    def diffusion_distance(self, i, j):
        return np.linalg.norm(
            self.diffusion_coordinates[i] - self.diffusion_coordinates[j]
        )

    def diffusion_rays(self, i):
        """
        Generate geodesic rays centered at the point i, in directions corresponding to each of i's nearest neighbors.
        Returns the indices
        """
        # Find max diffusion distance from i
        distances_to_i = self.diffusion_distances_to(i)
        max_dist_to_i = np.max(distances_to_i)
        # Convert this distance to a radius of inclusion
        radius = self.radius * max_dist_to_i
        # find k nearest neighbors for i.
        nn = np.argsort(distances_to_i)  # sorts the adjacency matrix
        knn = nn[1 : self.knn + 1]  # takes the k values with highest affinity
        # Simple proof of concept: can be heavily optimized
        # Loop through nearest neighbors and assemble rays for each
        rays = np.zeros((self.knn, self.num_steps, self.diffusion_coordinates.shape[1]))
        ray_coords = np.zeros((self.knn, self.num_steps), dtype=np.int)  # for debugging
        x = self.diffusion_coordinates[i]
        # Only consider coordinates within a specified radius from the central point
        points_within_radius = (distances_to_i <= radius).nonzero()[
            0
        ]  # gives the indices of points within the radius from i
        for m, k in enumerate(knn):
            # Loop through all points in the dataset and compute the distance from $p$ to the ray that passes through x and y
            point_dists = []
            y = self.diffusion_coordinates[k]
            normalized_ray_direction = (y - x) / np.linalg.norm(y - x)
            for n, j in enumerate(points_within_radius):
                p = self.diffusion_coordinates[j]
                # length of the hypotenuse from x to p
                c2 = np.linalg.norm(p - x) ** 2
                # length of ray to p's projection onto y-x
                a2 = (np.dot(normalized_ray_direction, p - x)) ** 2
                # length of shortest side from p to closest point on ray y-x
                b2 = c2 - a2
                point_dists.append(b2)
            # take the closest num_step points
            # print("Sorted point dists",np.sort(point_dists))
            ray_coords[m] = points_within_radius[
                np.argsort(point_dists)[: self.num_steps]
            ]
        return ray_coords

    def curvature(self, i, num_triangles=5):
        """Constructs a family of triangles with point i as the midpoint of the hypotenuse.
        Measures the deviation of the distance from i to the opposite vertex from the euclidean case.
        Averages this deviation into a curvature estimate.
        Args:
            i (int): index of point to construct triangles around.
            num_triangles (int): the number of triangles to sample *around each directional ray*.
        Returns:
            k, the averaged curvature across all rays
            curvs, the complete curvature information
        """
        # List "forbidden points": points too close to i to give good diffusion distances.
        distances_to_i = self.diffusion_distances_to(i)
        max_dist_to_i = np.max(distances_to_i)
        radius = (
            self.radius * max_dist_to_i
        )  # TODO:0 Reformulate the radius in terms of random walk steps
        points_within_radius = (distances_to_i <= radius).nonzero()[
            0
        ]  # gives the indices of points within the radius from i
        forbidden_points = (distances_to_i <= radius * 0.05).nonzero()[
            0
        ]  # TODO:0 The threshold for `forbidden_points` is arbitrary, and sometimes results in deleting all of the points, or all but 2 of the points
        # remove forbidden_points from points within radius
        points_within_radius = np.setdiff1d(
            points_within_radius, forbidden_points
        )  # TODO:0 Do this within the ray maker?
        # Obtain diffusion rays around i, along with the local neighborhood of points they were sampled from (points within the specified radius from i)
        ray_coords = self.diffusion_rays(
            i
        )  # TODO:0 diffusion_ray currently recalculates distances_to_i and points_within_radius
        # Loop through the rays, and for each, sample num_triangles
        # store the resulting curvatures here
        curvs = np.empty((len(ray_coords), num_triangles))
        triangles = []
        for m, ray in enumerate(ray_coords):
            # print("ray size before cleansing", len(ray))
            # split ray into points to the left and right of i.
            sorted_by_i = np.argsort(distances_to_i[ray])
            endpoint = sorted_by_i[
                -1
            ]  # find the index of the element in the ray that is the furthest from i
            closest_to_i = sorted_by_i[
                0
            ]  # the index of the element that is i (used later to split the ray into two pieces)
            ray_distances_to_endpoint = np.argsort(
                self.diffusion_distances_to(ray[endpoint])[ray]
            )  # the indices sorted by the distances from each point in the ray to the endpoint
            index_of_i = np.where(ray_distances_to_endpoint == closest_to_i)[0][0]
            ray1 = ray[ray_distances_to_endpoint[:index_of_i]]
            ray2 = ray[ray_distances_to_endpoint[index_of_i:]]
            # TODO: We technically only have to take distances between this and other points in the ray
            # remove forbidden points from ray
            # trimmed_ray = np.setdiff1d(ray, forbidden_points)
            # print(
            # "ray size after cleansing", len(trimmed_ray)
            # )  # TODO:0 Sometimes the cleansed array is empty

            # choose two points from the ray (ensuring that they aren't within the immediate neighborhood of i)
            a = np.random.choice(ray1)
            b = np.random.choice(ray2)
            C = self.diffusion_distance(a, b)
            AI = self.diffusion_distance(a, i)
            assert AI != 0
            # BI = self.diffusion_distance(b,i)
            # attach a variety of third points
            for n in range(num_triangles):
                # sample a random point
                c = np.random.choice(points_within_radius, replace=False)
                triangles.append([a, b, c])
                # calculate distance from c to i on manifold
                CI_intrinsic = self.diffusion_distance(i, c)
                # calculate euclidean distance from c to i
                A = self.diffusion_distance(c, b)

                B = self.diffusion_distance(a, c)
                # Use law of cosines to find angle by a
                # theta = np.arccos((B**2 + C**2 - A**2)/(2*B*C))
                # Use it again, but now to find the length CI
                CI_euclidean = np.sqrt(
                    B ** 2
                    + AI ** 2
                    - 2 * B * AI * ((B ** 2 + C ** 2 - A ** 2) / (2 * B * C))
                )
                # print(f"The diffusion distance is {CI_intrinsic}")
                # print(f"The euclidean distance is {CI_euclidean}")
                ratio = CI_euclidean / CI_intrinsic
                curvs[m][n] = ratio
        # To convert this to a curvature measurement, avg the ratios and subtract from 1 (a crude method)
        avg_curvatures = np.mean(curvs, axis=1)  # this yields averages for each ray
        # This gives a "sectional" curvature
        avg_curvatures = np.ones_like(avg_curvatures) - avg_curvatures
        # Which we can average further to the scalar curvature
        return np.mean(avg_curvatures), curvs, triangles, ray1, ray2

    def pointwise_curvature(self):
        # returns an [n_points] sized array of pointwise curvatures.
        # TODO: How can we speed up redundant distance calculations?
        # TODO: We probably don't need to compute the curvature of every point. Can we sample points, and then average the curvatures around them?
        curvatures = np.empty(self.num_points)
        for i in trange(self.num_points):
            curvatures[i] = self.curvature(i)[0]
        return curvatures


class DiffusionForkCurvatureV2:
    """
    Estimate the curvature from the Alexandrov Triangle condition, by comparing the distance across a triangle to the expected distance in the euclidean setting.
    As input, takes raw data points.
    """

    def __init__(
        self,
        graph,
        t=1,
        num_points=20,
        n_evecs=2,
        radius=0.3,
    ):
        # self.points = X
        # self.num_points = len(X)
        self.t = t
        self.num_steps = num_points
        self.name = "Diffusion Ray Curvature V2"  # to be programmatically accessed by printing functions, e.g.
        self.radius = radius
        self.A = graph.K.toarray()
        self.num_points = len(self.A)
        Dnoh = np.diag(1 / np.sum(self.A, axis=1) ** 0.5)
        Dn = np.diag(1 / np.sum(self.A, axis=1))
        # apply anisotropy to kernel
        # self.A = Dnoh @ self.A @ Dnoh
        # print("Using anisotropy alpha=0.5")
        # # Compute symmetric diffusion operator
        self.Ms = Dnoh @ self.A @ Dnoh  # TODO: can we keep things sparse for longer
        # # eigendecompose # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        # # Create diffusion map and diffusion coordinates (basis of diffusion distance)
        print("Eigendecomposing diffusion matrix")
        self.E, self.V = np.linalg.eigh(self.Ms)
        print(self.E)
        # self.E = np.flip(self.E)
        # self.V = np.flip(self.V, axis=1)
        # # correct eigenvecs of Ms to M
        self.V = Dnoh @ self.V
        # threshold by number of evecs requested
        self.E = self.E[-n_evecs - 1 : -1]
        self.V = self.V[:, -(n_evecs + 1) : -1]
        # print("Building diffusion coordinates")
        self.diffusion_coordinates = self.V * (self.E ** self.t)

        # dmap = diffusion_map.DiffusionMap.from_sklearn(
        #     epsilon=0.15, alpha=0, n_evecs=n_evecs, t=t
        # )
        # self.diffusion_coordinates = dmap.fit_transform(X)

    def diffusion_distances_to(self, i):
        return np.linalg.norm(
            self.diffusion_coordinates
            - (
                np.ones_like(self.diffusion_coordinates)
                @ np.diag(self.diffusion_coordinates[i])
            ),
            axis=1,
        )

    def diffusion_distance(self, i, j):
        return np.linalg.norm(
            self.diffusion_coordinates[i] - self.diffusion_coordinates[j]
        )

    def diffusion_ray_between(self, i, j):
        """
        Generate geodesic between i and j. Return the indices of points that lie approximately along the geodesic,
        along with their exact distance from the geodesic.
        """
        # * Capture the points in between i and j, to create a local set of diffusion coordinates
        d_ij = self.diffusion_distance(i, j)
        ds_to_i = self.diffusion_distances_to(i)
        ds_to_j = self.diffusion_distances_to(j)
        # find points within radius d_ij from i
        ds_to_i_threshold = ds_to_i < d_ij
        ds_to_j_threshold = ds_to_j < d_ij
        # combine thresholds to give indices of points within i and j
        threshold = ds_to_i_threshold * ds_to_j_threshold
        local_points = threshold.nonzero()[
            0
        ]  # gives indices of points within local region
        # * Find points close to the diffusion ray
        x = self.diffusion_coordinates[i]
        y = self.diffusion_coordinates[j]
        normalized_ray_direction = (y - x) / np.linalg.norm(y - x)
        point_dists = []
        # Loop through local points
        for n, k in enumerate(local_points):
            p = self.diffusion_coordinates[k]
            # length of the hypotenuse from x to p
            c2 = np.linalg.norm(p - x) ** 2
            # length of ray to p's projection onto y-x
            a2 = (np.dot(normalized_ray_direction, p - x)) ** 2
            # length of shortest side from p to closest point on ray y-x
            b2 = c2 - a2
            point_dists.append(b2)
        # take the closest num_step points
        # print("Sorted point dists",np.sort(point_dists))
        ray_coords = local_points[np.argsort(point_dists)[: self.num_steps]]
        ray_deviations = np.sort(point_dists)[: self.num_steps]
        return ray_coords, ray_deviations, local_points

    def curvature(self, i, num_triangles=5):
        """Constructs a family of triangles with point i as the midpoint of the hypotenuse.
        Measures the deviation of the distance from i to the opposite vertex from the euclidean case.
        Averages this deviation into a curvature estimate.
        Args:
            i (int): index of point to construct triangles around.
            num_triangles (int): the number of triangles to sample *around each directional ray*.
        Returns:
            k, the averaged curvature across all rays
            curvs, the complete curvature information
        """
        # List "forbidden points": points too close to i to give good diffusion distances.
        distances_to_i = self.diffusion_distances_to(i)
        max_dist_to_i = np.max(distances_to_i)
        radius = (
            self.radius * max_dist_to_i
        )  # TODO:0 Reformulate the radius in terms of random walk steps
        points_within_radius = (distances_to_i <= radius).nonzero()[
            0
        ]  # gives the indices of points within the radius from i
        forbidden_points = (distances_to_i <= radius * 0.05).nonzero()[
            0
        ]  # TODO:0 The threshold for `forbidden_points` is arbitrary, and sometimes results in deleting all of the points, or all but 2 of the points
        # remove forbidden_points from points within radius
        points_within_radius = np.setdiff1d(
            points_within_radius, forbidden_points
        )  # TODO:0 Do this within the ray maker?
        # Obtain diffusion rays around i, along with the local neighborhood of points they were sampled from (points within the specified radius from i)
        ray_coords = self.diffusion_rays(
            i
        )  # TODO:0 diffusion_ray currently recalculates distances_to_i and points_within_radius
        # Loop through the rays, and for each, sample num_triangles
        # store the resulting curvatures here
        curvs = np.empty((len(ray_coords), num_triangles))
        triangles = []
        for m, ray in enumerate(ray_coords):
            # print("ray size before cleansing", len(ray))
            # split ray into points to the left and right of i.
            sorted_by_i = np.argsort(distances_to_i[ray])
            endpoint = sorted_by_i[
                -1
            ]  # find the index of the element in the ray that is the furthest from i
            closest_to_i = sorted_by_i[
                0
            ]  # the index of the element that is i (used later to split the ray into two pieces)
            ray_distances_to_endpoint = np.argsort(
                self.diffusion_distances_to(ray[endpoint])[ray]
            )  # the indices sorted by the distances from each point in the ray to the endpoint
            index_of_i = np.where(ray_distances_to_endpoint == closest_to_i)[0][0]
            ray1 = ray[ray_distances_to_endpoint[:index_of_i]]
            ray2 = ray[ray_distances_to_endpoint[index_of_i:]]
            # TODO: We technically only have to take distances between this and other points in the ray
            # remove forbidden points from ray
            # trimmed_ray = np.setdiff1d(ray, forbidden_points)
            # print(
            # "ray size after cleansing", len(trimmed_ray)
            # )  # TODO:0 Sometimes the cleansed array is empty

            # choose two points from the ray (ensuring that they aren't within the immediate neighborhood of i)
            a = np.random.choice(ray1)
            b = np.random.choice(ray2)
            C = self.diffusion_distance(a, b)
            AI = self.diffusion_distance(a, i)
            assert AI != 0
            # BI = self.diffusion_distance(b,i)
            # attach a variety of third points
            for n in range(num_triangles):
                # sample a random point
                c = np.random.choice(points_within_radius, replace=False)
                triangles.append([a, b, c])
                # calculate distance from c to i on manifold
                CI_intrinsic = self.diffusion_distance(i, c)
                # calculate euclidean distance from c to i
                A = self.diffusion_distance(c, b)

                B = self.diffusion_distance(a, c)
                # Use law of cosines to find angle by a
                # theta = np.arccos((B**2 + C**2 - A**2)/(2*B*C))
                # Use it again, but now to find the length CI
                CI_euclidean = np.sqrt(
                    B ** 2
                    + AI ** 2
                    - 2 * B * AI * ((B ** 2 + C ** 2 - A ** 2) / (2 * B * C))
                )
                # print(f"The diffusion distance is {CI_intrinsic}")
                # print(f"The euclidean distance is {CI_euclidean}")
                ratio = CI_euclidean / CI_intrinsic
                curvs[m][n] = ratio
        # To convert this to a curvature measurement, avg the ratios and subtract from 1 (a crude method)
        avg_curvatures = np.mean(curvs, axis=1)  # this yields averages for each ray
        # This gives a "sectional" curvature
        avg_curvatures = np.ones_like(avg_curvatures) - avg_curvatures
        # Which we can average further to the scalar curvature
        return np.mean(avg_curvatures), curvs, triangles, ray1, ray2

    def pointwise_curvature(self):
        # returns an [n_points] sized array of pointwise curvatures.
        # TODO: How can we speed up redundant distance calculations?
        # TODO: We probably don't need to compute the curvature of every point. Can we sample points, and then average the curvatures around them?
        curvatures = np.empty(self.num_points)
        for i in trange(self.num_points):
            curvatures[i] = self.curvature(i)[0]
        return curvatures
