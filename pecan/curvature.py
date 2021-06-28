"""
Set of graph-based curvature measures for benchmarking
"""
import numpy as np
from DiffusionEMD import DiffusionCheb
from DiffusionEMD.estimate_utils import l1_distance_matrix
from pecan import data
from DiffusionEMD.estimate_utils import l1_distance_matrix
import graphtools

class Ollivier_Ricci_Curvature_DEMD():
	"""
	As input, takes a graphtools graph.
	Calculates the Ollivier-Ricci curvature using diffusion distance as the ground distance and Diffusion EMD as the Wasserstein distance.
	"""
	def __init__(self, graph, idleness_parameter): # TODO: Implement idlenss parameter
		# Compute symmetric diffusion operator
		self.Ms = graph.diff_aff.toarray() #TODO: can we keep things sparse for longer
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
		return np.linalg.norm(self.diffusion_coordinates[i] - self.diffusion_coordinates[j],ord=1) # use L1 norm
		
	def EMD(self,distributions):
		# takes two distributions as input, returns the DEMD between them.
		embeddings = self.demd.transform(distributions)
		d = np.linalg.norm(embeddings[0]-embeddings[1], ord=1)
		return d
	def curvature_between(self,i,j):
		"""
		Returns the Ollivier-Ricci-DEMD curvature between nodes i and j
		"""
		# create a pair of distributions by diffusing two diracs
		distributions = np.zeros((len(self.A),2))
		distributions[:,0][i] = 1
		distributions[:,1][j] = 1
		distributinos = self.M @ distributions # TODO: This is inefficient and could be combined with DEMD's diffusions.
		# find demd between distributions
		Wij = self.EMD(distributions)
		# find diffusion distance between i and j
		dij = self.diffusion_distances(i, j)
		# convert to Ollivier Ricci Curvature
		Kij = 1 - Wij/dij #TODO: warn when dij is zero
		return Kij
	