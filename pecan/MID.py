"""
.  . .-. .-. |   .  . .-. .-. .-. .-. .-. . . .-. .-.   .-. . .   .-. . . .-. .-. .-. .-. .-. .-. .-. .-. . . .-.   .-. .-. .-. .-. . . .-. .-. .-. . . .-.
|\/|  |  |  )    |\/|  |  |  )|-' | |  |  |\|  |  `-.   |(   |     |  |\|  |  |-  |(  `-. |-  |    |   |  |\| |..   |  ) |  |-  |-  | | `-.  |  | | |\| `-.
'  ` `-' `-' |   '  ` `-' `-' '   `-' `-' ' `  '  `-'   `-'  `    `-' ' `  '  `-' ' ' `-' `-' `-'  '  `-' ' ` `-'   `-' `-' '   '   `-' `-' `-' `-' ' ` `-'
"""

import phate
import numpy as np
import graphtools
import scprep
# the algorithm
def find_diffusion_midpoint(i,j,P,threshold=0):
    """
    Diffuses at increasing scales until a point of non-negligible overlap appears between the dirac diffused from i and the dirac diffused from j. Returns this (these) point(s) as the diffusion midpoints.
    """
    if i == j: #TODO: this is hacky
        # print("Really? i == i.")
        return [i]
    # initialize diracs at i and j
    diffused_i = np.zeros(P.shape[0])
    diffused_i[i] = 1
    diffused_j = np.zeros(P.shape[0])
    diffused_j[j] = 1
    indices_of_intersection = []
    while len(indices_of_intersection) == 0:
        # take one step of diffusion
        diffused_i = diffused_i @ P
        diffused_j = diffused_j @ P
        # find the intersection as the product of the diffusions
        intersection = diffused_i * diffused_j
        # print(max(intersection),len(intersection.nonzero()[0]))
        # remove points in the intersection which fall beneath the threshold
        intersection = (intersection >= threshold).astype(int) * intersection
        indices_of_intersection = intersection.nonzero()[0]
    # sort the indices of intersection by the magnitude of intersection
    idxs = np.argsort(intersection[indices_of_intersection])
    return indices_of_intersection[idxs][::-1]

def insert_midpoints(sorted_list, P, threshold=0):
    """
    Given a list of sorted indices, inserts the index of the midpoint between each consecutive pair of points, and returns a new list.
    """
    new_list = []
    for idx1,idx2 in zip(sorted_list[:-1],sorted_list[1:]):
        midpoints = find_diffusion_midpoint(idx1, idx2, P, threshold=threshold)
        m = midpoints[0]
        if m == idx1:
            new_list.append(idx1)
        else:
            new_list.extend([idx1, m])
    new_list.append(idx2)
    return new_list

def iterative_midpoints(sorted_list,P,num_iterations = 4, threshold=0):
    # print(num_iterations)
    for i in range(num_iterations):
        # print(f"iteration {i}")
        sorted_list = insert_midpoints(sorted_list,P,threshold=threshold)
    return sorted_list

def MIDdistance(i, j, P, D, threshold=0, num_iterations=4):
    """
    Diffusion distances work well globally in large point sets, but can be spurious in small neighborhoods. Here, we estimate  
    the distance between i and j as the sum of the euclidean distances between their recursive midpoints, which approximate a geodesic.
    """
    # print(num_iterations)
    midpoints = iterative_midpoints([i,j],P,num_iterations=num_iterations,threshold=threshold)
    distance = 0
    for i in range(len(midpoints)-1):
        x = midpoints[i]
        y = midpoints[i+1]
        distance += D[x,y]
    return distance

def MIDcurvature(x,P,D, radius_hops = 4, num_triangles=10, threshold = 0, num_iterations=5):
    """Estimates the Triangle Condition Curvature using midpoints and distances obtained by MID.py

    Args:
        x (int): index of point to estimate curvature around
        P (ndarray): Diffusion matrix on the graph from which x was drawn   
        D (ndarray): Distance matrix 
        radius_hops (int, optional): Size of area over which curvature is measured. Default: 4 hop neighborhood.
        num_samples (int, optional): Number of points to sample from which to determine the curvature. Of the points sampled, combinations of 2 of them are used to construct triangles.        
        num_triangles (int, optional): Number of triangles to sample from the samples. Cannot exceed num_samples * (num_samples - 1)
        threshold (int, optional): Only consider midpoints above this threshold. Defaults to 0.
        num_iterations (int, optional): Number of recursive midpoints to take. Creates 2**n midpoints between the points. Defaults to 5.
    """
    # * 1. Estimate the outer ring of the radius_hops neighborhood
    diffused = np.zeros(P.shape[0])
    diffused[x] = 1
    diffused = diffused @ np.linalg.matrix_power(P,radius_hops)
    diffused_plus_1 = diffused @ P
    # find indices covered by diffused_plus_1
    diffused_plus_1_indices = diffused_plus_1.nonzero()[0]
    # find indices covered by diffused
    diffused_indices = diffused.nonzero()[0]
    # ring of not-too-close proximity
    ring = np.setdiff1d(diffused_plus_1_indices,diffused_indices,assume_unique=True)
    triangle_endpoints = np.random.choice(ring,(num_triangles,2),replace=False)
    #* 2. Loop through triangle endpoints, construct midpoint of each, and measure curvature
    curvatures = []
    for idx, t in enumerate(triangle_endpoints):
        a = t[0]
        b = t[1]
        c = x
        m = find_diffusion_midpoint(a,b,P,threshold=threshold)[0]
        A = MIDdistance(b,c,P,D,threshold=threshold,num_iterations=num_iterations)
        B = MIDdistance(a,c,P,D,threshold=threshold,num_iterations=num_iterations)
        C = MIDdistance(a,b,P,D,threshold=threshold,num_iterations=num_iterations)
        sorted_dists = sorted([A,B,C])
        # make sure this is a real triangle, and not too skinny
        if sorted_dists[0] + sorted_dists[1] > 1.5*sorted_dists[2]: 
            AM = MIDdistance(a,m,P,D,threshold=threshold,num_iterations=num_iterations)
            MC_intrinsic = MIDdistance(m,c,P,D,threshold=threshold,num_iterations=num_iterations)
            MC_euclidean = np.sqrt(
                        B ** 2
                        + AM ** 2
                        - 2 * B * AM * ((B ** 2 + C ** 2 - A ** 2) / (2 * B * C))
                    )
            print("MC int",MC_intrinsic)
            print("MC euc", MC_euclidean)
            curvature = MC_intrinsic - MC_euclidean
            curvatures.append(curvature)
    return curvatures, ring, triangle_endpoints



def triangle_condition_curvature(x,P,D, radius_hops = 4, num_triangles=10, threshold = 0):
    """Estimates the Triangle Condition Curvature using midpoints and distances obtained by MID.py

    Args:
        x (int): index of point to estimate curvature around
        P (ndarray): Diffusion matrix on the graph from which x was drawn   
        D (ndarray): Distance matrix between points on graph
        radius_hops (int, optional): Size of area over which curvature is measured. Default: 4 hop neighborhood.
        num_samples (int, optional): Number of points to sample from which to determine the curvature. Of the points sampled, combinations of 2 of them are used to construct triangles.        
        num_triangles (int, optional): Number of triangles to sample from the samples. Cannot exceed num_samples * (num_samples - 1)
        threshold (int, optional): Only consider midpoints above this threshold. Defaults to 0.
    """
    # * 1. Estimate the outer ring of the radius_hops neighborhood
    diffused = np.zeros(P.shape[0])
    diffused[x] = 1
    print("powering P")
    diffused = diffused @ np.linalg.matrix_power(P,radius_hops) 
    print("finished powering P")
    diffused_plus_1 = diffused @ P
    # find indices covered by diffused_plus_1
    diffused_plus_1_indices = diffused_plus_1.nonzero()[0]
    # find indices covered by diffused
    diffused_indices = diffused.nonzero()[0]
    # ring of not-too-close proximity
    ring = np.setdiff1d(diffused_plus_1_indices,diffused_indices,assume_unique=True)
    triangle_endpoints = np.random.choice(ring,(num_triangles,2),replace=False)
    #* 2. Loop through triangle endpoints, construct midpoint of each, and measure curvature
    curvatures = []
    print(len(triangle_endpoints))
    for idx, t in enumerate(triangle_endpoints):
        a = t[0]
        b = t[1]
        c = x
        m = find_diffusion_midpoint(a,b,P,threshold=threshold)[0]
        A = D[b,c]
        B = D[a,c]
        C = D[a,b]
        sorted_dists = sorted([A,B,C])
        # make sure this is a real triangle, and not too skinny
        if sorted_dists[0] + sorted_dists[1] > 1.2*sorted_dists[2]: 
            AM = D[a,m]
            MC_intrinsic = D[m,c]
            MC_euclidean = np.sqrt(
                        B ** 2
                        + AM ** 2
                        - 2 * B * AM * ((B ** 2 + C ** 2 - A ** 2) / (2 * B * C))
                    )
            print("MC int",MC_intrinsic)
            print("MC euc", MC_euclidean)
            curvature = MC_intrinsic - MC_euclidean
            curvatures.append(curvature)
    return curvatures, ring, triangle_endpoints

def RayCurvature(x, P, D, radius_hops = 4, num_rays=10, threshold = 0, num_iterations=4):
    """ Like DiffusionRayCurvature, but uses geodesics determined by MID.

    Args:
        x (int): index of point to estimate curvature around
        P (ndarray): Diffusion matrix on the graph from which x was drawn   
        D (ndarray): Distance matrix between points on graph
        radius_hops (int, optional): Size of area over which curvature is measured. Default: 4 hop neighborhood.
        num_samples (int, optional): Number of points to sample from which to determine the curvature. Of the points sampled, combinations of 2 of them are used to construct triangles.        
        num_rays(int, optional): Number of triangles to sample from the samples. Cannot exceed num_samples * (num_samples - 1)
        threshold (int, optional): Only consider midpoints above this threshold. Defaults to 0.
    """
    # * 1. Estimate the outer ring of the radius_hops neighborhood
    diffused = np.zeros(P.shape[0])
    diffused[x] = 1
    print("powering P")
    diffused = diffused @ np.linalg.matrix_power(P,radius_hops) 
    print("finished powering P")
    diffused_plus_1 = diffused @ P
    # find indices covered by diffused_plus_1
    diffused_plus_1_indices = diffused_plus_1.nonzero()[0]
    # find indices covered by diffused
    diffused_indices = diffused.nonzero()[0]
    # ring of not-too-close proximity
    ring = np.setdiff1d(diffused_plus_1_indices,diffused_indices,assume_unique=True)
    ray_endpoints = np.random.choice(ring,num_rays,replace=False)
    #* 2. Loop through each ray endpoint and construct the MID geodesic between it and x
    rays = []
    for idx, t in enumerate(ray_endpoints):
        pts = iterative_midpoints([x,t],P,threshold=threshold,num_iterations=num_iterations)
        rays.append(pts)
    #* 3. Find the distances from ray to ray (using MID distances currently -- almost anything would likely be better.)
    distances_between_rays = []
    

    return curvatures, ring, triangle_endpoints