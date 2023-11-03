import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

import oineus as oin
from scipy.spatial.distance import pdist, squareform
import tqdm

def get_low(R,j):
    """
    get the low value of column j for incidence matrix R
    """
    if R[:,j].sum() == 0:
        return None
    else:
        return np.where(R[:,j] != 0)[0][-1]

def get_pairs(R):
    lows = [get_low(R,j) for j in range(R.shape[1])]
    pairs = [(lows[j],j) for j in range(len(lows)) if lows[j] is not None]
    essential_pairs = [(j,) for j in range(len(lows)) if (lows[j] is None) and (j not in lows)]

    pairs_all = pairs + essential_pairs

    simplex_ids = [p[-1] for p in pairs_all]
    
    return pairs_all, simplex_ids

def get_crossings(filtration0, filtration1):
    """
    Get the location of the crossings of the persistence diagrams using linear homotopy.

    Inputs:
    - filtration0 : the value of the simplices in the first filtration
    - filtration1 : the value of the simplices in the second filtration
    both should follow the same order of simplices

    Outputs:
    - cross_x : the location of the crossings
    - cross_idx : the index of the simplices that cross
    """
    ps = filtration0
    ms = filtration1 - filtration0

    inter_x = -(ps[None]-ps[:,None]) / (ms[None]-ms[:,None])
    inter_x[np.triu_indices(n=len(inter_x))] = -1
    cross_idx = np.where((inter_x >0)* (inter_x < 1))
    cross_x = inter_x[cross_idx]
    sort_idx = np.argsort(cross_x)
    return cross_x[sort_idx], (cross_idx[0][sort_idx], cross_idx[1][sort_idx])

def get_persistence_pairs(pairs,dims, filtration, p = 0):

    persistence_pairs = []
    if p != "all":
        for _, pair in enumerate(pairs):
            if len(pair)==2:
                if dims[pair[0]] ==p:
                    persistence_pairs.append((filtration[pair[0]], filtration[pair[1]]))
            else:
                if dims[pair[0]] ==p:
                    persistence_pairs.append((filtration[pair[0]], np.inf))
    else:
        for _, pair in enumerate(pairs):
            d = dims[pair[0]]
            if len(pair)==2:
                persistence_pairs.append((filtration[pair[0]], filtration[pair[1]],d))
            else:
                persistence_pairs.append((filtration[pair[0]], np.inf, d))

    return persistence_pairs


def filtration(simplex,dist):
    """
    Computes the Vietoris-Rips filtration value of a simplex based on the distance matrix
    Example:
        simplex = (0,2,3)
        dist = NxN distance matrix
    Returns value of the simplex
    """
    if len(simplex)==1:
        return 0
    else:
        if len(simplex)==2:
            return dist[simplex[0],simplex[1]]
        else:
            max = 0
            for comb in combinations(list(simplex),2):
                if dist[comb[0],comb[1]]>max:
                    max = dist[comb[0],comb[1]]
            return max
        
def get_pairs_sparse(R):
    lows = [r[-1] if len(r)!=0 else None for r in R]
    pairs = [(lows[j],j) for j in range(len(lows)) if lows[j] is not None]
    lows_set = set(lows)
    essential_pairs = [(j,) for j in range(len(lows)) if (lows[j] is None) and (j not in lows_set)]
    pairs_all = pairs + essential_pairs
    simplex_ids = [p[-1] for p in pairs_all]
    return pairs_all, simplex_ids

def list_to_str(l):
    return "-".join([str(x) for x in l])

def get_oineus_pairs(points, max_dim, vr_type="pc", filtration_args=None):

    if vr_type == "pc":
        fil = oin.get_vr_filtration(points, max_dim=max_dim, max_radius=filtration_args["max_radius"], n_threads = 1)
    elif vr_type == "distance":
        fil = oin.get_vr_filtration_from_pwdists(points, max_dim=max_dim, max_radius = filtration_args["max_radius"], n_threads = 1)
    else:
        raise ValueError("vr_type should be either 'pc' or 'distance'")
    
    dualize = False
    dcmp = oin.Decomposition(fil, dualize )

    rp = oin.ReductionParams()
    rp.compute_u = False
    rp.compute_v = True
    rp.n_threads = 1
    # perform reduction
    dcmp.reduce(rp)

    sorted_values = [s.value for s in fil.simplices()]
    dims= [len(s.vertices)-1 for s in fil.simplices()]
    R_ = dcmp.r_data


    pairs, simplex_ids = get_pairs_sparse(R_)
    persistence_pairs = get_persistence_pairs(pairs,dims, sorted_values, p="all")

    fil_simplices = fil.simplices()
    simplex_vertices_dict = {list_to_str(fil_simplices[s].vertices) : persistence_pairs[i] for i,s in enumerate(simplex_ids)}

    return simplex_vertices_dict #, persistence_pairs, simplex_ids #pairs, simplex_ids, simplex_vertices, persistence_pairs

def compute_vineyards(points_list, vr_type = "pc", filtration_args = None):

    """
    points_list : list of point clouds or distance matrices.

    filtration_type : "pc" for point clouds or "distance" for distance matrices.
    """

    print("Computing T0")
    simplex_vertices_dict0 = get_oineus_pairs(points_list[0], vr_type = "pc", filtration_args = filtration_args)
    

    dict_list = [simplex_vertices_dict0]
    times_list = [0]
    t_shift = 0 # time between point clouds

    vertices_uniques = set(simplex_vertices_dict0.keys())


    points_prev = points_list[0].copy()

    for points_next in points_list[1:2]:

        x_cross = np.linspace(0,1,10)

        for x_ in tqdm.tqdm(x_cross):

            # Straight Line homotopy between points
            # What about when points are absorbed?
            points_ = points_prev * (1-x_) + points_next * x_
            simplex_vertices_dict_ = get_oineus_pairs(points_, vr_type = vr_type, filtration_args = filtration_args)
            dict_list.append(simplex_vertices_dict_)
            times_list.append(x_ + t_shift)
            vertices_uniques = vertices_uniques.union(set(simplex_vertices_dict_.keys()))

        print(f"Computing T{t_shift+1}")
        simplex_vertices_dict1 = get_oineus_pairs(points_next, vr_type = vr_type, filtration_args = filtration_args)
        
        dict_list.append(simplex_vertices_dict1)
        vertices_uniques = vertices_uniques.union(set(simplex_vertices_dict1.keys()))
        times_list.append(1)

        points_prev = points_next.copy()
        t_shift += 1


    base_dict = {k:[] for k in vertices_uniques}

    for k in base_dict.keys():
        for t_, dict_ in enumerate(dict_list):
            base_dict[k].append((times_list[t_],) + dict_.get(k,(None,)))
            # if k == '0-1-4':
            #     print((times_list[t_],) + dict_.get(k,(None,)))

    return base_dict


if __name__ == "__main__":
    #distance matrix at time 0
    points = np.random.randn(10,3)
    #distance matrix at time 1
    points1 = np.random.randn(10,3)
    vines_dict = compute_vineyards([points,points1],vr_type="pc",filtration_args={"max_radius":10})