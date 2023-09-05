import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import oineus as oin


#Edwards Functions
def expand_complex(complex):
    if len(complex) == 1:
        return (complex,)
    expansion = []
    expansion.append(tuple(complex))
    for comb in combinations(complex, len(complex)-1):
        expansion.extend(expand_complex(comb))
    return expansion

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
        
def filtration_all_simplices(simplices, dist):
    """
    Computes the value of all simplices according to a distance matrix
    Input:
    - simplices = list of simplices (e.g. [(0,1),(2,3)])
    - dist = NxN distance matrix
    """
    return [filtration(simplex,dist) for simplex in simplices]

def sort_simplices(simplicial_complex, all_simplices, dist):
    """
    Sort the simplices according to their filtration value

    Input:
    - simplicial complex : the full complex
    - all_simplices: list of all simplices in the complex
    - dist: distance matrix of the points

    Output:
    - sorted_simplices: list of simplices sorted by filtration value
        the ordering is per dimension of simplices (dimension 0 first, then dimension 1, etc.)
    - sorted_values: list of filtration values (in the sorted order)
    """
    sorted_simplices = []
    sorted_values = []
    for d in range(1,1+len(simplicial_complex)):
        d_simplex_vals = [filtration(simplex,dist) for simplex in all_simplices if len(simplex)==d]
        d_simplex = [simplex for simplex in all_simplices if len(simplex)==d]
        sorted_list = sorted(zip(d_simplex_vals,d_simplex))
        sorted_d_simplex = [y for x,y in sorted_list ]
        simplex_vals = [x for x,y in sorted_list]
        sorted_simplices.extend(sorted_d_simplex)
        sorted_values.extend(simplex_vals)
    return sorted_simplices, sorted_values

def get_incidence_matrix(sorted_simplices):
    """
    Compute the incideence matrix from a list of sorted simplices
    """
    D = np.zeros((len(sorted_simplices),len(sorted_simplices)))
    for i in range(len(sorted_simplices)):
        for j in range(i+1,len(sorted_simplices)):
            if len(sorted_simplices[j])==len(sorted_simplices[i])+1:
                if set(sorted_simplices[i]).issubset(set(sorted_simplices[j])):
                    D[i,j] = 1
    return D


def get_low(R,j):
    """
    get the low value of column j for incidence matrix R
    """
    if R[:,j].sum() == 0:
        return None
    else:
        return np.where(R[:,j] != 0)[0][-1]
    
def get_all_low(R):
    """
    get the low values of all columns in incidence matrix R
    """
    return np.array([get_low(R,j) for j in range(R.shape[1])])

def reduce_R(D, verbose = False):
    """
    Compute the reduced form of the incidence matrix D.
    """
    R = D.copy()
    V = np.identity(R.shape[0])
    for j in range(R.shape[1]):

        low_j = get_low(R,j)
        
        if verbose:
            print(f"j = {j}")
            print(low_j)
        
        low_other = np.array([get_low(R,i) for i in range(R.shape[1]) if i != j])
        low_idx = np.array([i for i in range(R.shape[1]) if i != j])
        matches = low_idx[np.where(low_other == low_j)[0]]
        matches = matches[matches < j]
        while ((len(matches)>0) and (low_j is not None)):
            if verbose:
                print("matches")
                print(matches)
            R[:,j] = (R[:,j] + R[:,matches[0]]) % 2
            V[matches[0],j] += 1
            low_j = get_low(R,j)
            low_other = np.array([get_low(R,i) for i in range(R.shape[1]) if i != j])
            low_idx = np.array([i for i in range(R.shape[1]) if i != j])
            matches = low_idx[np.where(low_other == low_j)[0]]
            matches = matches[matches < j] 

    from sympy.matrices import Matrix
    U = Matrix(V)
    U = np.array(U.inv_mod(2)).astype(int)
    return R, U

def get_pairs(R):
    lows = [get_low(R,j) for j in range(R.shape[1])]
    pairs = [(lows[j],j) for j in range(len(lows)) if lows[j] is not None]
    essential_pairs = [(j,) for j in range(len(lows)) if (lows[j] is None) and (j not in lows)]

    pairs_all = pairs + essential_pairs

    simplex_ids = [p[-1] for p in pairs_all]
    
    return pairs_all, simplex_ids

def get_persistence_pairs_interval(pairs, dims, filtrations, p = 0, time = 0):
    assert time >= 0
    assert time <= 1
    assert len(filtrations)==2
    
    filtration = filtrations[0] * (1-time) + filtrations[1] * time
    
    persistence_pairs = get_persistence_pairs(pairs,dims, filtration, p = p)
    
    return persistence_pairs

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




if __name__ == "__main__":

    # simplicial_complex = [0,1,2,3]
    # dist = np.abs(np.random.randn(4,4))
    # dist = dist - np.diag(np.diag(dist))
    # dist = dist + dist.T

    # all_simplices = list(set(expand_complex(simplicial_complex)))

    # sorted_simplices, sorted_values = sort_simplices(simplicial_complex, all_simplices, dist)

    # dims = [len(simplex)-1 for simplex in sorted_simplices]
    # D = get_incidence_matrix(sorted_simplices)

    # R,U = reduce_R(D)
    # pairs, _ = get_pairs(R)
    # persistence_pairs = get_persistence_pairs(pairs,dims, sorted_values, p="all")
    # print(persistence_pairs)

    # #OINEUS
    # f = np.random.randn(8, 8, 8)
    # max_dim = 2
    # params = oin.ReductionParams()
    # params.n_threads = 16
    # dgms = oin.compute_diagrams_ls(data=f, negate=False, wrap=False, params=params, include_inf_points=True, max_dim=max_dim)
    # dgm = dgms.in_dimension(0)

    
    # colors_dict = color_dict = {0:"red", 1:"blue", 2:"green", 3:"orange"}
    # #convert pairs
    # fig=plt.figure()
    # ax=fig.add_axes([0,0,1,1])
    # ax.set_xlabel("Birth")
    # ax.set_xlabel("Death")
    # for i in range(max_dim+1):
    #     dgm = dgms.in_dimension(i)
    #     ax.scatter(dgm.T[0],dgm.T[1],c=color_dict[i])
    
    # ax.plot(np.linspace(0,10),np.linspace(0,10))

    
    # # fig.savefig("oineus_dgms.png")



    #VR Reduction
    np.random.seed(1)
    n_points = 20
    dim = 3
    points = np.random.uniform(size=(n_points, dim))


    # wrap: set to True to work on torus (periodic boundary conditions)
    fil = oin.get_vr_filtration(points=points, max_dim=3, max_radius=2, n_threads=1)

    cells = fil.cells()

    # Vertices in cells are ids, not sorted_ids
    print(f"Filtration with {len(cells)} cells created,\nvertex 0: {cells[0]},\nlast simplex: {cells[-1]}")

    # no cohomology
    dualize = False
    # create VRU decomposition object, does not perform reduction yet
    dcmp = oin.Decomposition(fil, dualize)


    # reduction parameters
    # relevant members:
    # rp.clearing_opt --- whether you want to use clearing, True by default
    # rp.compute_v: True by default
    # rp.n_threads: number of threads to use, default is 1
    # rp. compute_u: False by default (cannot do it in multi-threaded mode, so switch off just to be on the safe side)
    rp = oin.ReductionParams()

    # perform reduction
    dcmp.reduce(rp)

    # now we can acess V, R and U
    # indices are sorted_ids of simplices == indices in fil.cells()
    V = dcmp.v_data
    R = dcmp.r_data
    
    
    print(f"Example of a V column: {V[-1]}, this chain contains cells:")
    # for i,col in enumerate(V[:-1]):
    #     print(f"Col: {i}")
    #     # for sigma_idx in col:
    #     #     # print(cells[sigma_idx])

    # get diagram, including points at infinity
    include_inf_points=True
    dgm = dcmp.diagram(fil, include_inf_points)
