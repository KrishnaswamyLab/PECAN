import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import oineus as oin


def get_low(R,j):
    """
    get the low value of column j for incidence matrix R
    """
    if R[:,j].sum() == 0:
        return None
    else:
        return np.where(R[:,j] != 0)[0][-1]

def permute(R,i,j):
    """
    Permute the rows and columns of the matrix R according to the permutation (i,j)
    """
    R_ = R.copy()
    R_[[i,j]] = R_[[j,i]]
    R_[:,[i,j]] = R_[:,[j,i]]
    return R_


def process_permutations(R,U,i,j):
    """
    Process the permutations of the matrix R and U according to the permutation (i,j)
    """
    assert j == i+1

    all_lows = get_all_low(R)
    if (get_low(R,i) is None) and (get_low(R,j) is None):
        print("Case 1")
        R,U = case_1(R,U,i,j)
    elif (get_low(R,i) is not None) and (get_low(R,j) is not None):
        print("Case 2")
        R,U = case_2(R,U,i,j)
    elif (get_low(R,i) is not None) and (get_low(R,j) is None):
        print("Case 3")
        R,U = case_3(R,U,i,j)
    elif (get_low(R,i) is None) and (get_low(R,j) is not None):
        print("Case 4")
        R, U = case_4(R,U,i,j)

    return R, U


def case_1(R,U,i,j):
    R_ = R.copy()
    U_ = U.copy()
    all_lows = get_all_low(R_)
    #Case 1.1
    if (i in all_lows) and (j in all_lows):
        print("case 1.1")
        k = np.where(all_lows == i)[0][0]
        l = np.where(all_lows == j)[0][0]
        if R_[i,l] == 1:
            if k<l: #Case 1.1.1
                R_ = permute(R_,i,j)
                U_ = permute(U_,i,j)
                R_[:,l] = (R_[:,l] + R_[:,k]) % 2
                U_[l,:] = (U_[l,:] + U_[k,:]) % 2
            elif k>l: #Case 1.1.2
                R_ = permute(R_,i,j)
                U_ = permute(U_,i,j)
                R_[:,k] = (R_[:,k] + R_[:,l]) % 2
                U_[k,:] = (U_[k,:] + U_[l,:]) % 2
    else:
        print("case 1.2")
        R_ = permute(R_,i,j)
        U_ = permute(U_,i,j)
    return R_, U_

def case_2(R,U,i,j):
    #Case 2
    #Check if position of negative simplices.
    R_ = R.copy()
    U_ = U.copy()
    if U_[i,j] == 1:
        U_[j,:] = (U_[j,:] + U_[i,:]) % 2
        R_[:,j] = (R_[:,j] + R_[:,i]) % 2
        if get_low(R_,i) < get_low(R_,j):
            R_ = permute(R_,i,j)
            U_ = permute(U_,i,j)
        elif get_low(R_,i) > get_low(R_,j):
            R_ = permute(R_,i,j)
            U_ = permute(U_,i,j)
            R_[:,j] = (R_[:,j] + R_[:,i]) % 2
            U_[j,:] = (U_[j,:] + U_[i,:]) % 2
    else:
        R_ = permute(R_,i,j)
        U_ = permute(U_,i,j)
    return R_, U_

def case_3(R,U,i,j):
    R_ = R.copy()
    U_ = U.copy()
    if U_[i,j] == 1:
        U_[j,:] = (U_[j,:] + U_[i,:]) % 2
        R_[:,j] = (R_[:,j] + R_[:,i]) % 2
        U_ = permute(U_,i,j)
        R_ = permute(R_,i,j)
        U_[j,:] = (U_[j,:] + U_[i,:]) % 2
        R_[:,j] = (R_[:,j] + R_[:,i]) % 2
    else:
        R_ = permute(R_,i,j)
        U_ = permute(U_,i,j)
    return R_, U_

def case_4(R,U,i,j):
    #Case 4
    R_ = R.copy()
    U_ = U.copy()
    
    if U_[i,j] == 1:
        U_[i,j] = 0
    
    R_ = permute(R_,i,j)
    U_ = permute(U_,i,j)
    return R_, U_      



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



def plot_vineyards(simplex_id, simplex_ids,persistences, marker_dict, color_dict, legend_elements):

    if not isinstance(simplex_id,list):
        simplex_id = [simplex_id]

    fig,ax = plt.subplots()
    
    for s_id in simplex_id:
        x = []
        y = []
        c = []
        t = []
        p = []
        for step in range(len(simplex_ids)):
            id_ = np.where(np.array(simplex_ids[step])== s_id)[0]
            if len(id_)==0:
                continue
            else:
                id = id_.item()
            t.append(times[step])
            x.append(persistences[step][id][0])
            y.append(persistences[step][id][1])
            p.append(y[-1] - x[-1])
            c.append(persistences[step][id][2])
            ax.scatter(t[-1], p[-1], c= color_dict[c[-1]], marker = marker_dict[dims[s_id]])

        ax.plot(t,p, c = "grey")
    ax.set_xlabel("Time")
    ax.set_ylabel("Persistence")
    ax.legend(handles=legend_elements, loc='upper right',bbox_to_anchor=(1.5, 1.))


def forward(original_filtration,new_filtration,dualize=False):
    """
    Get the location of the crossings of the persistence diagrams using linear homotopy.

    Inputs:
    - original_filtration : list
        the value of the simplices in the first filtration
    - D : VR Decompoistion 
        the diagram of first filtration
    - new_filtration : list
        the value of the simplices in the second filtration

    Outputs:
    - D_new
        updated reduced incidence matrix

    """

    # Updating the incidence matrix

    dcmp = oin.Decomposition(original_filtration, dualize)
    rp = oin.ReductionParams()

    # perform reduction
    dcmp.reduce(rp)

    V = dcmp.v_data

    print(dir(V))
    V.sort()

    return V



    # x_cross, cross_idx = get_crossings(original_filtration,new_filtration)

    # cross_i_before_t = [idx for idx in range(len(x_cross)) if x_cross[idx]<=t]
    # cross_idx_before_t = [(cross_idx[0][i], cross_idx[1][i]) for i in cross_i_before_t]
    # cross_x_before_t = [x_cross[idx] for idx in cross_i_before_t]

    # # keeping track of the location of the simplices in the t=0 ordering
    # simplex_location = [i for i in range(len(original_filtration))]

    # D_new = D.copy()

    # persistences = [D]
    # simplex_ids = [simplex_ids]
    # times = [0]

    # for idx in range(len(cross_i_before_t)):
    #     times.append(cross_x_before_t[idx])
    #     simplex_0, simplex_1 = cross_idx_before_t[idx]
    #     swap_idx_unsorted = (simplex_location[simplex_0],simplex_location[simplex_1])
    #     swap_idx = sorted(swap_idx_unsorted)
    #     i = swap_idx[0]
    #     j = swap_idx[1]

    #     if dims[i] != dims[j]:
    #         print("DIM MISMATCH !")
    #         continue
    #     else:
    #         simplex_location[simplex_0] = swap_idx_unsorted[1]
    #         simplex_location[simplex_1] = swap_idx_unsorted[0]
    #         D_new = permute(D_new,i,j)

    #     reverse_loc = np.argsort(simplex_location)



    #     R_updated, _ = reduce_R(D_new)
    #     pairs_updated, simplex_ids_updated = get_pairs(R_updated)

    #     simplex_id_updated = [reverse_loc[v] for v in simplex_ids_updated]

    #     sorted_values_t = np.array(original_filtration) * (1-t) + np.array(new_filtration) * t # linear interpolation of the filtration values at time t according to old ordering

    #     sorted_values_t_updated = [sorted_values_t[v] for v in reverse_loc] # from old order to new order

    #     persistence_pairs_updated = get_persistence_pairs(pairs_updated,dims, sorted_values_t_updated, p="all")
    #     print(persistence_pairs_updated)

    #     persistences.append(persistence_pairs_updated)
    #     simplex_ids.append(simplex_id_updated)

    #     return D_new





if __name__ == "__main__":
    
    
    np.random.seed(1)
    n_points = 20
    dim = 3
    original = np.random.uniform(size=(n_points, dim))

    np.random.seed(2)
    new = np.random.uniform(size=(n_points, dim))

    # wrap: set to True to work on torus (periodic boundary conditions)
    original_filtration = oin.get_vr_filtration(points=original, max_dim=3, max_radius=2, n_threads=1)
    new_filtration = oin.get_vr_filtration(points=new, max_dim=3, max_radius=2, n_threads=1)

    original_cells = original_filtration.cells()
    new_cells = new_filtration.cells()

    # Vertices in cells are ids, not sorted_ids
    print(f"Original Filtration with {len(original_cells)} cells created,\nvertex 0: {original_cells[0]},\nlast simplex: {original_cells[-1]}")
    print(f"New Filtration with {len(new_cells)} cells created,\nvertex 0: {new_cells[0]},\nlast simplex: {new_cells[-1]}")

    V = forward(original_filtration,new_filtration)

    for col in V[-3:]:
        for sigma_idx in col:
            print("Original Filtration")
            print(original_cells[sigma_idx])
            print("New Filtration")
            print(new_cells[sigma_idx])
            print()

    original_filtration_vals = []
    new_filtration_vals = []

    for col in V:
        for idx in col:
            original_filtration_vals.append(original_cells[sigma_idx].value)
            new_filtration_vals.append(new_cells[sigma_idx].value)

    cross_x, cross_idx = get_crossings(np.array(original_filtration_vals),np.array(new_filtration_vals))

    print(cross_x)


