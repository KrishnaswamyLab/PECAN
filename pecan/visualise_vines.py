import numpy as np
import oineus as oin
import matplotlib
import matplotlib.pyplot as plt
import argparse



def get_vineyards(file):
    """Load Vineyard information from condensation output file"""
    data = np.load(file,allow_pickle=True)
    assert "vineyard" in data.keys()," 'Require `vineyard` key'"
    vineyard = data["vineyard"].flatten()[0]
    
    max_radius = data["max_radius"]
    return vineyard,max_radius

def get_interesting_vines(vineyard,threshhold=75,max_radius=5):
    """Select vines to visualise based on the length of the vine and
    the persistence of the feature througout the condensation process."""
    selected_vines = {}
    stats = {}
    dims = []
    M = []
    for simplex,vine in vineyard.items():
        total_persistence = 0
        for feature in vine:
            t,b,d,dim = feature
             # Replace Infty Values with Max Radius in the Filtration
            if np.isinf(d):
                d = max_radius
            total_persistence += np.abs(d - b)
            dims.append(dim)
        info = np.array([total_persistence,len(vine),dim])
        # Collect Stats for Each Simplex
        M.append(info)
        stats[simplex] = info

    # Stacking into Matrix
    M = np.vstack(M)  
    

    # Calculating Upper Quartile for Total Peristence and Time 
    for d in np.unique(dims):
        M_d = M[M[:, 2] == d][:,:2]
        upper = np.percentile(M_d,q=threshhold,axis=0)
        ids = list(np.where((M[:,0] >= upper[0]) & (M[:,1] >= upper[1]) & (M[:,2]==d))[0])
        selected_vines[d] = [list(stats.keys())[i] for i in ids]
    return selected_vines




def make_2d_vine_plot(vineyard,dim=0,threshold=75,max_radius=5):
    """Create 2D vineyard plot for *important* simplexes throughout
     throughout the condensation process."""

    simplices = get_interesting_vines(vineyard,threshold,max_radius)[dim]
    fig,ax = plt.subplots()
    # Tensor of point clouds over diffusion process


    for simplex in simplices:
        vine = vineyard[simplex]
        # print(simplex)
        # print(vine)
        # print()
        times = []
        persistence = []
        for feature in vine:
            t,b,d,dim_ = feature
            if dim_ == dim:
                times.append(t)
                if np.isinf(d):
                    d = max_radius
                persistence.append(np.abs(d-b))
        ax.scatter(times,persistence,s=30,alpha=0.3)
        ax.plot(times,persistence,label=simplex)

    plt.legend()

    #Tick Labels
    ticks =  ax.get_yticks()
    labels = [r"$\infty$" if x >= max_radius else np.round(x,2) for x in ticks]
    plt.yticks(ticks, labels)

    plt.title(f"In the {threshold}th Percentile for Length and Total Persistence.",fontdict={"size":10})
    fig.suptitle(f"Dim {dim} Vines",fontdict={"size":15})


    ax.set_xlabel('$t$')
    ax.set_ylabel('Persistence')
    
    return fig
    


if __name__ == '__main__':
    matplotlib.rcParams['lines.linewidth'] = 0.75

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    parser.add_argument(
        '-d', '--dimension', default=0, type=int,
        help='Homology Dimension'
    )

    parser.add_argument(
        '-P', '--percentile',
        default=90, type=int,
        help='Percentile of Vineyards to display'
    )


    args = parser.parse_args()
    vineyard,max_radius = get_vineyards(args.INPUT)
    fig = make_2d_vine_plot(vineyard,dim=args.dimension,threshold=args.percentile,max_radius=max_radius)
    fig.savefig("./testing_vine_plot.png")


    
    

