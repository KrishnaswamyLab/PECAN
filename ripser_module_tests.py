# Testing the ripser python interface as a replacement for the C++ command currently used
from ripser import ripser 
import numpy as np 
from persim import plot_diagrams

data = np.random.random((100,2))
diagrams = ripser(data)

print(diagrams)

usable_diagrams = diagrams['dgms']
plot_diagrams(usable_diagrams, show=True)