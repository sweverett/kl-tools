import matplotlib.pyplot as plt
import numpy as np

from cube import CubePars

class TNGSlit():
    def __init__(self, pars, **kwargs):
        """Pass a dictionary of metapars/CubePars object and it computes the 3D datacube
        for a TNG subhalo
        The datacube is set as an attribute

        Parameters
        ----------
        pars : dict, CubePars
            _description_
        subhalo_id : int, optional
            _description_, by default 2
        """        
        
        if not isinstance(pars, CubePars):
            self.cubepars = CubePars(pars)
        
        else:
            self.cubepars = pars
            
    
    def plot_datavector(self):
        plt.imshow(self.datavector, origin='lower')
        plt.savefig('slit_spec.png')
        plt.show()