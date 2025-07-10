import numpy as np
import h5py

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from acepython import run_ace
from astropy import units as u

class GridConstructor:
    
    """
    Class that helps with constructing precomputed volume mixing ratio grids using acepython.
    
    Constructs 4 dimensional grid spanning metallicity, C/O ratio, temperature and pressure.
    Number of points, ranges and scaling methods can be specified.
    
    Parameters
    ----------
    minima : tuple, optional
        Minimum values for the grid axes: (metallicity, C/O, temperature, pressure)
        Defaults to (0.01, 0.001, 200, 1e-5)
    
    maxima : tuple, optional
        Maximum values for the grid axes: (metallicity, C/O, temperature, pressure)
        Defaults to (2000, 3, 4000, 1e6)
    
    scaling : tuple of str, optional
        Scaling method for each axis, either 'linear' or 'logarithmic'.
        Defaults to ('logarithmic', 'linear', 'linear', 'logarithmic')
    
    species : list of str, optional
        List of chemical species included in the grid.
        Defaults to ["H2", "N2", "H2O", "CH4", "CO2", "CO", "He", "NH3", "HCN"]
    
    Methods
    -------
    construct_grid(wdir, name, n=30)
        Generates the grid by sampling the volume mixing ratios for each combination of
        metallicity, C/O, temperatue and pressure and stores it as an HDF5 file along
        with a convergence map. The variable n represents the amount of grid points for
        each axis.
    """
    
    def __init__(self,
                 minima=(0.01,0.001,200,1e-5),
                 maxima=(2000,3,4000,1e6),
                 scaling=('logarithmic','linear','linear','logarithmic'),
                 species=["H2", "N2", "H2O", "CH4", "CO2", "CO", "He", "NH3", "HCN"]):
        
        self.Z_min, self.CO_min, self.T_min, self.P_min = minima
        self.Z_max, self.CO_max, self.T_max, self.P_max = maxima
        self.Z_scaling, self.CO_scaling, self.T_scaling, self.P_scaling = scaling
        self.Z_unit, self.CO_unit, self.T_unit, self.P_unit = '[n(O)/n(H)] / [n(O)/n(H)]_sun', 'n(C)/n(O)','K', 'Pa'
        self.species = species
    
    def construct_grid(self, wdir, name, n=30):
        
        #--------------< Prepare >--------------#
        
        T_vals  = _create_axis_values(self.T_min,  self.T_max,  n, self.T_scaling)
        P_vals  = _create_axis_values(self.P_min,  self.P_max,  n, self.P_scaling)
        Z_vals  = _create_axis_values(self.Z_min,  self.Z_max,  n, self.Z_scaling)
        CO_vals = _create_axis_values(self.CO_min, self.CO_max, n, self.CO_scaling)
        
        log_Z_vals  = np.log10(Z_vals)
        log_CO_vals = np.log10(CO_vals)
        
        stored_species = self.species
        
        #--------------< Create data >--------------#
        
        data = np.empty(shape=[n,n,n,n,len(stored_species)])
        
        ij_jobs = [(i, j) for i in range(n) for j in range(n)]
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            for i, j, block in tqdm(pool.map(
                    partial(_worker_ij,
                            n=n,
                            log_Z_vals=log_Z_vals, 
                            log_CO_vals=log_CO_vals, 
                            T_vals=T_vals, 
                            P_vals=P_vals,
                            stored_species=stored_species),
                    ij_jobs, chunksize=1), total=len(ij_jobs)):
                data[i, j] = block
        
        #--------------< Create convergence map >--------------#
        
        data_converged = np.zeros((n, n, n, n), dtype=bool)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):

                        VMR = data[i,j,k,l]
                        converged = True

                        for m in range(len(stored_species)):
                            x = VMR[m]
                            if x <= 0 or np.isnan(x) or np.isinf(x):
                                converged = False

                        data_converged[i,j,k,l] = converged
        
        #--------------< Store data as hdf5 file >--------------#
        
        data = np.nan_to_num(data, nan=0.0)
        
        with h5py.File(wdir + name, "w") as f:
            f.attrs["temperature_min"]     = self.T_min
            f.attrs["temperature_max"]     = self.T_max
            f.attrs["temperature_scaling"] = self.T_scaling
            f.attrs["temperature_unit"]    = self.T_unit

            f.attrs["pressure_min"]        = self.P_min
            f.attrs["pressure_max"]        = self.P_max
            f.attrs["pressure_scaling"]    = self.P_scaling
            f.attrs["pressure_unit"]       = self.P_unit

            f.attrs["metallicity_min"]     = self.Z_min
            f.attrs["metallicity_max"]     = self.Z_max
            f.attrs["metallicity_scaling"] = self.Z_scaling
            f.attrs["metallicity_unit"]    = self.Z_unit

            f.attrs["C/O_min"]             = self.CO_min
            f.attrs["C/O_max"]             = self.CO_max
            f.attrs["C/O_scaling"]         = self.CO_scaling
            f.attrs["C/O_unit"]            = self.CO_unit

            f.attrs["stored_species"] = ["H2", "N2", "H2O", "CH4", "CO2", "CO", "He", "NH3", "HCN"]

            f.create_dataset("data", data=data)
            f.create_dataset("map_converged", data=data_converged)
    

def _create_axis_values(axis_min, axis_max, axis_num_values, axis_scaling):
    if axis_scaling == 'linear':
        return np.linspace(axis_min, axis_max, axis_num_values)
    elif axis_scaling == 'logarithmic':
        return np.logspace(np.log10(axis_min), np.log10(axis_max), axis_num_values)
    else:
        return None

def _worker_ij(args, n, log_Z_vals, log_CO_vals, T_vals, P_vals, stored_species):
    i, j = args
    log_Z  = log_Z_vals[i]
    log_CO = log_CO_vals[j]

    #Values from https://arxiv.org/pdf/1912.00844
    A_H_sun  = 12.00
    A_He_sun = 10.92
    A_N_sun  =  7.85
    A_O_sun  =  8.71

    elements = np.array(["H", "He", "C", "N", "O"])
    abundances = np.array([A_H_sun, A_He_sun, 0, 0, 0])
    
    abundances[3] = A_N_sun + log_Z             #A(N)
    abundances[4] = A_O_sun + log_Z             #A(O)
    abundances[2] = abundances[4] + log_CO      #A(C)
    
    result_block = np.empty((n, n, len(stored_species)))

    for k, T in enumerate(T_vals):
        for l, P in enumerate(P_vals):
            species, mix, _ = run_ace(
                T << u.K,
                P << u.Pa,
                elements=elements,
                abundances=abundances,
            )

            VMR = []
            for spec in stored_species:
                if spec in species:
                    species_index = species.index(spec)
                    VMR.append(mix[species_index])

            VMR = np.array(VMR).T.reshape(len(stored_species))

            result_block[k, l] = np.asarray(VMR, dtype=float)

    return i, j, result_block
    