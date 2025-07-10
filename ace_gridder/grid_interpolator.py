import numpy as np
import h5py

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion

from acepython import run_ace
from astropy import units as u

class GridInterpolator:
    """
    Class that helps reading and analysing the precomputed grids from the hdf5 files.
    
    Getter Functions
    ----------------
    get_data(self)
        Returns the data as a 5 dimensional array of axes: metallicity, co_ratio, temperature, pressure, species.
        Data points represent volume mixing ratios and are stored as a 64 bit floating point value.
    
    get_converged_map(self)
        Returns the converged map
    
    get_eroded_map(self)
        Returns the eroded map
    
    def get_species(self):
        Returns a list of the stored species
    
    def get_metallicity_axis_values(self):
        Returns the metallicity axis points
    
    def get_co_ratio_axis_values(self):
        Returns the co_ratio axis points
    
    def get_temperature_axis_values(self):
        Returns the temperature axis points
    
    def get_pressure_axis_values(self):
        Returns the pressure axis points
    
    def get_axis_scaling(self):
        Retursn the scaling method used for each axis, either 'linear' or 'logarithmic'.
    
    Methods
    -------
    load_data(self, directory, filename)
        Loads the data from the given hdf5 file and computes an eroded mask for the is_interpolable function.
    
    get_random_params(self, n)
        Returns random points within the bounds of the parameter space.
        
    is_interpolable(self, params)
        Returns a boolean list of whether interpolation can occur at each test point.
    
    get_interpolator(self, method, scaling=None)
        Returns a scipy RegularGridInterpolator function using the interpolation method 'method'
        and scales the axes according to the 'scaling' variable. 'scaling' defaults to the standard grid
        scaling of each axis.
    
    interpolate(self, params, interpolator)
        Uses the scipy RegularGridInterpolator function returned by get_interpolator to interpolate the grid
        at a list of test points 'params'.
    
    calculate_logarithmic_error(self, VMR_interpolated, VMR_acepython)
        Returns the logarithmic error(s) between interpolated value(s) and direct acepython result(s).
    
    def scale_params(self, params, scaling=None)
        Takes the logarithm of values along a certain axis if that axis has scaling set to 'logarithmic'.
        'scaling' defaults to the standard grid scaling of each axis.
    
    def VMR_acepython(self, params)
        Calculates and returns the volume mixing ratios outputted by acepython at each test point.
    """
    
    def __init__(self):
        
        pass
    
    def load_data(self, directory, filename):
        
        with h5py.File(directory + filename, "r") as f:
            self.temperature_min     = f.attrs["temperature_min"]
            self.temperature_max     = f.attrs["temperature_max"]
            self.temperature_scaling = f.attrs["temperature_scaling"]
            self.temperature_unit    = f.attrs["temperature_unit"]
            
            self.pressure_min     = f.attrs["pressure_min"]
            self.pressure_max     = f.attrs["pressure_max"]
            self.pressure_scaling = f.attrs["pressure_scaling"]
            self.pressure_unit    = f.attrs["pressure_unit"]
            
            self.metallicity_min     = f.attrs["metallicity_min"]
            self.metallicity_max     = f.attrs["metallicity_max"]
            self.metallicity_scaling = f.attrs["metallicity_scaling"]
            self.metallicity_unit    = f.attrs["metallicity_unit"]
            
            self.co_ratio_min     = f.attrs["C/O_min"]
            self.co_ratio_max     = f.attrs["C/O_max"]
            self.co_ratio_scaling = f.attrs["C/O_scaling"]
            self.co_ratio_unit    = f.attrs["C/O_unit"]
            
            self.grid_scaling = [self.metallicity_scaling, self.co_ratio_scaling, self.temperature_scaling, self.pressure_scaling]
            
            n = len(f["data"][:])
            
            self.temperatures   = self.create_axis_values(self.temperature_min, self.temperature_max, n, self.temperature_scaling)
            self.pressures      = self.create_axis_values(self.pressure_min,    self.pressure_max,    n, self.pressure_scaling)
            self.metallicities  = self.create_axis_values(self.metallicity_min, self.metallicity_max, n, self.metallicity_scaling)
            self.co_ratios      = self.create_axis_values(self.co_ratio_min,    self.co_ratio_max,    n, self.co_ratio_scaling)
            
            self.stored_species = f.attrs["stored_species"]
            
            self.data = f["data"][:]
            self.map_converged = f["map_converged"][:]
            
            structure = np.ones((3, 3, 3, 3), dtype=bool)
            self.eroded_mask = binary_erosion(self.map_converged, structure=structure, border_value=1)
            self.eroded_mask = binary_erosion(self.eroded_mask, structure=structure, border_value=1)
    
    #--------------< Getter functions >--------------#
    
    def get_data(self):
        return self.data
    
    def get_converged_map(self):
        return self.map_converged
    
    def get_eroded_map(self):
        return self.eroded_mask
    
    def get_species(self):
        return self.stored_species
    
    def get_metallicity_axis_values(self):
        return self.metallicities
    
    def get_co_ratio_axis_values(self):
        return self.co_ratios
    
    def get_temperature_axis_values(self):
        return self.temperatures
    
    def get_pressure_axis_values(self):
        return self.pressures
    
    def get_axis_scaling(self):
        return self.grid_scaling
    
    #--------------< Core functions >--------------#
    
    def get_random_params(self, n):
        
        rnd_temperature = self.get_random_values_in_range(self.temperature_min, self.temperature_max, n, self.temperature_scaling)
        rnd_pressure    = self.get_random_values_in_range(self.pressure_min,    self.pressure_max,    n, self.pressure_scaling)
        rnd_metallicity = self.get_random_values_in_range(self.metallicity_min, self.metallicity_max, n, self.metallicity_scaling)
        rnd_co_ratio    = self.get_random_values_in_range(self.co_ratio_min,    self.co_ratio_max,    n, self.co_ratio_scaling)
        
        return np.array(list(zip(rnd_metallicity, rnd_co_ratio, rnd_temperature, rnd_pressure)))
    
    def get_interpolator(self, method, scaling=None):
        if scaling == None:
            scaling = self.grid_scaling
        
        interp_points_metallicities = self.scale_axis_values(self.metallicities, scaling[0])
        interp_points_co_ratios     = self.scale_axis_values(self.co_ratios,     scaling[1])
        interp_points_temperature   = self.scale_axis_values(self.temperatures,  scaling[2])
        interp_points_pressures     = self.scale_axis_values(self.pressures,     scaling[3])
        
        interp_points_tuple = (interp_points_metallicities, interp_points_co_ratios, interp_points_temperature, interp_points_pressures)
        
        return RegularGridInterpolator(interp_points_tuple, self.data, method=method)
    
    def interpolate(self, params, interpolator):
        """
        Parameter order: (metallicity, co_ratio, temperature, pressure)
        """
        return self.stored_species, interpolator(params)
    
    def calculate_logarithmic_error(self, VMR_interpolated, VMR_acepython):
        
        return np.log10(VMR_interpolated) - np.log10(VMR_acepython)
    
    def is_interpolable(self, params):
        """
        input must be list of tuples or 2d list.
        """
        interpolable = np.zeros(len(params), dtype=bool)
        
        for m, params_m in enumerate(params):
            i = np.searchsorted(self.metallicities, params_m[0], side='right') - 1
            j = np.searchsorted(self.co_ratios,     params_m[1], side='right') - 1
            k = np.searchsorted(self.temperatures,  params_m[2], side='right') - 1
            l = np.searchsorted(self.pressures,     params_m[3], side='right') - 1
            interpolable[m] = self.eroded_mask[i,j,k,l]
        
        return interpolable
    
    def scale_params(self, params, scaling=None):
        if scaling == None:
            scaling = self.grid_scaling
        
        metallicity, co_ratio, temperature, pressure = params.T
        
        metallicity = self.scale_axis_values(metallicity, scaling[0])
        co_ratio    = self.scale_axis_values(co_ratio,    scaling[1])
        temperature = self.scale_axis_values(temperature, scaling[2])
        pressure    = self.scale_axis_values(pressure,    scaling[3])
        
        return np.array(list(zip(metallicity, co_ratio, temperature, pressure)))
    
    def VMR_acepython(self, params):
        
        #Values from https://arxiv.org/pdf/1912.00844
        A_H_sun  = 12.00
        A_He_sun = 10.92
        A_C_sun  =  8.47
        A_N_sun  =  7.85
        A_O_sun  =  8.71
        
        elements = np.array(["H", "He", "C", "N", "O"])
        abundances = np.array([A_H_sun, A_He_sun, 0, 0, 0])
        
        VMR = np.zeros((len(params), len(self.stored_species)))
        
        for i, params_i in enumerate(params):
            metallicity, co_ratio, temperature, pressure = params_i
            
            log_Z = np.log10(metallicity)
            log_CO = np.log10(co_ratio)
            
            abundances[3] = A_N_sun + log_Z         #A(N)
            abundances[4] = A_O_sun + log_Z         #A(O)
            abundances[2] = abundances[4] + log_CO  #A(C)
            
            species, mix_profile, mu_profile = run_ace(
                np.array([temperature]) << u.K,
                np.array([pressure]) << u.Pa,
                elements = elements,
                abundances = abundances,
            )
            
            VMR_i = []
            for spec in self.stored_species:
                if spec in species:
                    k = species.index(spec)
                    VMR_i.append(mix_profile[k])
            
            VMR[i] = np.array(VMR_i).T.reshape(len(self.stored_species))
        
        return self.stored_species, VMR
    
    #-------------< Helper functions >-------------#
    
    def create_axis_values(self, axis_min, axis_max, axis_num_values, axis_scaling):
        if axis_scaling == 'linear':
            return np.linspace(axis_min, axis_max, axis_num_values)
        elif axis_scaling == 'logarithmic':
            return np.logspace(np.log10(axis_min), np.log10(axis_max), axis_num_values)
        else:
            return None
    
    def scale_axis_values(self, axis_values, axis_scaling):
        if axis_scaling == 'linear':
            return axis_values
        elif axis_scaling == 'logarithmic':
            return np.log10(axis_values)
        else:
            return None
    
    def get_random_values_in_range(self, minimum, maximum, n, axis_scaling):
        if axis_scaling == 'linear':
            return np.random.uniform(low=minimum, high=maximum, size=n)
        elif axis_scaling == 'logarithmic':
            return 10**np.random.uniform(low=np.log10(minimum), high=np.log10(maximum), size=n)
        else:
            return None
    