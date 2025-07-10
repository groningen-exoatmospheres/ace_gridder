# Ace Gridder
Python package for creating accelerated chemistry for exoplanet atmospheres via interpolations.

## Installation

The following commands will build and install ace_gridder:
```bash
git clone https://github.com/groningen-exoatmospheres/ace_gridder.git
cd ace_gridder
pip install .
```

## Features

ace_gridder constructs precomputed volume mixing ratio grids over a range of metallicity, C/O ratio, temperature and pressure using the acepython package. The range, scaling method and number of grid points can be customized. Grids are stored in hdf5 format along with a convergence map. ace_gridder can also load these precomputed grids and has build in methods for interpolation and error analysis. See ace_gridder/grid_constructor.py and ace_gridder/grid_interpolator.py for more information on how to create, load and analyse these grids.

## Example

See jypyter notebook 'Example_notebook.ipynb'
