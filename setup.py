from setuptools import setup, find_packages

setup(
    name='ace_gridder',
    version='0.1',
    package_dir={'': 'ace_gridder'},
    packages=find_packages(where='ace_gridder'),
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "tqdm",
        "acepython",
        "astropy"
    ],
    author='Tristan Popken',
    description='Python package for creating accelerated chemistry for exoplanet atmospheres via interpolations.',
    url='https://github.com/groningen-exoatmospheres/ace_gridder',
)
