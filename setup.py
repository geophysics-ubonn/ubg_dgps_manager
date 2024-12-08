#!/usr/bin/env python
import os
import glob

from setuptools import setup
from setuptools import find_packages

version_long = '0.3.0'

# package data
os.chdir('lib/ubg_dgps_manager/')
package_data = glob.glob('help_text/*')
print(package_data)
os.chdir('../../')


if __name__ == '__main__':
    setup(
        name='ubg_dgps_manager',
        version=version_long,
        description='dGPS manager of the Geophysics Section of the University of Bonn',
        author='Maximilian Weigand',
        author_email='mweigand@geo.uni-bonn.de',
        license='MIT',
        url='https://github.com/geophysics-ubonn/ubg_dgps_manager',
        packages=find_packages("lib"),
        package_dir={'': 'lib'},
        package_data={
            'ubg_dgps_manager': package_data
        },
        install_requires=[
            'cartopy',
            'geojson',
            'numpy',
            'pyproj',
            'shapely',
            'ipywidgets',
            'crtomo_tools',
            'markdown',
        ],
    )
