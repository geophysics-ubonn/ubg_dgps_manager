#!/usr/bin/env python
# import os
# import glob

from setuptools import setup
from setuptools import find_packages

version_long = '0.2.0'

# generate entry points
# entry_points = {'console_scripts': []}
# scripts = [
#          os.path.basename(script)[0:-3] for script in glob.glob('src/*.py')]
# for script in scripts:
#     print(script)
#     entry_points['console_scripts'].append(
#         '{0} = {0}:main'.format(script)
#     )

# print(scripts, entry_points)

if __name__ == '__main__':
    setup(
        name='dpgs_manager',
        version=version_long,
        description='dGPS manager Geophysics Uni Bonn',
        author='Maximilian Weigand',
        author_email='mweigand@geo.uni-bonn.de',
        license='MIT',
        # url='https://github.com/geophysics-ubonn/reda',
        # packages=['dpgs_manager', ],
        # package_dir={
        #     # '': 'src',
        #     'dgps_manager': 'lib/dgps_manager',
        # },
        packages=find_packages("lib"),
        package_dir={'': 'lib'},

        # py_modules=scripts,
        # entry_points=entry_points,
        install_requires=[
        ],
    )
