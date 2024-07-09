#!/usr/bin/env python

from distutils.core import setup
from distutils import util

if __name__ == "__main__":

    pathMySubPackage1 = util.convert_path('nmrTPy/process')
    pathMySubPackage2 = util.convert_path('nmrTPy/plot')
    setup(
        name = 'nmrTPy',
        version = '0.1',
        install_requires = ['numpy', 'scipy', 'lmfit', 'pandas','nmrglue'],
        package_dir = {
            'nmrTPy': 'nmrTPy',
            'nmrTPy.process': pathMySubPackage1,
            'mnmrTPy.plot': pathMySubPackage2},
        packages = ['nmrTPy', 'nmrTPy.process',
                  'nmrTPy.plot']
    )