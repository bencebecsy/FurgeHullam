#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2024 Bence Becsy
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Setup script for FurgeHullam
can install like this: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    author='Bence Becsy',
    name='FurgeHullam',
    version='1.0.0',
    install_requires=[
         'numpy',
         'scipy',
         #'enterprise_extensions',
         'numba',
         'numba_scipy',
         'quantecon',
    ],
    python_requires='>=3.7',
    packages=find_packages(include=['FurgeHullam']),
    long_description=open('README.md').read(),
    )
