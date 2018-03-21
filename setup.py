#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for farneback3d.

    This file was generated with PyScaffold 2.5.7.post0.dev6+ngcef9d7f, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import sys
import os
from setuptools import setup


on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    if not on_rtd:
        setup(setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'] + sphinx,
          use_pyscaffold=True)
    else:
        setup(setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'] + sphinx,
                install_requires=[],
                tests_require=[],
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
