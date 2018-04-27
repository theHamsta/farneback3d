===========
farneback3d
===========

.. image:: https://badge.fury.io/py/farneback3d.svg
    :target: https://badge.fury.io/py/farneback3d
.. image:: https://travis-ci.org/theHamsta/farnback3d.svg?branch=master
    :target: https://travis-ci.org/theHamsta/farnback3d


A CUDA implementation of the Farneback optical flow algorithm [1]_ for the calculation of dense volumetric flow fields. Since this algorithm is based on the approximation of the signal by polynomial expansion it is especial suited for the motion estimation in smooth signals without clear edges.

To know more about the implementation, have a look on `this OpenCV class <https://docs.opencv.org/3.3.0/de/d9e/classcv_1_1FarnebackOpticalFlow.html>`_ that was used as a template for this implementation.

Development takes place on `Github <https://github.com/theHamsta/farneback3d>`_.

Python interface
================

The project uses `pycuda <https://github.com/inducer/pycuda>`_ to provide a pure-python package available on PyPi

.. code-block:: bash

    pip install farneback3d

Usage:

.. code-block:: python

    import farneback3d

    ... # create some numpy volumes vol0 and vol1 (can also be pycuda GPUArrays) 

    # set parameters for optical flow (parameters have the same meaning as in the OpenCV implementation)
    optflow = farneback3d.Farneback(
            pyr_scale=0.8,         # Scaling between multi-scale pyramid levels
            levels=6,              # Number of multi-scale levels
            num_iterations=5,      # Iterations on each multi-scale level
            winsize=9,             # Window size for Gaussian filtering of polynomial coefficients
            poly_n=5,              # Size of window for weighted least-square estimation of polynomial coefficients
            poly_sigma=1.2,        # Sigma for Gaussian weighting of least-square estimation of polynomial coefficients
        )

    # calculate frame-to-frame flow between vol0 and vol1
    flow = optflow.calc_flow(vol0, vol1)


C++ interface
=============

To be implemented...


Future plans
=============

The current implementation uses a naive approach to perform the necessary convolutions.
The algorithm could be sped up drastically by performing separable convolutions along each coordinate axis.

References
==========

.. [1] Farnebäck, Gunnar. "Two-frame motion estimation based on polynomial expansion." Scandinavian conference on Image analysis. Springer, Berlin, Heidelberg, 2003.
