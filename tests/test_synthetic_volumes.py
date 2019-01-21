#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import farneback3d
import numpy as np


__author__ = "Stephan Seitz"
__copyright__ = "Stephan Seitz"
__license__ = "none"

TOLERANCE = 1e-2


def test_moving_cube():

    movement_vector = [0.3, 0.1, 1]

    vol0 = np.zeros([60, 60, 60], np.float32)
    vol0[10:15, 20:25, 30:35] = 100.
    # flow_ground_truth = 4* np.ones([*vol0.shape,3], np.float32)
    flow_ground_truth = np.stack([movement_vector[2] * np.ones(vol0.shape, np.float32), movement_vector[1] * np.ones(
        vol0.shape, np.float32), movement_vector[0] * np.ones(vol0.shape, np.float32)], 0)

    vol1 = farneback3d.warp_by_flow(vol0, flow_ground_truth)

    optflow = farneback3d.Farneback(
        levels=5, num_iterations=5, poly_n=5, quit_at_level=-1, use_gpu=True, fast_gpu_scaling=True)
    flow = optflow.calc_flow(vol0, vol1)

    for axis in range(len(movement_vector)):
        print(np.max(flow[2-axis]) - movement_vector[axis])
        assert abs(np.max(flow[2-axis]) - movement_vector[axis]) < TOLERANCE


def test_moving_cube_larger_distance():

    movement_vector = [4, 1, 1]

    vol0 = np.zeros([60, 60, 60], np.float32)
    vol0[10:15, 20:25, 30:35] = 100.
    # flow_ground_truth = 4* np.ones([*vol0.shape,3], np.float32)
    flow_ground_truth = np.stack([movement_vector[2] * np.ones(vol0.shape, np.float32), movement_vector[1] * np.ones(
        vol0.shape, np.float32), movement_vector[0] * np.ones(vol0.shape, np.float32)], 0)

    vol1 = farneback3d.warp_by_flow(vol0, flow_ground_truth)

    optflow = farneback3d.Farneback(
        levels=5, num_iterations=5, poly_n=5, quit_at_level=-1, use_gpu=True, fast_gpu_scaling=True)
    flow = optflow.calc_flow(vol0, vol1)

    for axis in [0]:
        print(np.max(flow[2-axis]) - movement_vector[axis])
        assert abs(np.max(flow[2-axis]) - movement_vector[axis]) < TOLERANCE


def test_default_values():

    a = np.ones([20] * 3)
    b = np.ones([20] * 3)

    optflow = farneback3d.Farneback()

    rtn = optflow.calc_flow(a, b)
    assert rtn is not None


if __name__ == "__main__":
    # test_moving_cube()
    # test_moving_cube_larger_distance()
    test_default_values()
