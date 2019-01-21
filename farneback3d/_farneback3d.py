"""
3d Python implementation of 2d optical flow "opencv/modules/video/src/optflowgf.cpp"

"""
# TODO make class save fft kernels for fast filtering

import numpy as np
import scipy.ndimage as sciimg
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os
import json

import farneback3d._utils
from farneback3d._utils import divup
import farneback3d._filtering

_NUM_POLY_COEFFICIENTS = 10
_MIN_VOL_SIZE = 32


class Farneback:

    def __init__(self,
                 pyr_scale=0.9,
                 levels=15,
                 winsize=9,
                 num_iterations=5,
                 poly_n=5,
                 poly_sigma=1.2,
                 use_gaussian_kernel:  bool = True,
                 use_initial_flow=None,
                 quit_at_level=None,
                 use_gpu=True,
                 upscale_on_termination=True,
                 fast_gpu_scaling=True,
                 vesselmask_gpu=None
                 ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.num_iterations = num_iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.use_gaussian_kernel = use_gaussian_kernel
        self.use_initial_flow = use_initial_flow
        self.use_gpu = use_gpu
        self.upscale_on_termination = upscale_on_termination
        self._fast_gpu_scaling = fast_gpu_scaling
        self.quit_at_level = quit_at_level
        self._dump_everything = False
        self._show_everything = False
        self._vesselmask_gpu = vesselmask_gpu
        self._resize_kernel_size_factor = 4
        self._max_resize_kernel_size = 9

        with open(os.path.join(os.path.dirname(__file__), 'farneback_kernels.cu')) as f:
            read_data = f.read()
        f.closed

        mod = SourceModule(read_data)
        self._update_matrices_kernel = mod.get_function(
            'FarnebackUpdateMatrices')
        self._invG_gpu = mod.get_global('invG')[0]
        self._weights_gpu = mod.get_global('weights')[0]
        self._poly_expansion_kernel = mod.get_function('calcPolyCoeficients')
        self._warp_kernel = mod.get_function('warpByFlowField')
        self._r1_texture = mod.get_texref('sourceTex')
        self._solve_equations_kernel = mod.get_function('solveEquationsCramer')

    def load_parameters_from_config_file(self, config_file=None):
        if not config_file:
            config_file = os.path.join(
                os.path.dirname(__file__), "optflow.json")

        with open(config_file) as file:
            config_str = file.read()

        config = farneback3d._utils.DictWrapper(json.loads(config_str))

        self.levels = config.levels
        self.num_iterations = config.num_iterations
        self.poly_n = config.poly_n
        self.quit_at_level = config.quit_at_level
        self.upscale_on_termination = config.upscale_on_termination
        self.poly_sigma = config.poly_sigma
        self.winsize = config.winsize
        self.use_gaussian_kernel = config.use_gaussian_kernel
        self._resize_kernel_size_factor = config.resize_kernel_size_factor
        self._max_resize_kernel_size = config.max_resize_kernel_size

    def write_parameters_from_config_file(self, config_file):
        config_dict = {}
        config_dict["levels"] = self.levels
        config_dict["num_iterations"] = self.num_iterations
        config_dict["poly_n"] = self.poly_n
        config_dict["quit_at_level"] = self.quit_at_level
        config_dict["upscale_on_termination"] = self.upscale_on_termination
        config_dict["poly_sigma"] = self.poly_sigma
        config_dict["winsize"] = self.winsize
        config_dict["use_gaussian_kernel"] = self.use_gaussian_kernel
        config_dict["resize_kernel_size_factor"] = self._resize_kernel_size_factor
        config_dict["max_resize_kernel_size"] = self._max_resize_kernel_size
        with open(config_file, 'w') as file:
            json.dump(config_dict, file)

    def calc_flow(self,
                  cur_vol: np.ndarray,
                  next_vol: np.ndarray
                  ):

        dim = len(cur_vol.shape)

        assert dim == 3, 'wrong dimension'
        assert self.quit_at_level is None or self.quit_at_level < self.levels
        assert cur_vol.shape == next_vol.shape
        assert cur_vol.dtype == np.float32, 'wrong dtype'
        assert next_vol.dtype == np.float32, 'wrong dtype'
        assert not self.use_initial_flow

        prev_flow_gpu = None
        imgs = [gpuarray.to_gpu(next_vol), gpuarray.to_gpu(cur_vol)]
        flow_gpu = None

        for k in range(self.levels, -1, -1):
            print('Scale %i' % k)

            if k == self.quit_at_level:
                if k != -1 and self.upscale_on_termination and prev_flow_gpu is not None:
                    flow_gpu = gpuarray.GPUArray(
                        [dim, *cur_vol.shape], cur_vol.dtype)

                    prev_flow_gpu *= 1 / scale
                    for i in range(3):
                        self._resize(
                            prev_flow_gpu[i], flow_gpu[i], dst_shape=cur_vol.shape)

                break

            scale = self.pyr_scale**k

            if np.any(np.array(cur_vol.shape) * scale < _MIN_VOL_SIZE):
                continue

            sigma = (1. / scale - 1) * 0.5
            smooth_sz = int(sigma * self._resize_kernel_size_factor + 0.5) | 1
            smooth_sz = max(smooth_sz, 3)
            smooth_sz = min(smooth_sz, self._max_resize_kernel_size)

            scale_shape = [int(np.round(x * scale)) for x in cur_vol.shape]

            if prev_flow_gpu is not None:
                flow_gpu *= 1 / self.pyr_scale
                scaled_flow_gpu = gpuarray.GPUArray(
                    [3, *scale_shape], np.float32)
                for i in range(3):
                    self._resize(
                        flow_gpu[i], scaled_flow_gpu[i], dst_shape=scale_shape)

                flow_gpu = scaled_flow_gpu

            else:
                flow_gpu = gpuarray.zeros([dim, *scale_shape], np.float32)

            R = gpuarray.GPUArray(
                [2, _NUM_POLY_COEFFICIENTS - 1, *flow_gpu.shape[1:]], np.float32)
            M = gpuarray.GPUArray([9, *flow_gpu.shape[1:]], np.float32)

            for i in range(2):
                if k == 0:
                    I = imgs[i]
                else:
                    I = farneback3d._filtering.smooth_cuda_gauss(imgs[i], sigma,
                                                                 smooth_sz)
                    I = self._resize(I, scaling=scale)

                self._FarnebackPolyExp(I, R[i], self.poly_n, self.poly_sigma)

            # dsareco.visualization.imshow(R[0])
            # dsareco.visualization.imshow(R[1])

            self._FarnebackUpdateMatrices_gpu(R[0], R[1], flow_gpu, M)

            # dsareco.visualization.imshow(M,"M")

            for i in range(self.num_iterations):
                print('iteration %i' % i)
                if self.use_gaussian_kernel:
                    self._FarnebackUpdateFlow_GaussianBlur_gpu(
                        R[0], R[1], flow_gpu, M, self.winsize, i < self.num_iterations - 1)
                else:
                    raise ValueError('only use_gaussian_kernel implemented')
                # dsareco.utils.imshow(np.moveaxis(flow,3,0),"%i, scale %i" %(i,k))
                # dsareco.vtk.imageToVTK('/dos/d/output/flow_%03d_%03d' %(k,i), pointData={'x' : np.ascontiguousarray(flow[:,:,:,0]),  'y': np.ascontiguousarray(flow[:,:,:,1]), 'z': np.ascontiguousarray(flow[:,:,:,2])})

            prev_flow_gpu = flow_gpu

        if flow_gpu is not None:
            return flow_gpu.get()
        else:
            return np.zeros([3, *cur_vol.shape], np.float32)

    def _FarnebackPrepareGaussian(self, n, sigma):
        if sigma < 1e-7:
            sigma = n * 0.3

        x = np.arange(-n, n + 1, dtype=np.float32)
        g = np.exp(-(x * x) / (2 * sigma * sigma))
        g /= np.sum(g)
        # xg = x * g
        # xxg = x * xg

        G = np.zeros(
            (_NUM_POLY_COEFFICIENTS, _NUM_POLY_COEFFICIENTS), np.float32)

        G_half = np.zeros((10, 2 * n + 1, 2 * n + 1, 2 * n + 1), np.float32)

        # G:  sum_xyz weight_xyz *(1 x y z xx yy zz xy xz yz) * (1 x y z xx yy zz xy xz yz)^T

        for z in range(-n, n + 1):
            for y in range(-n, n + 1):
                for x in range(-n, n + 1):
                    gauss_weight = g[z + n] * g[y + n] * g[x + n]
                    base_vector = np.atleast_2d(np.array(
                        [1, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z], np.float32))
                    matrix = np.matmul(np.transpose(base_vector), base_vector)
                    G += matrix * gauss_weight
                    G_half[:, z + n, y + n, x + n] = gauss_weight * base_vector

        invG = np.linalg.inv(G)

        return invG, G_half

    def _FarnebackPolyExp(self, img_gpu, poly_coefficients_gpu, n, sigma):
        assert img_gpu.size == poly_coefficients_gpu[0].size
        assert img_gpu.dtype == np.float32
        assert poly_coefficients_gpu.dtype == np.float32

        invG, G_half = self._FarnebackPrepareGaussian(n, sigma)

        block = (32, 32, 1)
        grid = (int(divup(img_gpu.shape[2], block[0])),
                int(divup(img_gpu.shape[1], block[1])), 1)

        cuda.memcpy_htod(self._invG_gpu, invG)
        cuda.memcpy_htod(self._weights_gpu, G_half)

        self._poly_expansion_kernel(img_gpu,
                                    poly_coefficients_gpu,
                                    np.int32(img_gpu.shape[2]),
                                    np.int32(img_gpu.shape[1]),
                                    np.int32(img_gpu.shape[0]),
                                    np.int32(2 * n + 1),
                                    grid=grid, block=block)

    def _FarnebackUpdateMatrices_gpu(self, R0_gpu, R1_gpu, flow_gpu, M_gpu):

        R1_warped_gpu = gpuarray.empty_like(R1_gpu)

        block = (32, 32, 1)
        grid = (int(divup(flow_gpu.shape[3], block[0])),
                int(divup(flow_gpu.shape[2], block[1])), 1)

        for i in range(_NUM_POLY_COEFFICIENTS - 1):
            farneback3d._utils.ndarray_to_float_tex(
                self._r1_texture, R1_gpu[i])
            self._warp_kernel(
                flow_gpu,
                R1_warped_gpu[i],
                np.int32(flow_gpu.shape[3]),
                np.int32(flow_gpu.shape[2]),
                np.int32(flow_gpu.shape[1]),
                np.float32(1),
                np.float32(1),
                np.float32(1),
                block=block, grid=grid)

        self._update_matrices_kernel(R0_gpu,
                                     R1_warped_gpu,
                                     flow_gpu,
                                     M_gpu,
                                     np.int32(flow_gpu.shape[3]),
                                     np.int32(flow_gpu.shape[2]),
                                     np.int32(flow_gpu.shape[1]),
                                     block=block, grid=grid)

    def _FarnebackUpdateFlow_GaussianBlur_gpu(self, poly_coefficients0, poly_coefficients1, flow_gpu, M, winsize, update_matrices):
        sigma = self.winsize * 0.3

        M_filtered_gpu = gpuarray.GPUArray(M.shape, M.dtype)

        for i in range(M.shape[0]):
            farneback3d._filtering.smooth_cuda_gauss(
                M[i], sigma, winsize, rtn_gpu=M_filtered_gpu[i])

        block = (32, 32, 1)
        grid = (int(divup(flow_gpu.shape[3], block[0])),
                int(divup(flow_gpu.shape[2], block[1])), 1)

        self._solve_equations_kernel(M_filtered_gpu, flow_gpu, np.int32(flow_gpu.shape[3]), np.int32(
            flow_gpu.shape[2]), np.int32(flow_gpu.shape[1]), block=block, grid=grid)

        if update_matrices:
            self._FarnebackUpdateMatrices_gpu(
                poly_coefficients0, poly_coefficients1, flow_gpu, M)

    def _resize(self, src_vol, dst_vol=None, dst_shape=None, scaling=None):

        if self._fast_gpu_scaling:
            return farneback3d._utils.resize_gpu(src_vol, dst_vol, dst_shape, scaling)
        else:
            if scaling:
                scaling_vec = [scaling, scaling, scaling]
            else:
                assert dst_shape is not None
                scaling_vec = [dst_shape[i] / src_vol.shape[i]
                               for i in range(len(dst_shape))]

            result = sciimg.zoom(src_vol.get(), scaling_vec)

            if dst_vol is None:
                dst_vol = gpuarray.to_gpu(result)
            else:
                dst_vol.set(result)

            return dst_vol


def warp_by_flow(vol, flow3d):
    """
    Only used for testing at the moment
    (warps currently backward)

    """
    with open(os.path.join(os.path.dirname(__file__), 'farneback_kernels.cu')) as f:
        read_data = f.read()
    f.closed

    mod = SourceModule(read_data)
    interpolation_kernel = mod.get_function('warpByFlowField')

    r1_texture = mod.get_texref('sourceTex')

    farneback3d._utils.ndarray_to_float_tex(r1_texture, vol)
    rtn_gpu = gpuarray.GPUArray(vol.shape, vol.dtype)
    flow_gpu = gpuarray.to_gpu(flow3d)

    block = (32, 32, 1)
    grid = (int(divup(flow3d.shape[3], block[0])),
            int(divup(flow3d.shape[2], block[1])), 1)

    interpolation_kernel(
        flow_gpu,
        rtn_gpu,
        np.int32(flow_gpu.shape[3]),
        np.int32(flow_gpu.shape[2]),
        np.int32(flow_gpu.shape[1]),
        np.float32(1),
        np.float32(1),
        np.float32(1),
        block=block, grid=grid)

    return rtn_gpu.get()
