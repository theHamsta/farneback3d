import os
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule


def divup(a, b):
    if a % b:
        return a / b + 1
    else:
        return a / b


def ndarray_to_float_tex(tex_ref, ndarray, address_mode=cuda.address_mode.BORDER, filter_mode=cuda.filter_mode.LINEAR):
    if isinstance(ndarray, np.ndarray):
        cu_array = cuda.np_to_array(ndarray, 'C')
    elif isinstance(ndarray, gpuarray.GPUArray):
        cu_array = cuda.gpuarray_to_array(ndarray, 'C')
    else:
        raise TypeError(
            'ndarray must be numpy.ndarray or pycuda.gpuarray.GPUArray')

    cuda.TextureReference.set_array(tex_ref, cu_array)

    cuda.TextureReference.set_address_mode(
        tex_ref, 0, address_mode)
    if ndarray.ndim >= 2:
        cuda.TextureReference.set_address_mode(
            tex_ref, 1, address_mode)
    if ndarray.ndim >= 3:
        cuda.TextureReference.set_address_mode(
            tex_ref, 2,  address_mode)
    cuda.TextureReference.set_filter_mode(
        tex_ref, filter_mode)
    tex_ref.set_flags(tex_ref.get_flags(
    ) & ~cuda.TRSF_NORMALIZED_COORDINATES & ~cuda.TRSF_READ_AS_INTEGER)


with open(os.path.join(os.path.dirname(__file__), 'resize.cu')) as f:
    _read_data = f.read()

_mod = SourceModule(_read_data)
_tex_ref = _mod.get_texref('sourceTex')
_kernel = _mod.get_function('resize')


def resize_gpu(src_vol, dst_vol=None, dst_shape=None, scaling=None):

    if dst_shape is None:
        assert scaling
        dst_shape = [np.int(np.round(i * scaling)) for i in src_vol.shape]

    if dst_vol is None:
        dst_vol = gpuarray.GPUArray(dst_shape, np.float32)

    ndarray_to_float_tex(_tex_ref, src_vol)

    block = (32, 32, 1)
    grid = (int(divup(dst_vol.shape[2], block[0])),
            int(divup(dst_vol.shape[1], block[1])), 1)

    _kernel(dst_vol,
            np.int32(src_vol.shape[2]),
            np.int32(src_vol.shape[1]),
            np.int32(src_vol.shape[0]),
            np.int32(dst_vol.shape[2]),
            np.int32(dst_vol.shape[1]),
            np.int32(dst_vol.shape[0]),
            grid=grid, block=block)

    return dst_vol


# from https://www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-methods
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (...) %2.2f sec' %
              (method.__name__, te - ts))
        return result

    return timed


class DictWrapper:
    def __init__(self, _dict):
        self.dict = _dict

    def __getattr__(self, entry):
        rtn = self.dict[entry]
        if isinstance(rtn, dict):
            return DictWrapper(rtn)
        else:
            return rtn

    # def __setattr__(self, entry, val):
    #     self.dict[entry] = val

    def __setitem__(self, entry, val):
        self.dict[entry] = val

    def __getitem__(self, entry):

        return self.__getattr__(entry)
