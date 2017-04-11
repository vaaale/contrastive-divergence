import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
import skcuda.linalg as culinalg
from pycuda.curandom import rand as curand

from pycuda.elementwise import ElementwiseKernel

culinalg.init()

f_sigmoid = ElementwiseKernel(
    "float *dest, float *a",
    "dest[i] =  1.0 / (1.0 + exp(-1.0 * a[i]))",
    "f_sigmoid")

f_stoch_gt = ElementwiseKernel(
    "float *dest, float *a, float *b",
    "dest[i] = (a[i] > b[i] ? 1.0 : 0.0)",
    "f_sigmoid")

f_scale = ElementwiseKernel(
    "float *dest, float *a, float b",
    "dest[i] = a[i] * b",
    "f_scale")


def sigmoid(a_gpu):
    dest_gpu = gpuarray.empty_like(a_gpu)
    f_sigmoid(dest_gpu, a_gpu)
    return dest_gpu


def scale(a_gpu, alpha):
    dest_gpu = gpuarray.empty_like(a_gpu)
    f_scale(dest_gpu, a_gpu, alpha)
    return dest_gpu


def stochastic_gt(a_gpu, numcases, numhid, rnd_nums):
    b_gpu = gpuarray.to_gpu(rnd_nums)
    dest_gpu = gpuarray.empty_like(a_gpu)
    f_stoch_gt(dest_gpu, a_gpu, b_gpu)
    return dest_gpu


if __name__ == '__main__':
    arr = np.asarray(np.random.rand(2, 2), dtype=np.float32)
    #arr = np.random.rand(2, 2)

    arr_gpu = gpuarray.to_gpu(arr)
    scaled_gpu = scale(arr_gpu, 0.1)
    print("Original")
    print(arr)

    print("Scaled")
    print(scaled_gpu.get())
