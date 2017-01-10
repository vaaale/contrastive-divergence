import pycuda.gpuarray as gpuarray
import pycuda.autoinit

import skcuda.linalg as culinalg
from pycuda.curandom import rand as curand

from pycuda.elementwise import ElementwiseKernel
culinalg.init()

# define elementwise `add()` function
add = ElementwiseKernel(
    "float *a, float *b, float *c",
    "c[i] = a[i] + b[i]",
    "add")

# create a couple of random matrices with a given shape

shape = 128, 1024
a_gpu = curand(shape)
b_gpu = curand(shape)

# compute sum on a gpu
c_gpu = gpuarray.empty_like(a_gpu)
add(a_gpu, b_gpu, c_gpu)

# check the result
import numpy.linalg as la

print(c_gpu - (a_gpu + b_gpu))
assert la.norm((c_gpu - (a_gpu + b_gpu)).get()) < 1e-5
