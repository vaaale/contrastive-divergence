#!/usr/bin/env python

"""
Demonstrates multiplication of two matrices on the GPU.
"""
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
import time
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import skcuda

culinalg.init()

# Double precision is only supported by devices with compute
# capability >= 1.3:

print('Testing matrix multiplication for type ' + str(np.dtype(np.float32)))
a = np.asarray(np.random.rand(10, 5), np.float32)
b = np.asarray(np.random.rand(5, 2), np.float32)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)

c_gpu = culinalg.dot(a_gpu, b_gpu)
print('Success status: ', np.allclose(np.dot(a, b), c_gpu.get()))
a_gpu.gpudata.free()
b_gpu.gpudata.free()
c_gpu.gpudata.free()



msize = 20000
a = np.random.randn(msize, msize)
a = a.astype(np.float32)
dest = np.zeros_like(a)
print(a.nbytes)

mod = SourceModule("""
#include <math.h>
  __global__ void myfunc(float *dest, float *a)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dest[idx] =  1 / (1+exp(-1 * a[idx]));
  }
  """)
sigmoid = mod.get_function("myfunc")
time_start = time.time()
#sigmoid(drv.Out(dest), drv.In(a), block=(1024, 1, 1), grid=(int(msize * msize / 1024) + 1, 1, 1))
dest_gpu = gpuarray.to_gpu(dest)
sigmoid(dest_gpu, drv.In(a), block=(1024, 1, 1), grid=(int(msize * msize / 1024) + 1, 1, 1))

dest = dest_gpu.get()

print(time.time() - time_start)
print(a[0][0])

time_start = time.time()
b = 1 / (1 + np.exp(-1 * a))
print(time.time() - time_start)

print('Success status: ', np.allclose(b, dest))
