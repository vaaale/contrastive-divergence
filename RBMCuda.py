import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
import time
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand

culinalg.init()
functions = SourceModule("""
#include <math.h>
  __global__ void sigmoid(float *dest, float *a)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dest[idx] =  1 / (1+exp(-1 * a[idx]));
  }

  __global__ void stochastic_gt(float *dest, float *a, float *b)
  {
    c[i] = a[i] > b b[i] ? 1.0 : 0.0;
  }

  """)
f_sigmoid = functions.get_function("sigmoid")
f_stoch_gt = functions.get_function("stochastic_gt")

def sigmoid(a):
    dim = a.shape
    dest = np.zeros_like(a)
    dest_gpu = gpuarray.to_gpu(dest)
    f_sigmoid(dest_gpu, drv.In(a), block=(1024, 1, 1), grid=(int(dim[0] * dim[1] / 1024) + 1, 1, 1))
    return dest_gpu


def RBM(batchdata, numhid, params):
    type = params['type']
    noise = True if 'noise' in params else False
    epsilonw = params['epsilonw']
    epsilonvb = params['epsilonvb']
    epsilonhb = params['epsilonhb']

    weightcost = params['weightcost']

    initialmomentum = params['initialmomentum']
    finalmomentum = params['finalmomentum']
    maxepoch = params['maxepoch']

    numbatches = len(batchdata)
    numcases, numdims = batchdata[0].shape

    print('Initializing RBM: {}'.format(numhid))
    # Initializing symmetric weights and biases.
    vishid = 0.1 * np.random.randn(numdims, numhid)
    hidbiases = np.zeros((1, numhid))
    visbiases = np.zeros((1, numdims))

    vishid_gpu = gpuarray.to_gpu(vishid)
    hidbiases_gpu = gpuarray.to_gpu(hidbiases)
    visbiases_gpu = gpuarray.to_gpu(visbiases)

    # vishidinc = np.zeros((numdims, numhid))
    # hidbiasinc = np.zeros((1, numhid))
    # visbiasinc = np.zeros((1, numdims))
    batchposhidprobs = []

    batchdata_gpu = gpuarray.to_gpu(batchdata)

    for epoch in range(maxepoch):
        print('Epoch {}'.format(epoch))
        errsum = 0
        for batch in range(numbatches):
            # Positive phase
            data = batchdata_gpu[batch]
            if type == 'sigmoid':
                poshidprobs_gpu = sigmoid(cumisc.subtract(culinalg.dot(data, vishid_gpu), hidbiases_gpu))
                #poshidprobs = 1 / (1 + np.exp(-np.dot(data, vishid) - hidbiases))
            else:
                poshidprobs_gpu = cumisc.add(culinalg.dot(data, vishid_gpu), hidbiases)

            # if epoch == maxepoch - 1:
            #     batchposhidprobs.append(poshidprobs)
            posprods_gpu = culinalg.dot(culinalg.transpose(data), poshidprobs_gpu)
            poshidact_gpu = cumisc.sum(poshidprobs_gpu, axis=0)
            posvisact_gpu = cumisc.sum(data, axis=0)

            # end of positive phase

            if noise:
                poshidstates = poshidprobs + np.random.normal(size=poshidprobs.shape)
            else:
                poshidstates = np.asarray(poshidprobs > np.random.rand(numcases, numhid), dtype='float32')

            # negative phase

            negdata = 1 / (1 + np.exp(-np.dot(poshidstates, vishid.T) - visbiases))
            if type == 'sigmoid':
                neghidprobs = 1 / (1 + np.exp(-np.dot(negdata, vishid) - hidbiases))
            else:
                neghidprobs = np.dot(negdata, vishid) + hidbiases

            negprods = np.dot(np.transpose(negdata), neghidprobs)
            neghidact = np.sum(neghidprobs, axis=0)
            negvisact = np.sum(negdata, axis=0)
            # end of negative phase
            diff = data - negdata
            err = np.sum(np.square(diff))
            errsum = err + errsum

            if epoch > 5:
                momentum = finalmomentum
            else:
                momentum = initialmomentum

            # update of weights and biases
            vishidinc = momentum * vishidinc + epsilonw * ((posprods - negprods) / numcases - weightcost * vishid)
            visbiasinc = momentum * visbiasinc + (epsilonvb / numcases) * (posvisact - negvisact)
            hidbiasinc = momentum * hidbiasinc + (epsilonhb / numcases) * (poshidact - neghidact)
            vishid += vishidinc
            visbiases += visbiasinc
            hidbiases += hidbiasinc

            # if batch % 100 == 0:
            #     display(data.reshape(100, 28,28), negdata.reshape(100, 28,28))


            # end of updates
        print('Epoch: {}, error: {}'.format(epoch, errsum))

    print('Model shapes:')
    print(vishid.shape)
    print(visbiases.shape)
    print(hidbiases.shape)
    model = {
        'vishid': vishid,
        'visbiases': visbiases,
        'hidbiases': hidbiases
    }

    return model, batchposhidprobs
