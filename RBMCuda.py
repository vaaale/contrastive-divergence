import numpy as np
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culinalg
import time
from pycuda.elementwise import ElementwiseKernel
import skcuda.misc as cumisc

from mnist.display import display

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


def stochastic_gt(a_gpu, numcases, numhid):
    b_gpu = gpuarray.to_gpu(np.random.rand(numcases, numhid))
    dest_gpu = gpuarray.empty_like(a_gpu)
    f_stoch_gt(dest_gpu, a_gpu, b_gpu)
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
    vishid = 0.1 * np.random.randn(numdims, numhid).astype(np.float32)
    hidbiases = np.zeros((1, numhid)).astype(np.float32)
    visbiases = np.zeros((1, numdims)).astype(np.float32)

    vishidinc = np.zeros((numdims, numhid))
    hidbiasinc = np.zeros((1, numhid))
    visbiasinc = np.zeros((1, numdims))

    vishidinc_gpu = gpuarray.to_gpu(vishidinc.astype(np.float32))
    hidbiasinc_gpu = gpuarray.to_gpu(hidbiasinc.astype(np.float32))
    visbiasinc_gpu = gpuarray.to_gpu(visbiasinc.astype(np.float32))

    vishid_gpu = gpuarray.to_gpu(vishid)
    hidbiases_gpu = gpuarray.to_gpu(hidbiases)
    visbiases_gpu = gpuarray.to_gpu(visbiases)

    batchposhidprobs = []

    batchdata_gpu = gpuarray.to_gpu(np.asarray(batchdata))

    for epoch in range(maxepoch):
        print('Epoch {}'.format(epoch))
        errsum = 0
        epoch_time = 0
        for batch in range(numbatches):
            b_start = time.time()
            # Positive phase
            data_gpu = batchdata_gpu[batch]
            if type == 'sigmoid':
                poshidprobs_gpu = sigmoid(cumisc.subtract(culinalg.dot(data_gpu, vishid_gpu), hidbiases_gpu))
            else:
                poshidprobs_gpu = cumisc.add(culinalg.dot(data_gpu, vishid_gpu), hidbiases_gpu)



            if epoch == maxepoch - 1:
                poshidprobs = poshidprobs_gpu.get()
                batchposhidprobs.append(poshidprobs)

            posprods_gpu = culinalg.dot(culinalg.transpose(data_gpu), poshidprobs_gpu)
            poshidact_gpu = cumisc.sum(poshidprobs_gpu, axis=0)
            posvisact_gpu = cumisc.sum(data_gpu, axis=0)


            # end of positive phase

            if noise:
                #poshidstates = poshidprobs + np.random.normal(size=poshidprobs.shape)
                poshidstates_gpu = cumisc.add(poshidprobs_gpu, gpuarray.to_gpu(np.random.normal(size=poshidprobs_gpu.shape).astype(np.float32)))
            else:
                poshidstates = np.asarray(poshidprobs_gpu.get() > np.random.rand(numcases, numhid), dtype='float32')
                #poshidstates_gpu = stochastic_gt(poshidprobs_gpu, numcases, numhid)
                poshidstates_gpu = gpuarray.to_gpu(poshidstates)


            # negative phase
            # negdata = 1 / (1 + np.exp(-np.dot(poshidstates, vishid.T) - visbiases))
            negdata_gpu = sigmoid(cumisc.subtract(culinalg.dot(poshidstates_gpu, culinalg.transpose(vishid_gpu)), visbiases_gpu))
            if type == 'sigmoid':
                #neghidprobs = 1 / (1 + np.exp(-np.dot(negdata, vishid) - hidbiases))
                neghidprobs_gpu = sigmoid(cumisc.subtract(culinalg.dot(negdata_gpu, vishid_gpu), hidbiases_gpu))
            else:
                #neghidprobs = np.dot(negdata, vishid) + hidbiases
                neghidprobs_gpu = cumisc.add(culinalg.dot(negdata_gpu, vishid_gpu), hidbiases_gpu)

            negprods_gpu = culinalg.dot(culinalg.transpose(negdata_gpu), neghidprobs_gpu)
            neghidact_gpu = cumisc.sum(neghidprobs_gpu, axis=0)
            negvisact_gpu = cumisc.sum(negdata_gpu, axis=0)

            # end of negative phase
            diff = data_gpu.get() - negdata_gpu.get()
            err = np.sum(np.square(diff))
            errsum = err + errsum

            if epoch > 5:
                momentum = finalmomentum
            else:
                momentum = initialmomentum


            # update of weights and biases

            # vishidinc = momentum * vishidinc + (epsilonw / numcases) * ((posprods - negprods) - weightcost * vishid)

            vishid_momentum_gpu = scale(vishidinc_gpu, momentum)
            weightcost_gpu = scale(vishid_gpu, weightcost)
            statistics_gpu = scale(cumisc.subtract(posprods_gpu, negprods_gpu), (epsilonw/numcases))
            vishidinc_gpu = cumisc.add(vishid_momentum_gpu, statistics_gpu)
            vishidinc_gpu = cumisc.subtract(vishidinc_gpu, weightcost_gpu)
            # np.allclose(vishidinc, vishidinc_gpu.get())

            # visbiasinc = momentum * visbiasinc + (epsilonvb / numcases) * (posvisact - negvisact)
            visbiasinc = momentum * visbiasinc_gpu.get() + (epsilonvb / numcases) * (posvisact_gpu.get() - negvisact_gpu.get())
            visbiasinc_gpu = gpuarray.to_gpu(visbiasinc)

            # visbias_momentum_gpu = scale(visbiasinc_gpu, momentum)
            # visbias_statistics_gpu = cumisc.subtract(posvisact_gpu, negvisact_gpu)
            # visbiasinc_gpu = cumisc.add(visbias_momentum_gpu, scale(visbias_statistics_gpu, (epsilonvb / numcases)))
            # np.allclose(visbiasinc, visbiasinc_gpu.get())

            # hidbiasinc = momentum * hidbiasinc + (epsilonhb / numcases) * (poshidact - neghidact)
            hidbiasinc = momentum * hidbiasinc_gpu.get() + (epsilonhb / numcases) * (poshidact_gpu.get() - neghidact_gpu.get())
            vishidinc_gpu = gpuarray.to_gpu(hidbiasinc)
            # hidbias_momentum_gpu = scale(hidbiasinc_gpu, momentum)
            # hidbias_statistics_gpu = cumisc.subtract(poshidact_gpu, neghidact_gpu)
            # hidbiasinc_gpu = cumisc.add(hidbias_momentum_gpu, scale(hidbias_statistics_gpu, (epsilonhb / numcases)))
            # np.allclose(hidbiasinc, hidbiasinc_gpu.get())


            vishid_gpu = cumisc.add(vishid_gpu, vishidinc_gpu)
            visbiases_gpu = cumisc.add(visbiases_gpu, visbiasinc_gpu)
            hidbiases_gpu = cumisc.add(hidbiases_gpu, hidbiasinc_gpu)

            # if batch % 100 == 0:
            #     negdata = negdata_gpu.get()
            #     display(data_gpu.get().reshape(100, 28,28), negdata.reshape(100, 28,28))

            b_end = time.time()
            epoch_time += (b_end - b_start)
            # print("Time: {}".format(epoch_time))

            # end of updates
        print('Epoch:({} seconds) {}, error: {}'.format(epoch_time, epoch, errsum))


    vishid = vishid_gpu.get()
    visbiases = visbiases_gpu.get()
    hidbiases = hidbiases_gpu.get()
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
