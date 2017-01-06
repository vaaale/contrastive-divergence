import numpy as np

from display import display


def RBM(batchdata, numhid, params):
    type = params['type']
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
    vishid = 0.1*np.random.randn(numdims, numhid)
    hidbiases = np.zeros((1, numhid))
    visbiases = np.zeros((1, numdims))

    vishidinc = np.zeros((numdims, numhid))
    hidbiasinc = np.zeros((1, numhid))
    visbiasinc = np.zeros((1, numdims))
    batchposhidprobs = np.zeros((numcases, numhid, numbatches))

    for epoch in range(maxepoch):
        print('Epoch {}'.format(epoch))
        errsum = 0
        for batch in range(numbatches):
            # Positive phase
            #data = batchdata[:, :, batch]
            data = batchdata[batch]
            if type == 'sigmoid':
                poshidprobs = 1 / (1 + np.exp(-np.dot(data, vishid) - hidbiases))
            else:
                poshidprobs = np.dot(data, vishid) + hidbiases

            if epoch == maxepoch-1:
                batchposhidprobs[:, :, batch] = poshidprobs
            posprods = np.dot(np.transpose(data), poshidprobs)
            poshidact = np.sum(poshidprobs, axis=0)
            posvisact = np.sum(data, axis=0)

            # end of positive phase

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

    model = {
        'vishid': vishid,
        'visbiases': visbiases,
        'hidbiases': hidbiases
    }

    return model, batchposhidprobs
