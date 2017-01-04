import numpy as np


def RBM(batchdata, numhid, params):
    type = params['type']
    epsilonw = params['epsilonw']
    epsilonvb = params['epsilonvb']
    epsilonhb = params['epsilonhb']

    weightcost = params['weightcost']

    initialmomentum = params['initialmomentum']
    finalmomentum = params['finalmomentum']
    maxepoch = params['maxepoch']

    numcases, numdims, numbatches = batchdata.shape

    restart = 1
    # Initializing symmetric weights and biases
    if restart == 1:
        print('Initializing RBM: {}'.format(numhid))
        restart = 0

        # Initializing symmetric weights and biases.
        vishid = 0.1 * np.random.randn(numdims, numhid)
        hidbiases = np.zeros((1, numhid))
        visbiases = np.zeros((1, numdims))

        poshidprobs = np.zeros((numcases, numhid))
        neghidprobs = np.zeros((numcases, numhid))
        posprods = np.zeros((numdims, numhid))
        negprods = np.zeros((numdims, numhid))
        vishidinc = np.zeros((numdims, numhid))
        hidbiasinc = np.zeros((1, numhid))
        visbiasinc = np.zeros((1, numdims))
        batchposhidprobs = np.zeros((numcases, numhid, numbatches))

    for epoch in range(maxepoch):
        print('Epoch {}'.format(epoch))
        errsum = 0
        for batch in range(numbatches):
            # Positive phase
            data = batchdata[:, :, batch]
            if type == 'sigmoid':
                poshidprobs = 1 / (1 + np.exp(-np.dot(data, vishid) - np.matlib.repmat(hidbiases, numcases, 1)))
            else:
                poshidprobs = np.dot(data, vishid) + np.matlib.repmat(hidbiases, numcases, 1)

            if epoch == maxepoch-1:
                batchposhidprobs[:, :, batch] = poshidprobs
            posprods = np.dot(data.T, poshidprobs)
            poshidact = np.sum(poshidprobs)
            posvisact = np.sum(data)
            # end of positive phase
            poshidstates = poshidprobs > np.random.rand(numcases, numhid)
            # negative phase
            negdata = 1 / (1 + np.exp(-np.dot(poshidstates, vishid.T) - np.matlib.repmat(visbiases, numcases, 1)))
            if type == 'sigmoid':
                neghidprobs = 1 / (1 + np.exp(-np.dot(negdata, vishid) - np.matlib.repmat(hidbiases, numcases, 1)))
            else:
                neghidprobs = np.dot(negdata, vishid) + np.matlib.repmat(hidbiases, numcases, 1)

            negprods = np.dot(negdata.T, neghidprobs)
            neghidact = np.sum(neghidprobs)
            negvisact = np.sum(negdata)
            # end of negative phase
            err = np.sum((data - negdata) ** 2)
            errsum = err + errsum

            if epoch > 5:
                momentum = finalmomentum
            else:
                momentum = initialmomentum

            # update of weights and biases
            vishidinc = momentum * vishidinc + epsilonw * ((posprods - negprods) / numcases - weightcost * vishid)
            visbiasinc = momentum * visbiasinc + (epsilonvb / numcases) * (posvisact - negvisact)
            hidbiasinc = momentum * hidbiasinc + (epsilonhb / numcases) * (poshidact - neghidact)
            vishid = vishid + vishidinc
            visbiases = visbiases + visbiasinc
            hidbiases = hidbiases + hidbiasinc

            # end of updates
        print('Epoch: {}, error: {}'.format(epoch, errsum))

    model = {
        'vishid': vishid,
        'visbiases': visbiases,
        'hidbiases': hidbiases
    }

    return model, batchposhidprobs
