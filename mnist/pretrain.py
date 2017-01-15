import pickle

#from RBM import RBM
import pycuda.autoinit

from RBMCuda import RBM


def pretrain_mnist(x_train, model_path):
    params = {
        'type': 'sigmoid',
        'epsilonw': 0.1,
        'epsilonvb': 0.1,
        'epsilonhb': 0.1,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 1
    }
    model, batchdata = RBM(x_train, 1000, params)
    pickle.dump(model, open(model_path + '/layer1.pkl', 'wb'))

    model, batchdata = RBM(batchdata, 500, params)
    pickle.dump(model, open(model_path + '/layer2.pkl', 'wb'))

    model, batchdata = RBM(batchdata, 250, params)
    pickle.dump(model, open(model_path + '/layer3.pkl', 'wb'))

    params = {
        'type': 'linear',
        'noise': 'gaussian',
        'epsilonw': 0.001,
        'epsilonvb': 0.001,
        'epsilonhb': 0.001,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 1
    }
    model, batchdata = RBM(batchdata, 2, params)
    pickle.dump(model, open(model_path + '/layer4.pkl', 'wb'))

    pycuda.autoinit.context.detach()
