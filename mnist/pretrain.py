import pickle

from TfRBM import RBM


def pretrain_mnist(x_train, model_path):

    params = {
        'activation': 'sigmoid',
        'epsilonw': 0.1,
        'epsilonvb': 0.1,
        'epsilonhb': 0.1,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 20
    }
    model, batchdata = RBM(x_train, len(x_train), 100, 784, 1000, params, None)
    pickle.dump(model, open(model_path + '/layer1.pkl', 'wb'))

    model, batchdata = RBM(batchdata, 500, params)
    pickle.dump(model, open(model_path + '/layer2.pkl', 'wb'))

    model, batchdata = RBM(batchdata, 250, params)
    pickle.dump(model, open(model_path + '/layer3.pkl', 'wb'))

    params = {
        'activation': 'linear',
        'noise': 'gaussian',
        'epsilonw': 0.001,
        'epsilonvb': 0.001,
        'epsilonhb': 0.001,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 50
    }
    model, batchdata = RBM(batchdata, 2, params)
    pickle.dump(model, open(model_path + '/layer4.pkl', 'wb'))


