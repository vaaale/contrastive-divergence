import pickle
from RBM import RBM


def pretrain_news20(x_train, model_path):
    params = {
        'type': 'sigmoid',
        'epsilonw': 0.1,
        'epsilonvb': 0.1,
        'epsilonhb': 0.1,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 20
    }
    model, batchdata = RBM(x_train, 1000, params)
    pickle.dump(model, open(model_path + '/layer1.pkl', 'wb'))

    params = {
        'type': 'sigmoid',
        'epsilonw': 0.1,
        'epsilonvb': 0.1,
        'epsilonhb': 0.1,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 40
    }
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
        'maxepoch': 100
    }
    model, batchdata = RBM(batchdata, 2, params)
    pickle.dump(model, open(model_path + '/layer4.pkl', 'wb'))


def pretrain_mnist(x_train, model_path):
    params = {
        'type': 'sigmoid',
        'epsilonw': 0.1,
        'epsilonvb': 0.1,
        'epsilonhb': 0.1,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 10
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
        'maxepoch': 10
    }
    model, batchdata = RBM(batchdata, 2, params)
    pickle.dump(model, open(model_path + '/layer4.pkl', 'wb'))
