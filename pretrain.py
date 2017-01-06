import pickle
from RBM import RBM

batch_size = 100

def pretrain(x_train):
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
    pickle.dump(model, open('models/layer1.pkl', 'wb'))

    model, batchdata = RBM(batchdata, 250, params)
    pickle.dump(model, open('models/layer2.pkl', 'wb'))

    params = {
        'type': 'linear',
        'epsilonw': 0.001,
        'epsilonvb': 0.001,
        'epsilonhb': 0.001,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 10
    }
    model, batchdata = RBM(batchdata, 2, params)
    pickle.dump(model, open('models/layer3.pkl', 'wb'))


