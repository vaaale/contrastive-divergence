import pickle

# from RBM import RBM
from TfRBM import RBM
import numpy as np
import h5py


def create_outputter(filename, batch_size, num_features):
    f = h5py.File(filename, 'w')
    dset = f.create_dataset('output',
                            shape=(0, batch_size, num_features),
                            chunks=(1, batch_size, num_features),
                            maxshape=(None, batch_size, num_features),
                            dtype='f')

    def write(batch):
        dset.resize(len(dset) + 1, axis=0)
        dset[len(dset) - 1, :] = batch

    return write


def pretrain_news20(x_train, model_path):
    def batch_gen(X):
        indices = list(range(len(X)))
        while True:
            np.random.shuffle(indices)
            for i in indices:
                yield X[i]

    def hdf5_generator(filename):
        with h5py.File(filename, 'r') as f:
            dset = f['output']
            num_rows = len(dset)
            indices = list(range(num_rows))
            print('Number of rows {}'.format(num_rows))
            while True:
                np.random.shuffle(indices)
                for i in indices:
                    data = dset[i]
                    yield data

    params = {
        'activation': 'sigmoid',
        'epsilonw': 0.1,
        'epsilonvb': 0.1,
        'epsilonhb': 0.1,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 50
    }
    numbatches = len(x_train)
    batch_size = 100
    numdims = 2000
    numhid = 500
    model = RBM(batch_gen(x_train), numbatches, numdims, numhid, params, create_outputter('mnist/data/rbm1_output.hdf5', batch_size, numhid))
    pickle.dump(model, open(model_path + '/layer1.pkl', 'wb'))

    numdims = 500
    numhid = 250
    model = RBM(hdf5_generator('mnist/data/rbm1_output.hdf5'), numbatches, numdims, numhid, params, create_outputter('mnist/data/rbm2_output.hdf5', batch_size, numhid))
    pickle.dump(model, open(model_path + '/layer2.pkl', 'wb'))

    params = {
        'activation': 'sigmoid',
        'epsilonw': 0.01,
        'epsilonvb': 0.01,
        'epsilonhb': 0.01,
        'weightcost': 0.0002,
        'initialmomentum': 0.5,
        'finalmomentum': 0.9,
        'maxepoch': 100
    }
    numdims = 250
    numhid = 125
    model = RBM(hdf5_generator('mnist/data/rbm2_output.hdf5'), numbatches, numdims, numhid, params, create_outputter('mnist/data/rbm3_output.hdf5', batch_size, numhid))
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
        'maxepoch': 100
    }
    numdims = 125
    numhid = 2
    model = RBM(hdf5_generator('mnist/data/rbm3_output.hdf5'), numbatches, numdims, numhid, params, create_outputter('mnist/data/rbm4_output.hdf5', batch_size, numhid))
    pickle.dump(model, open(model_path + '/layer4.pkl', 'wb'))
