import pickle
import h5py
import numpy as np
from PIL import Image
from util import tile_raster_images

from TfRBM import RBM


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


def pretrain_mnist(x_train, model_path):
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

    def mnist_callback(epoch, n_w):
        image = Image.fromarray(
            tile_raster_images(
                X=n_w.T,
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
        )
        image.save("mnist/output/rbm_{}.png".format(epoch))


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
    numbatches = len(x_train)
    model = RBM(batch_gen(x_train), numbatches, 784, 1000, params, create_outputter('mnist/data/rbm1_output.hdf5', 100, 1000), callbacks=[mnist_callback])
    pickle.dump(model, open(model_path + '/layer1.pkl', 'wb'))

    model = RBM(hdf5_generator('mnist/data/rbm1_output.hdf5'), numbatches, 1000, 500, params, create_outputter('mnist/data/rbm2_output.hdf5', 100, 500))
    pickle.dump(model, open(model_path + '/layer2.pkl', 'wb'))

    model = RBM(hdf5_generator('mnist/data/rbm2_output.hdf5'), numbatches, 500, 250, params, create_outputter('mnist/data/rbm3_output.hdf5', 100, 250))
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
    model = RBM(hdf5_generator('mnist/data/rbm3_output.hdf5'), numbatches, 250, 2, params, create_outputter('mnist/data/rbm4_output.hdf5', 100, 2))
    pickle.dump(model, open(model_path + '/layer4.pkl', 'wb'))


