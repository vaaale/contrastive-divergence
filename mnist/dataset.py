import numpy as np
from keras.datasets import mnist


def mnist_batches(batch_size):
    num_dim = 784
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_cases = len(X_train)
    num_batches = num_cases / batch_size
    X_train = X_train.reshape(num_cases, num_dim)

    X_train = X_train.astype('float32')
    X_train /= 255

    batchdata = [X_train[b * batch_size:b * batch_size + batch_size] for b in np.arange(num_batches)]

    # display(batchdata[0].reshape(100, 28,28), batchdata[2].reshape(100, 28,28))

    return batchdata


def mnist_data():
    num_dim = 784
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_cases = len(X_train)
    X_train = X_train.reshape(num_cases, num_dim)

    X_train = X_train.astype('float32')
    X_train /= 255

    # display(batchdata[0].reshape(100, 28,28), batchdata[2].reshape(100, 28,28))

    return X_train, y_train


if __name__ == '__main__':
    data, labels = mnist_data()

    print(len(data))
    print(data[0].shape)
    print(len(labels))
    print(labels[0].shape)
