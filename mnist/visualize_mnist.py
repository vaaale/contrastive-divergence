import matplotlib
import matplotlib.pyplot as plt

from mnist.dataset import mnist_data
from mnist.display import display
from mnist.finetune import build_mnist_model


def visualize(data, labels):
    numdim = data[0].shape[0]

    autoencoder, encoder = build_mnist_model(numdim)

    autoencoder.load_weights('mnist/models/final-model.hdf5')

    n = 10000
    batch = data[0:n]
    y = labels[0:n]

    encoded = encoder.predict(batch)
    reconstructions = autoencoder.predict(batch)

    display(batch[0:10].reshape(10, 28, 28), reconstructions[0:10].reshape(10, 28, 28))

    # print(matplotlib.backend)
    plt.figure()
    plt.jet()
    plt.scatter(encoded[:,0], encoded[:,1], c=y, s=1)
    plt.show()


if __name__ == '__main__':
    data, labels = mnist_data()
    visualize(data, labels)