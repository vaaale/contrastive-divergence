from mnist.dataset import mnist_batches, mnist_data
from mnist.finetune import finetune_mnist
from mnist.pretrain import pretrain_mnist


def train_mnist():
    batch_size = 100

    x_train = mnist_batches(batch_size)
    pretrain_mnist(x_train, 'mnist/models')

    x_train, _ = mnist_data()
    finetune_mnist(x_train, 'mnist/models')


if __name__ == '__main__':
    train_mnist()
