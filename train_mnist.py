from mnist.dataset import mnist_batches, mnist_data
from mnist.finetune import finetune_mnist
from mnist.pretrain import pretrain_mnist
import time


def train_mnist():
    batch_size = 100

    x_train = mnist_batches(batch_size)
    pt_start = time.time()
    pretrain_mnist(x_train, 'mnist/models')
    pt_end = time.time()

    # x_train, _ = mnist_data()
    # ft_start = time.time()
    # finetune_mnist(x_train, 'mnist/models')
    # ft_end = time.time()

    print('Pretraining: {0}'.format(pt_end-pt_start))
    # print('Finetuning: {0}'.format(ft_end-ft_start))

if __name__ == '__main__':
    train_mnist()
