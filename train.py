from dataset import news20_minibatches, mnist_data, mnist_batches, news20_data
from finetune import finetune_mnist, finetune_news20
from pretrain import pretrain_mnist, pretrain_news20


def train_mnist():
    batch_size = 100

    x_train = mnist_batches(batch_size)
    pretrain_mnist(x_train, 'models/mnist')

    x_train, _ = mnist_data()
    finetune_mnist(x_train, 'models/mnist')


def train_news20():
    batch_size = 100

    x_train = news20_minibatches(batch_size)
    pretrain_news20(x_train, 'models/news20')

    x_train, _ = news20_data()
    finetune_news20(x_train, 'models/news20')


if __name__ == '__main__':
    #train_mnist()
    train_news20()


