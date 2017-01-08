from news20.dataset import news20_minibatches, news20_data
from news20.finetune import finetune_news20
from news20.pretrain import pretrain_news20
from news20.visualize_news20 import visualize


def train_news20():
    batch_size = 100

    x_train = news20_minibatches(batch_size)
    pretrain_news20(x_train, 'news20/models')

    x_train, labels = news20_data()
    finetune_news20(x_train, 'news20/models')

    visualize(x_train, labels)



if __name__ == '__main__':
    # train_mnist()
    train_news20()
