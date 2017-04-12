from news20.dataset import news20_minibatches, news20_data
from news20.finetune import finetune_news20
from news20.pretrain import pretrain_news20
from news20.visualize_news20 import visualize
import time


def train_news20():
    batch_size = 100
    x_train = news20_minibatches(batch_size)
    pt_start = time.time()
    pretrain_news20(x_train, 'news20/models')
    pt_end = time.time()

    x_train, labels = news20_data()
    ft_start = time.time()
    finetune_news20(x_train, 'news20/models')
    ft_end = time.time()

    print('Pretraining: {0}'.format(pt_end-pt_start))
    print('Finetuning: {0}'.format(ft_end-ft_start))

    visualize(x_train, labels)



if __name__ == '__main__':
    train_news20()
