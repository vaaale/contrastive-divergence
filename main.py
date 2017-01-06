from dataset import news20, mnist_data, mnist_batches
from pretrain import pretrain
from finetune import finetune

batch_size = 100

x_train = mnist_batches(batch_size)

pretrain(x_train)

x_train = mnist_data(batch_size)
finetune(x_train)



