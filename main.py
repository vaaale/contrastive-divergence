from dataset import news20, mnist_data
from pretrain import pretrain
from finetune import finetune

batch_size = 100

x_train = news20(batch_size)

pretrain(x_train)

finetune(x_train)



