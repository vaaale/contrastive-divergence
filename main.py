from dataset import dataset
from pretrain import pretrain
from finetune import finetune

batch_size = 100

x_train = dataset(batch_size)

#pretrain(x_train)

finetune(x_train)



