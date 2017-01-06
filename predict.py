from dataset import mnist_data
from display import display
from finetune import build_model
import matplotlib.pyplot as plt

data, labels = mnist_data()
numdim = data[0].shape[0]

autoencoder, encoder = build_model(numdim)

autoencoder.load_weights('models/final-model.hdf5')

n = 5000
batch = data[0:n]
y = labels[0:n]

encoded = encoder.predict(batch)
reconstructions = autoencoder.predict(batch)

display(batch[0:10].reshape(10, 28, 28), reconstructions[0:10].reshape(10, 28, 28))


plt.figure()
plt.scatter(encoded[:,0], encoded[:,1], c=y)
plt.show()