import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dataset import news20_data
from finetune import build_news20_model
import numpy as np


data, labels = news20_data()
numdim = data[0].shape[0]

autoencoder, encoder = build_news20_model(numdim)

autoencoder.load_weights('models/news20/final-model.hdf5')

projections = encoder.predict(data)

plt.figure()
plt.scatter(projections[:, 0], projections[:, 1], c=labels)
plt.show()
