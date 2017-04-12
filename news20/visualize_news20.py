import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from news20.dataset import news20_data
from news20.finetune import build_news20_model


def visualize(data, labels):
    numdim = data[0].shape[0]

    autoencoder, encoder = build_news20_model(numdim)

    autoencoder.load_weights('news20/models/final-model.hdf5')

    projections = encoder.predict(data)

    # tsne = TSNE(n_components=2, perplexity=20.0, learning_rate=1000, early_exaggeration=2)
    # projections = tsne.fit_transform(projections)

    plt.figure()
    plt.scatter(projections[:, 0], projections[:, 1], c=labels, s=1)
    plt.show()


if __name__ == '__main__':
    data, labels = news20_data()
    visualize(data, labels)
