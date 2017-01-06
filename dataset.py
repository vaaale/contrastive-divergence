from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from keras.datasets import mnist

from display import display


def news20(batch_size):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    print(len(newsgroups_train.data))

    vectorizer = CountVectorizer(stop_words=stopwords.words('english'), max_features=2000)
    x_train = vectorizer.fit_transform(newsgroups_train.data)

    x_train = x_train.toarray()
    x_train = x_train[:(x_train.shape[0] // batch_size) * batch_size]
    x_train = np.reshape(x_train, (batch_size, x_train.shape[1], x_train.shape[0] // batch_size))
    x_train = x_train.astype('float32')
    x_train /= 255

    print(x_train.shape)
    return x_train


def mnist_batches(batch_size):
    num_dim = 784
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_cases = len(X_train)
    num_batches = num_cases / batch_size
    X_train = X_train.reshape(num_cases, num_dim)

    X_train = X_train.astype('float32')
    X_train /= 255

    batchdata = [X_train[b*batch_size:b*batch_size+batch_size] for b in np.arange(num_batches)]

    #display(batchdata[0].reshape(100, 28,28), batchdata[2].reshape(100, 28,28))

    return batchdata


def mnist_data(batch_size):
    num_dim = 784
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_cases = len(X_train)
    num_batches = num_cases / batch_size
    X_train = X_train.reshape(num_cases, num_dim)

    X_train = X_train.astype('float32')
    X_train /= 255

    #display(batchdata[0].reshape(100, 28,28), batchdata[2].reshape(100, 28,28))

    return X_train