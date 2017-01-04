from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from keras.datasets import mnist


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


def mnist_data(batch_size):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_train /= 255
    dim = X_train.shape
    numbatches = dim[0] // batch_size
    X_train = X_train.reshape((batch_size, dim[1]*dim[2], numbatches))
    print(X_train.shape)
    return X_train