from keras.preprocessing.text import text_to_word_sequence
from nltk import stem
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from keras.datasets import mnist, reuters
import pickle
import os

from display import display

def reuters_batches(batch_size):

    X_train = fetch_rcv1()

    return X_train


def news20_minibatches(batch_size):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    if not os.path.isfile('data/news20.pkl'):
        texts = newsgroups_train.data
        labels = newsgroups_train.target
        print(len(texts))

        stemmer = stem.snowball.EnglishStemmer()

        data = [' '.join([stemmer.stem(w) for w in text_to_word_sequence(sent)]) for sent in texts]

        vectorizer = CountVectorizer(stop_words=stopwords.words('english'), max_features=2000)
        x_train = vectorizer.fit_transform(data)

        x_train = x_train.toarray()
        x_train = x_train.astype('float32')
        x_train /= 255
        dataset = {'data': x_train, 'labels': labels}
        pickle.dump(dataset, open('data/news20.pkl', 'wb'))
    else:
        dataset = pickle.load(open('data/news20.pkl', 'rb'))

    x_train = dataset['data']
    num_batches = int(len(x_train) / batch_size)
    batchdata = [x_train[b * batch_size:b * batch_size + batch_size] for b in np.arange(num_batches)]

    return batchdata


def news20_data():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    if not os.path.isfile('data/news20.pkl'):
        texts = newsgroups_train.data
        labels = newsgroups_train.target
        print(len(texts))

        stemmer = stem.snowball.EnglishStemmer()

        data = [' '.join([stemmer.stem(w) for w in text_to_word_sequence(sent)]) for sent in texts]

        vectorizer = CountVectorizer(stop_words=stopwords.words('english'), max_features=2000)
        x_train = vectorizer.fit_transform(data)

        x_train = x_train.toarray()
        x_train = x_train.astype('float32')
        x_train /= 255
        dataset = {'data': x_train, 'labels': labels}
        pickle.dump(dataset, open('data/news20.pkl', 'wb'))
    else:
        dataset = pickle.load(open('data/news20.pkl', 'rb'))

    return dataset['data'], dataset['labels']


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


def mnist_data():
    num_dim = 784
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_cases = len(X_train)
    X_train = X_train.reshape(num_cases, num_dim)

    X_train = X_train.astype('float32')
    X_train /= 255

    #display(batchdata[0].reshape(100, 28,28), batchdata[2].reshape(100, 28,28))

    return X_train, y_train



if __name__ == '__main__':
    data, labels = news20_data()

    print(len(data))
    print(data[0].shape)
    print(len(labels))
    print(labels[0].shape)