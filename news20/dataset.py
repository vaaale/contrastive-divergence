import os
import pickle

import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from nltk import stem
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


def build_dataset():
    newsgroups_train = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    texts = newsgroups_train.data
    labels = newsgroups_train.target
    print(len(texts))

    stemmer = stem.snowball.EnglishStemmer()

    texts = [' '.join([stemmer.stem(w) for w in text_to_word_sequence(sent)]) for sent in texts]
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'), max_features=2000)
    x_train = vectorizer.fit_transform(texts)

    x_train = x_train.toarray()
    x_train = x_train.astype('float32')

    x_train = normalize(x_train, axis=1, norm='l2')

    return x_train, labels


def news20_minibatches(batch_size):
    if not os.path.isfile('news20/data/news20.pkl'):
        x_train, labels = build_dataset()
        dataset = {'data': x_train, 'labels': labels}
        pickle.dump(dataset, open('news20/data/news20.pkl', 'wb'))
    else:
        dataset = pickle.load(open('news20/data/news20.pkl', 'rb'))

    x_train = dataset['data']
    num_batches = int(len(x_train) / batch_size)
    batchdata = [x_train[b * batch_size:b * batch_size + batch_size] for b in np.arange(num_batches)]

    return batchdata


def news20_data():
    if not os.path.isfile('news20/data/news20.pkl'):
        x_train, labels = build_dataset()
        dataset = {'data': x_train, 'labels': labels}
        pickle.dump(dataset, open('news20/data/news20.pkl', 'wb'))
    else:
        dataset = pickle.load(open('news20/data/news20.pkl', 'rb'))

    return dataset['data'], dataset['labels']


if __name__ == '__main__':
    data, labels = news20_data()

    print(len(data))
    print(data[0].shape)
    print(len(labels))
    print(labels[0].shape)
