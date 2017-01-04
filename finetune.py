import numpy as np
import pickle
from keras.engine import Input
from keras.layers import Dense
from keras.models import Model


def build_model():
    input = Input(shape=(2000,))
    x = Dense(500, activation='sigmoid')(input)
    x = Dense(250, activation='sigmoid')(x)
    encoded = Dense(2, activation='linear')(x)
    y = Dense(250, activation='sigmoid')(encoded)
    y = Dense(500, activation='sigmoid')(y)
    decoded = Dense(2000, activation='sigmoid')(y)

    autoencoder = Model(input=input, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return autoencoder


def finetune(x_data):
    batch_size, dim, num_batches = x_data.shape
    x_data = x_data.reshape(batch_size*num_batches, dim)
    print(x_data.shape)
    layer1 = pickle.load(open('models/layer1.pkl', 'rb'))
    layer2 = pickle.load(open('models/layer2.pkl', 'rb'))
    layer3 = pickle.load(open('models/layer3.pkl', 'rb'))

    model = build_model()


    weights = [
        layer1['vishid'], layer1['hidbiases'].reshape(layer1['hidbiases'].shape[1]),
        layer2['vishid'], layer2['hidbiases'].reshape(layer2['hidbiases'].shape[1]),
        layer3['vishid'], layer3['hidbiases'].reshape(layer3['hidbiases'].shape[1]),
        layer3['vishid'].T, layer3['visbiases'].reshape(layer3['visbiases'].shape[1]),
        layer2['vishid'].T, layer2['visbiases'].reshape(layer2['visbiases'].shape[1]),
        layer1['vishid'].T, layer1['visbiases'].reshape(layer1['visbiases'].shape[1])
    ]

    model.set_weights(weights=weights)

    history = model.fit(x_data, x_data, batch_size=100, nb_epoch=200, verbose=1)




