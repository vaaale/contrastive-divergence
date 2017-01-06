import pickle

from keras.callbacks import ModelCheckpoint
from keras.engine import Input
from keras.layers import Dense
from keras.models import Model
import numpy as np


def build_model(numdim):
    input = Input(shape=(numdim,))
    x = Dense(1000, activation='sigmoid')(input)
    x = Dense(250, activation='sigmoid')(x)
    encoded = Dense(2, activation='linear')(x)
    y = Dense(250, activation='sigmoid')(encoded)
    y = Dense(1000, activation='sigmoid')(y)
    decoded = Dense(numdim, activation='sigmoid')(y)

    autoencoder = Model(input=input, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return autoencoder


def finetune(x_data):
    numdim = x_data[0].shape[0]

    layer1 = pickle.load(open('models/layer1.pkl', 'rb'))
    layer2 = pickle.load(open('models/layer2.pkl', 'rb'))
    layer3 = pickle.load(open('models/layer3.pkl', 'rb'))

    model = build_model(numdim)


    weights = [
        layer1['vishid'], layer1['hidbiases'].reshape(layer1['hidbiases'].shape[1]),
        layer2['vishid'], layer2['hidbiases'].reshape(layer2['hidbiases'].shape[1]),
        layer3['vishid'], layer3['hidbiases'].reshape(layer3['hidbiases'].shape[1]),
        layer3['vishid'].T, layer3['visbiases'].reshape(layer3['visbiases'].shape[1]),
        layer2['vishid'].T, layer2['visbiases'].reshape(layer2['visbiases'].shape[1]),
        layer1['vishid'].T, layer1['visbiases'].reshape(layer1['visbiases'].shape[1])
    ]

    model.set_weights(weights=weights)

    checkpoint = ModelCheckpoint('models/final-model.hdf5', monitor='loss', save_best_only=True,
                                 mode='min', save_weights_only=True, verbose=1)
    history = model.fit(x_data, x_data, batch_size=100, nb_epoch=200, verbose=1, callbacks=[checkpoint])



# plt.figure()
# plt.scatter(encoded_txts[:500,0], encoded_txts[:500,1], c=colorlist[:500])
# plt.show()

