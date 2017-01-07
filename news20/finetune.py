import pickle

from keras.callbacks import ModelCheckpoint
from keras.engine import Input
from keras.layers import Dense
from keras.models import Model

from news20.dataset import news20_data


def build_news20_model(numdim):
    input = Input(shape=(numdim,))
    x = Dense(1000, activation='sigmoid')(input)
    x = Dense(500, activation='sigmoid')(x)
    x = Dense(250, activation='sigmoid')(x)
    encoded = Dense(2, activation='linear')(x)
    y = Dense(500, activation='sigmoid')(encoded)
    y = Dense(250, activation='sigmoid')(y)
    y = Dense(1000, activation='sigmoid')(y)
    decoded = Dense(numdim, activation='sigmoid')(y)

    autoencoder = Model(input=input, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

    encoder = Model(input=input, output=encoded)
    encoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return autoencoder, encoder


def finetune_news20(x_data, model_path):
    numdim = x_data[0].shape[0]

    layer1 = pickle.load(open(model_path + '/layer1.pkl', 'rb'))
    layer2 = pickle.load(open(model_path + '/layer2.pkl', 'rb'))
    layer3 = pickle.load(open(model_path + '/layer3.pkl', 'rb'))
    layer4 = pickle.load(open(model_path + '/layer4.pkl', 'rb'))

    model, encoder = build_news20_model(numdim)

    weights = [
        layer1['vishid'], layer1['hidbiases'].reshape(layer1['hidbiases'].shape[1]),
        layer2['vishid'], layer2['hidbiases'].reshape(layer2['hidbiases'].shape[1]),
        layer3['vishid'], layer3['hidbiases'].reshape(layer3['hidbiases'].shape[1]),
        layer4['vishid'], layer4['hidbiases'].reshape(layer4['hidbiases'].shape[1]),
        layer4['vishid'].T, layer4['visbiases'].reshape(layer4['visbiases'].shape[1]),
        layer3['vishid'].T, layer3['visbiases'].reshape(layer3['visbiases'].shape[1]),
        layer2['vishid'].T, layer2['visbiases'].reshape(layer2['visbiases'].shape[1]),
        layer1['vishid'].T, layer1['visbiases'].reshape(layer1['visbiases'].shape[1])
    ]

    model.set_weights(weights=weights)

    checkpoint = ModelCheckpoint(model_path + '/final-model.hdf5', monitor='loss', save_best_only=True,
                                 mode='min', save_weights_only=True, verbose=1)
    history = model.fit(x_data, x_data, batch_size=100, nb_epoch=200, shuffle=True, verbose=1, callbacks=[checkpoint])


if __name__ == '__main__':
    x_train, _ = news20_data()
    finetune_news20(x_train, 'news20/models')
