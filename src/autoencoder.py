# Source: https://blog.keras.io/building-autoencoders-in-keras.html
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
import keras

import pandas as pd
import numpy as np
import pickle

import logging
from sklearn.feature_extraction.text import TfidfVectorizer


def autoencoder(x):
    '''encode tfidf vectors into lower dimensional space'''
    x_train = x[5000:,:]
    x_test = x[:5000,:]
    input_cols= x.shape[1]
    input_tweet = Input(shape=(input_cols,))
    encoded = Dense(60, activation='tanh')(input_tweet)
    encoded = Dense(30, activation='tanh')(encoded)
    decoded = Dense(60, activation='tanh')(encoded)
    decoded = Dense(input_cols, activation='sigmoid')(decoded)
    # take in tweet and reconstruct it
    autoencoder = Model(input=input_tweet, output=decoded)
    # create encoder
    encoder = Model(input=input_tweet, output=encoded)
    # final layer encoder input shape
    encoded_input = Input(shape=(60,))
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    adam = keras.optimizers.Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))
    encoded_tweets = encoder.predict(np.array(x))
    return encoded_tweets, encoder
