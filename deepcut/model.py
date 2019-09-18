#!/usr/bin/env python
# encoding: utf-8
from keras.models import Model
from keras.layers import Input, Dense, Embedding, \
    Concatenate, Flatten, SpatialDropout1D, \
    BatchNormalization, Conv1D, Maximum, ZeroPadding1D, Lambda
from keras.layers import TimeDistributed
from keras.optimizers import Adam

import os

def conv_unit(inp, n_gram, no_word=200, window=2):
    out = Conv1D(no_word, window, strides=1, padding="valid", activation='relu')(inp)
    out = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(out)
    out = ZeroPadding1D(padding=(0, window - 1))(out)
    return out

def zero(a, when=""):

    if not os.environ.get("DEEPCUT_ZERO_LAYER", False):
        return a

    print("---------")
    print(os.environ.get("DEEPCUT_ZERO_LAYER"))

    cases = os.environ.get("DEEPCUT_ZERO_LAYER").split(",")
    if when in cases:
        print("Zeroing %s" % when)
        return Lambda(lambda x: x*0.0)(a)
    else:
        return a

def get_convo_nn2(no_word=200, n_gram=21, no_char=178):
    input1 = Input(shape=(n_gram,))
    input2 = Input(shape=(n_gram,))

    a = Embedding(no_char, 32, input_length=n_gram)(input1)
    a = SpatialDropout1D(0.15)(a)
    a = BatchNormalization()(a)

    a_concat = []
    for i in range(1,9):
        ac = zero(a, when="conv-%d" % i)
        a_concat.append(conv_unit(ac, n_gram, no_word, window=i))
    for i in range(9,12):
        ac = zero(a, when="conv-%d" % i)
        cc = conv_unit(ac, n_gram, no_word - 50, window=i)
        a_concat.append(cc)

    ac = zero(a, when="conv-%d" % 12)
    cc = conv_unit(ac, n_gram, no_word - 100, window=12)
    a_concat.append(cc)
    a_sum = Maximum()(a_concat)

    b = Embedding(12, 12, input_length=n_gram)(input2)
    b = SpatialDropout1D(0.15)(b)
    b = zero(b, when="ch-type-emb")

    x = Concatenate(axis=-1)([a, a_sum, b])
    #x = Concatenate(axis=-1)([a_sum, b])
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2], outputs=out)
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy', metrics=['acc'])
    return model

def shrink_model(model):
    input1 = model.get_layer("input_1").input
    input2 = model.get_layer("input_2").input


    a = model.get_layer("embedding_1")(input1)
    a = model.get_layer("batch_normalization_1")(a)

    a_concat = []

    for i in range(1,13):
        if i in [7, 9, 10]: # got from experiments
            print("Skiping conv-%d" % i)
            continue

        ac = model.get_layer("conv1d_%d" % i)(a)
        ac = model.get_layer("time_distributed_%d" % i)(ac)
        ac = model.get_layer("zero_padding1d_%d" % i)(ac)

        a_concat.append(ac)

    a_sum = Maximum()(a_concat)

    b = model.get_layer("embedding_2")(input2)
    x = Concatenate(axis=-1)([a, a_sum, b])

    x = model.get_layer("batch_normalization_2")(x)

    x = model.get_layer("flatten_1")(x)
    x = model.get_layer("dense_13")(x)
    out = model.get_layer("dense_14")(x)

    shrinked_model = Model(inputs=[input1, input2], outputs=out)

    return shrinked_model