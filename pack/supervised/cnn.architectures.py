from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from keras.models import Model

def alexnet(in_shape=(227 ,227 ,3), n_classes=1000, opt='sgd'):
    in_layer = Input(in_shape)
    conv1 = Conv2D(96, 11, strides=4, activation='relu')(in_layer)
    pool1 = MaxPool2D(3, 2)(conv1)
    conv2 = Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)
    pool2 = MaxPool2D(3, 2)(conv2)
    conv3 = Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)
    conv4 = Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)
    pool3 = MaxPool2D(3, 2)(conv4)
    flattened = Flatten()(pool3)
    dense1 = Dense(4096, activation='relu')(flattened)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(4096, activation='relu')(drop1)
    drop2 = Dropout(0.5)(dense2)
    preds = Dense(n_classes, activation='softmax')(drop2)

    model = Model(in_layer, preds)
    extractor = Model(in_layer, flattened)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model, extractor