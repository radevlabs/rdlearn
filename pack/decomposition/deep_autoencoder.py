from ..base import DecompositionBase
from ..utils import Timer
from ..utils import sprint
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import Callback
import numpy as np
import os

class Verbose(Callback):
    def __init__(self, max_epoch, max_loss=None):
        super(Callback).__init__()
        self.max_loss = max_loss
        self.max_epoch = max_epoch

    def on_epoch_end(self, epoch, logs={}):
        loss = logs['loss']
        sprint(f'\rEpoch {epoch + 1}/{self.max_epoch} - loss: {loss:.20f}')

    def on_train_end(self, logs={}):
        print('')

class DeepAutoencoder(DecompositionBase):
    """

    """

    def __init__(self, encoder_model, loss='mse', hidden_activation='sigmoid', output_activation='linear', use_gpu=True, optimizer='rmsprop', n_features=2):
        """

        :param n_features:
        :param encoder_model:
        :param loss:
        :param hidden_activation:
        :param output_activation:
        :param use_gpu:
        :param optimizer:
        """
        super().__init__(n_features=n_features)
        self._hactivation = hidden_activation
        self._oactivation = output_activation
        self._loss = loss
        self._encoder_model = encoder_model
        self._optimizer = 'rmsprop'
        self._scaler = None
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def fit(self, x, epochs=100, minmaxscaler=(0, 1), verbose=True):
        """

        :param x:
        :param epochs:
        :return:
        """
        x = np.array(x)
        if minmaxscaler is not None:
            scaler = MinMaxScaler(feature_range=minmaxscaler)
            scaler.fit(x)
            self._scaler = scaler
            x = self._scaler.transform(x)

        self.timer = Timer().start()

        autoencoder, self._encoder, self.activations = self._build(x)
        autoencoder.compile(optimizer=self._optimizer, loss='mse')

        if verbose:
            autoencoder.summary()
            self.history = autoencoder.fit(x=x,
                                           y=x,
                                           epochs=epochs,
                                           batch_size=256,
                                           verbose=0,
                                           callbacks=[Verbose(max_epoch=epochs)])
        else:
            self.history = autoencoder.fit(x=x,
                                           y=x,
                                           epochs=epochs,
                                           batch_size=256,
                                           verbose=0)

        self.timer.end()

        return self

    def transform(self, x):
        """

        :param x:
        :return:
        """
        if self._scaler is not None:
            x = self._scaler.transform(x)
        return self._encoder.predict(x)

    def _build(self, x):
        """

        :return:
        """
        from keras.layers import Input, Dense
        from keras.models import Model
        
        emodel = self._encoder_model.split('-')
        emodel = list(map(int, emodel))

        coded_model = emodel[-1]
        del emodel[-1]

        dmodel = emodel.copy()
        dmodel.reverse()

        input_ = Input(shape=(x.shape[1],))
        activations = []

        if len(emodel) > 0:
            encoded = Dense(emodel[0], activation=self._hactivation)(input_)
            activations.append(self._hactivation)
            del emodel[0]
            for n in emodel:
                encoded = Dense(n, activation=self._hactivation)(encoded)
                activations.append(self._hactivation)
            encoded = Dense(coded_model, activation=self._hactivation)(encoded)
            activations.append(self._hactivation)
        else:
            encoded = Dense(coded_model, activation=self._hactivation)(input_)
            activations.append(self._hactivation)

        if len(dmodel) > 0:
            decoded = Dense(dmodel[0], activation=self._hactivation)(encoded)
            activations.append(self._hactivation)
            del dmodel[0]
            for n in dmodel:
                decoded = Dense(n, activation=self._hactivation)(decoded)
                activations.append(self._hactivation)
            decoded = Dense(x.shape[1], activation=self._oactivation)(decoded)
            activations.append(self._oactivation)
        else:
            decoded = Dense(x.shape[1], activation=self._oactivation)(encoded)
            activations.append(self._oactivation)

        encoder = Model(input_, encoded)
        autoencoder = Model(input_, decoded)

        return autoencoder, encoder, activations
