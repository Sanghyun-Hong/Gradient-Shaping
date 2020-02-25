"""
    Convolutional Neural Networks
"""
# basic
import numpy as np

# JAX (for analysis)
from jax.experimental import stax

# tensorflow
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten


# ----------------------------------------------------------------
#  Convolutional neural networks
# ----------------------------------------------------------------
class BadNet(Model):
    def __init__(self, nhidden, nout, activation='relu', ishape=(28, 28, 1)):
        super(BadNet, self).__init__()

        # layers: convs
        self.conv1 = Conv2D(16, (3, 3), activation=activation, padding='same', input_shape=ishape)
        self.conv2 = Conv2D(32, (3, 3), activation=activation, padding='same')
        self.pool1 = MaxPooling2D((2, 2))

        # layers: linears
        self.flat1 = Flatten()
        self.fc1   = Dense(nhidden, activation=activation)
        self.fc2   = Dense(nout)    # logits

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.flat1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvNet(Model):
    def __init__(self, nhidden, nout, activation='relu', ishape=(32, 32, 3), vars=None):
        super(ConvNet, self).__init__()

        # store the variables
        self.vars = vars

        # layers
        if not vars:
            # convolutional parts
            # Saved: kernel_regularizer=l2(l2_ratio) --- later on
            self.conv1 = Conv2D(16, (3, 3), activation=activation, padding='same', input_shape=ishape)
            self.conv2 = Conv2D(32, (3, 3), activation=activation, padding='same')
            self.pool1 = MaxPooling2D((2, 2))

            # linears
            self.flat1 = Flatten()
            self.fc1   = Dense(nhidden, activation=activation)
            self.fc2   = Dense(nout)    # logits
        else:
            self.conv1 = Conv2D(16, (3, 3), \
                activation=activation, \
                padding='same', input_shape=ishape, \
                kernel_initializer=self._kernc1_init, \
                bias_initializer=self._biasc1_init)
            self.conv2 = Conv2D(32, (3, 3), \
                activation=activation, \
                padding='same', \
                kernel_initializer=self._kernc2_init, \
                bias_initializer=self._biasc2_init)
            self.pool1 = MaxPooling2D((2, 2))

            # linears
            self.flat1 = Flatten()
            self.fc1   = Dense(nhidden, \
                activation=activation, \
                kernel_initializer=self._kernd1_init, \
                bias_initializer=self._biasd1_init)
            self.fc2   = Dense(nout, \
                kernel_initializer=self._kernd2_init, \
                bias_initializer=self._biasd2_init)    # logits

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.flat1(x)
        p = self.fc1(x)             # save the penultimate output
        x = self.fc2(p)
        return x, p

    """
        Initializer for the kernal/bias --- Names are correct
    """
    def _kernc1_init(self, shape, dtype=None):
        return self.vars['conv_net/conv2d/kernel:0']

    def _biasc1_init(self, shape, dtype=None):
        return self.vars['conv_net/conv2d/bias:0']

    def _kernc2_init(self, shape, dtype=None):
        return self.vars['conv_net/conv2d_1/kernel:0']

    def _biasc2_init(self, shape, dtype=None):
        return self.vars['conv_net/conv2d_1/bias:0']

    def _kernd1_init(self, shape, dtype=None):
        return self.vars['conv_net/dense/kernel:0']

    def _biasd1_init(self, shape, dtype=None):
        return self.vars['conv_net/dense/bias:0']

    def _kernd2_init(self, shape, dtype=None):
        return self.vars['conv_net/dense_1/kernel:0']

    def _biasd2_init(self, shape, dtype=None):
        return self.vars['conv_net/dense_1/bias:0']


# ----------------------------------------------------------------
# BadNet Model (JAX)
# ----------------------------------------------------------------
def BadNetJAX(nhidden, nout):
    return stax.serial(
            stax.Conv(16, (3, 3), padding='same'),
            stax.Relu,
            stax.Conv(32, (3, 3), padding='same'),
            stax.Relu,
            stax.MaxPool((2, 2), (2, 2)),
            stax.Flatten,
            stax.Dense(nhidden),
            stax.Relu,
            stax.Dense(nout)
        )
