"""
    Multi-layer perceptrons
"""
# basic
import numpy as np

# JAX (for analysis)
from jax.experimental import stax

# tensorflow
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten, Dropout


# ----------------------------------------------------------------
#  Multi layer perceptrons (feedfoward networks)
# ----------------------------------------------------------------
class ShallowMLP(Model):
    def __init__(self, nhidden, nout, activation='relu', vars=None):
        super(ShallowMLP, self).__init__()

        # store the variables
        self.vars = vars

        # input: reshape the data
        self.flat = Flatten()

        # hidden layers and output
        if not vars:
            self.fc1  = Dense(nhidden, activation=activation)   # kernel_regularizer=l2(l2_ratio)) -- later on
            self.fc2  = Dense(nhidden, activation=activation)
            self.out  = Dense(nout) # logits
        else:
            self.fc1  = Dense(nhidden, \
                activation=activation, \
                kernel_initializer=self._kern0_init, \
                bias_initializer=self._bias0_init)
            self.fc2  = Dense(nhidden, \
                activation=activation, \
                kernel_initializer=self._kern1_init, \
                bias_initializer=self._bias1_init)
            self.out  = Dense(nout, \
                kernel_initializer=self._kern2_init, \
                bias_initializer=self._bias2_init)

    def call(self, x, training=False):
        x = self.flat(x)
        x = self.fc1(x)
        p = self.fc2(x)         # save the penultimate output
        x = self.out(p)
        return x, p

    """
        Initializer for the kernal/bias --- Names are correct
    """
    def _kern0_init(self, shape, dtype=None):
        return self.vars['shallow_mlp/dense/kernel:0']

    def _bias0_init(self, shape, dtype=None):
        return self.vars['shallow_mlp/dense/bias:0']

    def _kern1_init(self, shape, dtype=None):
        return self.vars['shallow_mlp/dense_1/kernel:0']

    def _bias1_init(self, shape, dtype=None):
        return self.vars['shallow_mlp/dense_1/bias:0']

    def _kern2_init(self, shape, dtype=None):
        return self.vars['shallow_mlp/dense_2/kernel:0']

    def _bias2_init(self, shape, dtype=None):
        return self.vars['shallow_mlp/dense_2/bias:0']


# ----------------------------------------------------------------
# MLP Model (JAX)
# ----------------------------------------------------------------
def ShallowMLPJAX(nhidden, nout):
    return stax.serial(
            stax.Flatten,
            stax.Dense(nhidden),
            stax.Relu,
            stax.Dense(nhidden),
            stax.Relu,
            stax.Dense(nout)
        )
