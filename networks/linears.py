"""
    Linear Models (LinearRegression)
"""
# basic
import numpy as np

# JAX (for analysis)
from jax.experimental import stax

# tensorflow
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten



# ----------------------------------------------------------------
# Linear Regression Model (Keras)
# ----------------------------------------------------------------
class LinearRegression(Model):
    def __init__(self, nout, activation='linear', vars=None):
        super(LinearRegression, self).__init__()

        # store the variables
        self.vars = vars

        # input/output: reshape the data and do logit regression
        self.flat = Flatten()
        if not vars:
            self.out = Dense(nout) # kernel_regularizer=l2(l2_ratio)) --- later on
        else:
            self.out = Dense(nout, \
                kernel_initializer=self._kern_init, \
                bias_initializer=self._bias_init)

    def call(self, x, training=False):
        x = self.flat(x)
        x = self.out(x)
        return x, tf.identity(x)

    """
        Initializer for the kernal/bias --- Names are pre-defined by TF
    """
    def _kern_init(self, shape, dtype=None):
        return self.vars['linear_regression/dense/kernel:0']

    def _bias_init(self, shape, dtype=None):
        return self.vars['linear_regression/dense/bias:0']


# ----------------------------------------------------------------
# Linear Regression Model (JAX)
# ----------------------------------------------------------------
def LinearRegressionJAX(nout, activation='linear'):
    return stax.serial(
            stax.Flatten,
            stax.Dense(nout)
        )
