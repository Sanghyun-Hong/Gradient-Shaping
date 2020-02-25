"""
    Define optimizers (used in Eager Execution)
"""

# tensorflow modules
import tensorflow as tf
from tensorflow.compat.v1.train import GradientDescentOptimizer, AdamOptimizer

# tensorflow-privacy (since we use the bleeding-edge version)
try:
    from tensorflow_privacy.privacy.optimizers import dp_optimizer
except:
    from privacy.optimizers import dp_optimizer


# ------------------------------------------------------------
#  Optimizer define functions
# ------------------------------------------------------------
def define_optimizer(network, lr):
    if 'lr' == network:
        optimizer = AdamOptimizer(lr)
    else:
        optimizer = GradientDescentOptimizer(lr)
    return optimizer

def define_dpoptimizer(network, lr, batch, nclip, noise):
    if 'lr' == network:
        optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                        l2_norm_clip=nclip,
                        noise_multiplier=noise,
                        learning_rate=lr)
    else:
        optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
                        l2_norm_clip=nclip,
                        noise_multiplier=noise,
                        learning_rate=lr)
    return optimizer
