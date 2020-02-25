"""
    Datasets: loader library for the various dataset
    [Return types: train - float32, labels - int32]
"""
# basics
import sys
import socket
import numpy as np
from distutils.version import LooseVersion

# sklearn
import scipy.io as sio
from sklearn import metrics

# tensorflow modules
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist, cifar10

# custom libs
from utils import io


# ------------------------------------------------------------
#  Preprocess functions
# ------------------------------------------------------------
def _linearize(x_train, x_test, format='IMG'):
    """
        Linearize the datasets
    """
    # for the quantized 0~255 images
    if 'IMG' == format:
        x_train = x_train / 255.
        x_test  = x_test  / 255.
        return x_train, x_test

    # unknown cases
    else:
        assert False, ('[_linearize] {} - unknown format'.format(format))

def _normalize(dataset, x_train, x_test):
    """
        Standardize the datasets
    """
    # for Image datasets
    if dataset in ['fashion_mnist', 'cifar10']:
        data_mean, data_std = \
            np.mean(x_train, axis=(0,1,2,3)), \
            np.std(x_train, axis=(0,1,2,3))
        x_train = (x_train - data_mean)/(data_std + 1e-7)
        x_test  = (x_test - data_mean)/(data_std + 1e-7)
        return x_train, x_test

    # unknown cases
    else:
        assert False, ('[_normalize] {} - normalize'.format(dataset))
    # done.


# ------------------------------------------------------------
#  To load very specific datasets
# ------------------------------------------------------------
def _load_numpy_dataset(datafile):
    with np.load(datafile) as infile:
        x_train, y_train, x_test, y_test = \
            [infile['arr_%d' % i] for i in range(len(infile.files))]
    (x_train, y_train) = \
        np.array(x_train, dtype=np.float32), \
        np.array(y_train, dtype=np.int32)
    (x_test, y_test) = \
        np.array(x_test, dtype=np.float32), \
        np.array(y_test, dtype=np.int32)
    return (x_train, y_train), (x_test, y_test)

def _load_subtask_data(datapath):
    entire_data = io.load_from_pickle(datapath)
    return (entire_data['x-train'], entire_data['y-train']), \
           (entire_data['x-test'],  entire_data['y-test'])

def _load_holdout_data(datapath):
    entire_data = io.load_from_pickle(datapath)
    return (entire_data['x-train'], entire_data['y-train']), \
           (entire_data['x-test'],  entire_data['y-test']), \
           (entire_data['x-hout'],  entire_data['y-hout'])


# ------------------------------------------------------------
#  Define functions
# ------------------------------------------------------------
def define_dataset(dataset, datapath, linearize=True):
    # --------------------
    # Fashion-MNIST dataset
    # --------------------
    if 'fashion_mnist' == dataset:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        # : expand the dimension
        x_train = np.expand_dims(x_train, axis=3)
        x_test  = np.expand_dims(x_test,  axis=3)
        # : linearize data
        if linearize:
            (x_train, x_test) = _linearize(x_train, x_test)
            # :: convert to float32 and int32
            (x_train, x_test) = \
                x_train.astype('float32'), x_test.astype('float32')
            (y_train, y_test) = \
                y_train.astype('int32'), y_test.astype('int32')
        return (x_train, y_train), (x_test, y_test)

    # --------------------
    # CIFAR10 dataset
    # --------------------
    elif 'cifar10' == dataset:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # : squeeze the dimension
        y_train = np.squeeze(y_train, axis=1)
        y_test  = np.squeeze(y_test,  axis=1)
        # : linearize data, and convert to float32
        if linearize:
            (x_train, x_test) = _linearize(x_train, x_test)
            # :: convert to float32 and int32
            (x_train, x_test) = \
                x_train.astype('float32'), x_test.astype('float32')
            (y_train, y_test) = \
                y_train.astype('int32'), y_test.astype('int32')
        return (x_train, y_train), (x_test, y_test)

    # --------------------
    # Purchase-100 dataset
    # --------------------
    elif 'purchases' == dataset:
        (x_train, y_train), (x_test, y_test) = _load_numpy_dataset(datapath)
        # : already linearized and converted to (float32, int32)
        return (x_train, y_train), (x_test, y_test)

    # --------------------
    # Subtask datasets
    # --------------------
    elif 'subtask' == dataset:
        (x_train, y_train), (x_test, y_test) = _load_subtask_data(datapath)
        # : linearize data, and convert to float32
        if linearize:
            (x_train, x_test) = _linearize(x_train, x_test)
            # :: convert to float32 and int32
            (x_train, x_test) = \
                x_train.astype('float32'), x_test.astype('float32')
            (y_train, y_test) = \
                y_train.astype('int32'), y_test.astype('int32')
        return (x_train, y_train), (x_test, y_test)
    else:
        assert False, \
            ('[define_dataset] Error: undefined dataset - {}'.format(dataset))
    # done.

def define_dataset_w_holdout(dataset, datapth):
    # --------------------
    # Subtask datasets
    # --------------------
    if 'subtask' == dataset:
        (x_train, y_train), (x_test, y_test), \
            (x_hout, y_hout) = _load_holdout_data(datapth)
        # : linearize data
        x_train = x_train / 255.
        x_test  = x_test  / 255.
        x_hout  = x_hout  / 255.
        # :: convert to float32 and int32
        (x_train, x_test, x_hout) = \
            x_train.astype('float32'), x_test.astype('float32'), x_hout.astype('float32')
        (y_train, y_test, y_hout) = \
            y_train.astype('int32'), y_test.astype('int32'), y_hout.astype('int32')
        return (x_train, y_train), (x_test, y_test), (x_hout, y_hout)
    else:
        assert False, \
            ('[define_dataset] Error: undefined dataset - {}'.format(dataset))
    # done.


# ------------------------------------------------------------
#  TF <-> Numpy conversion interface
# ------------------------------------------------------------
def convert_to_tf_dataset(x_data, y_data, batch=1, shuffle=False):
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_data, tf.float32),
         tf.cast(y_data, tf.int32)))
    if shuffle:
        tf_dataset = tf_dataset.shuffle( \
            buffer_size=x_data.shape[0], seed=215).batch(batch)
    else:
        tf_dataset = tf_dataset.batch(batch)
    return tf_dataset


# ------------------------------------------------------------
#  To load the poison data
# ------------------------------------------------------------
def load_lfip_poisons(poison_path):
    poison_data = io.load_from_pickle(poison_path)
    return (poison_data['x-train'], poison_data['y-train']), \
        (poison_data['x-test'], poison_data['y-test']), \
        (poison_data['x-poisons'], poison_data['y-poisons'])

def load_slab_poisons(poison_path):
    numpy_data = sio.loadmat(poison_path)
    # need to convert -1 to 0
    y_train = numpy_data['y_train']
    y_train[y_train == -1] = 0
    y_test  = numpy_data['y_test']
    y_test[y_test == -1] = 0
    y_poison= numpy_data['bestY'][0]
    y_poison[y_poison == -1] = 0
    # return...
    return (numpy_data['X_train'], y_train.reshape(-1)), \
        (numpy_data['X_test'], y_test.reshape(-1)), \
        (numpy_data['bestX'][0], y_poison.reshape(-1))

def load_backdoor_poisons(poison_path):
    poison_data = io.load_from_pickle(poison_path)
    # load the data
    bx_train = poison_data['x-train']
    by_train = poison_data['y-train']
    bx_test  = poison_data['x-test']
    by_test  = poison_data['y-test']
    # linearize the data
    (bx_train, bx_test) = _linearize(bx_train, bx_test)
    return (bx_train, by_train), (bx_test, by_test)

def load_poisons(datapth, x_test, y_test, sort=False):
    # the index of the current target
    tfilename = datapth.split('/')[-1]
    tar_index = tfilename.replace('.pkl', '')
    tar_index = int(tar_index.split('_')[-1])

    # load the attacker's target
    x_target  = x_test[tar_index:(tar_index+1)]
    y_target  = y_test[tar_index:(tar_index+1)]

    # data-holder (poisons)
    x_poisons = []
    y_poisons = []

    # extract the poison instances
    poison_data = io.load_from_pickle(datapth)

    # Note: sort by the distance order in the feature space (multipoison case)
    if sort:
        poison_data = sorted(poison_data, key=lambda x: x['perturbs'])

    for each_data in poison_data:
        x_poisons.append(each_data['instance'])         # this is [0,1] data
        y_poisons.append(each_data['class'])

    # convert to the numpy array: data-holders
    x_poisons = np.array(x_poisons)
    y_poisons = np.array(y_poisons)

    # convert to the compatible format (when exists)
    if (x_poisons.size != 0) and (y_poisons.size != 0):
        x_poisons = np.concatenate(x_poisons, axis=0)
        y_poisons = np.asarray(y_poisons, dtype=np.int32)

    # return the poisons
    return (x_poisons, y_poisons), (x_target, y_target)
