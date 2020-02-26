"""
    Analyze feature collison (w. Eager Execution of TF)
"""
import csv, os, sys
# suppress tensorflow errors -- too many, what's the purpose?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import argparse
import itertools
import numpy as np

# JAX models (for privacy analysis)
from jax import grad, partial, random, tree_util, vmap
from jax.lax import stop_gradient
from jax.experimental import optimizers, stax
from networks.linears import LinearRegressionJAX
from networks.mlps import ShallowMLPJAX

# tensorflow modules
import tensorflow as tf
from tensorflow.compat.v1.logging import set_verbosity, ERROR

# custom libs
from utils import io
from utils import datasets, models, optims


# ------------------------------------------------------------
#  Global variables
# ------------------------------------------------------------
_rand_fix = 215
_verbose  = True
_fn_holder= None
_dataindex= {
    # [For the subset (FashionMNIST 3/4) of the entire FashionMNIST]
    'subtask': {
        'one'  : { 0: 0, 1: 1, },   # choose 0 - 0th, 1 - 1th
        'multi': {
            0: [0, 2, 5, 8, 9, 10, 11, 12, 13, 15, \
                16, 19, 20, 21, 23, 32, 33, 35, 36, 40, \
                41, 42, 43, 45, 46, 48, 49, 54, 55, 56, \
                57, 58, 59, 61, 63, 64, 65, 69, 73, 77, \
                79, 80, 84, 86, 89, 90, 91, 93, 95, 99, \
                101, 102, 104, 109, 110, 112, 114, 117, 118, 120, \
                122, 125, 126, 127, 134, 135, 139, 141, 143, 144, \
                147, 148, 149, 150, 151, 153, 155, 158, 160, 161, \
                162, 163, 167, 168, 169, 172, 174, 175, 177, 178, \
                180, 182, 184, 185, 187, 188, 189, 191, 196, 198],
            1: [1, 3, 4, 6, 7, 14, 17, 18, 22, 24, \
                25, 26, 27, 28, 29, 30, 31, 34, 37, 38, \
                39, 44, 47, 50, 51, 52, 53, 60, 62, 66, \
                67, 68, 70, 71, 72, 74, 75, 76, 78, 81, \
                82, 83, 85, 87, 88, 92, 94, 96, 97, 98, \
                100, 103, 105, 106, 107, 108, 111, 113, 115, 116, \
                119, 121, 123, 124, 128, 129, 130, 131, 132, 133, \
                136, 137, 138, 140, 142, 145, 146, 152, 154, 156, \
                157, 159, 164, 165, 166, 170, 171, 173, 176, 179, \
                181, 183, 186, 190, 192, 193, 194, 195, 197, 199],
        },
    },
    # [For the entire FMNIST]
    'fashion_mnist': {
        'one'  : { 0: 3, 1: 19, },  # 0th means the first class (i.e. 3), and the other is 1 (i.e., 4)
        'multi': {
            0: [3, 20, 25, 31, 47, 49, 50, 51, 58, 59, \
                70, 73, 81, 91, 94, 114, 190, 204, 211, 215, \
                223, 247, 248, 250, 251, 259, 261, 268, 277, 318, \
                323, 326, 327, 332, 334, 351, 370, 374, 375, 378, \
                395, 407, 439, 453, 455, 471, 478, 488, 496, 500, \
                508, 509, 536, 550, 568, 571, 595, 596, 609, 626, \
                631, 642, 649, 667, 670, 673, 686, 701, 709, 714, \
                725, 756, 757, 802, 824, 827, 835, 841, 854, 863, \
                868, 871, 876, 888, 911, 929, 944, 948, 951, 961, \
                996, 997, 1011, 1026, 1037, 1038, 1039, 1049, 1053, 1063],
            1: [19, 22, 24, 28, 29, 68, 75, 76, 96, 117, \
                128, 134, 139, 179, 181, 185, 188, 194, 205, 238, \
                239, 240, 241, 242, 255, 263, 264, 290, 296, 311, \
                312, 315, 324, 339, 362, 381, 387, 388, 396, 397, \
                399, 412, 413, 426, 433, 442, 457, 463, 464, 468, \
                473, 481, 486, 535, 541, 555, 557, 567, 578, 603, \
                612, 615, 622, 625, 634, 648, 656, 658, 672, 677, \
                687, 698, 720, 726, 731, 733, 743, 752, 763, 778, \
                788, 810, 833, 842, 851, 878, 889, 912, 923, 933, \
                938, 966, 977, 979, 993, 1021, 1027, 1032, 1046, 1057],
        },
    },
}


# ------------------------------------------------------------
#  Perform interpolation
# ------------------------------------------------------------
def _do_interpolation(data, labels, dindex, imode, alpha):
    # sanity check
    assert (0. <= alpha <= 1.), ('Error: alpha [{}] should be in [0,1]'.format(alpha))

    # load the data indexes
    data_indexes = dindex[imode]

    # choose the indexes, currently only support 0/1 - binary data
    if 'one' == imode:
        data0  = data[data_indexes[0]:(data_indexes[0]+1)]
        data1  = data[data_indexes[1]:(data_indexes[1]+1)]
        labels = labels[data_indexes[0]:(data_indexes[0]+1)]

    elif 'multi' == imode:
        data0  = data[np.array(data_indexes[0])]
        data1  = data[np.array(data_indexes[1])]
        labels = labels[np.array(data_indexes[0])]

    # do interpolation (clip within [0,1])
    datai = (1-alpha)*data0 + alpha*data1
    datai = np.clip(datai, 0., 1.)
    return (datai, labels)


# ------------------------------------------------------------
#  Valiadation datasets
# ------------------------------------------------------------
def _validate(model, validset):
    corrects = []
    for (_, (data, labels)) in enumerate(validset.take(-1)):
        logits, penultimate = model(data, training=False)
        predicts = tf.argmax(logits, axis=1)
        predicts = tf.dtypes.cast(predicts, tf.int32)
        corrects.append(tf.equal(predicts, labels).numpy())
    cur_acc = np.mean(corrects)
    return cur_acc


# ------------------------------------------------------------
#  JAX related
# ------------------------------------------------------------
def _data_loader(x_train, y_train, batch_size, num_batches):
    # [Note]: only use the numpy random here; otherwise, all should be JAX numpy
    from numpy import random as npramdom
    from numpy import argwhere as nargwhere
    rstate = npramdom.RandomState(_rand_fix)
    while True:
        permutation = rstate.permutation(x_train.shape[0])
        for bidx in range(num_batches):
            batch_indexes = permutation[bidx*batch_size:(bidx+1)*batch_size]
            yield x_train[batch_indexes], y_train[batch_indexes]

def _shape_data(data, labels, dummy_dim=False):
    orig_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
    return np.reshape(data, orig_shape), labels

def _convert_to_onehot(labels, total=10):
    # use the original numpy functions
    from numpy import zeros as nzeros
    from numpy import arange as narange
    # to one-hot
    new_labels = nzeros((labels.size, total))
    new_labels[narange(labels.size), labels] = 1.
    return new_labels

def _validate_JAX(params, applyfn, data, labels):
    predict = applyfn(params, data)
    predict = np.argmax(predict, axis=1)
    # convert to index encoding
    oracles = np.argmax(labels, axis=1)
    return np.mean(predict == oracles)

def _loss(params, batch):
    global _fn_holder
    data, labels = batch
    logits = _fn_holder(params, data)
    logits = stax.logsoftmax(logits)  # log normalize
    return -np.mean(np.sum(logits * labels, axis=1))  # cross entropy loss

def _split_poisons_JAX( \
    poison_data, poison_labels, total_data, total_labels, verbose=False):
    """
        Identify whether the batch includes poisons
    """
    # reduce one extra dimension, added, from the total data
    total_dims = (total_data.shape[0],) + tuple(total_data.shape[2:])
    total_data = total_data.reshape(total_dims)

    # data-holder
    poison_indexes = []

    # iterate over the total data, and see if any data is in poisons
    for pidx, each_poison in enumerate(poison_data):
        # : search the inclusion
        if len(each_poison.shape) == 1:
            search_result = (each_poison == total_data).all((1))
        else:
            search_result = (each_poison == total_data).all((1, 2, 3))
        # : search the index
        search_tindex = [i for i, tfval in enumerate(search_result) if tfval]
        # : skip, if the index is the same
        if not search_tindex: continue
        # : only include when the labels are correct
        if (poison_labels[pidx] == total_labels[search_tindex[0]]).any():
            poison_indexes.append(search_tindex[0])

    # split into two ...
    poison_indexes = np.array(poison_indexes)
    clean_indexes  = np.array([ \
        didx for didx in range(len(total_data)) if didx not in poison_indexes])

    # expand the data back
    total_dims = (total_data.shape[0], 1) + tuple(total_data.shape[1:])
    total_data = total_data.reshape(total_dims)

    # deal with the no-poison cases
    if (poison_indexes.size == 0):
        return total_data, total_labels, np.array([]), np.array([])

    # sane cases
    return total_data[clean_indexes], total_labels[clean_indexes], \
            total_data[poison_indexes], total_labels[poison_indexes]


# ------------------------------------------------------------
#  Misc. function
# ------------------------------------------------------------
def store_updates_to_csvfile(filename, data):
    with open(filename, 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for each in data:
            csv_writer.writerow([each])
    # done.


"""
    Main
"""
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    #   Arguments for this script: command line compatibility
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser( \
        description='Analyze the gradients when there is feature collison.')

    # load arguments (use -es to fit the # of characters)
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        help='the name of a dataset (default: fashion_mnist)')
    parser.add_argument('--datapth', type=str, default='...',
                        help='the location of a dataset (default: ...)')

    # model parameters
    parser.add_argument('--network', type=str, default='convnet',
                        help='the name of a network (default: simple)')
    parser.add_argument('--netbase', type=str, default='',
                        help='the location of baseline model (default: ...)')

    # interpolation ratio
    parser.add_argument('--imode', type=str, default='one',
                        help='interpolation mode (one or multi, based on the # poisons)')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='interpolation ratio between the two samples (default: 0.0)')

    # load arguments
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))


    # ------------------------------------------------------------
    #  Tensorflow configurations
    # ------------------------------------------------------------
    # control tensorflow info. level
    set_verbosity(tf.compat.v1.logging.ERROR)

    # enable eager execution
    tf.enable_eager_execution()


    # ------------------------------------------------------------
    #  Load the baseline model
    # ------------------------------------------------------------
    # extract the basic information from the baseline model (always vanilla)
    net_tokens = args.netbase.split('/')
    if 'subtask' == args.dataset:
        # : subtask case
        net_tokens = net_tokens[3].split('_')
    else:
        # : fashion_mnist/cifar10
        net_tokens = net_tokens[2].split('_')

    # model parameters
    batch_size = int(net_tokens[2])
    epochs     = int(net_tokens[3])
    epochs     = 40 if epochs > 40 else epochs
    learn_rate = float(net_tokens[4])

    # error case
    if 'dp_' in args.netbase:
        assert False, ('Error: Baseline accuracy cannot come from a DP-model.')

    # load the model
    baseline_vars  = models.extract_tf_model_parameters(args.network, args.netbase)
    baseline_model = models.load_model( \
        args.dataset, args.datapth, args.network, vars=baseline_vars)
    print (' : Load the baseline model [{}] from [{}]'.format(args.network, args.netbase))


    # ------------------------------------------------------------
    #  Load the dataset (Data + Poisons)
    # ------------------------------------------------------------
    # load the dataset
    (x_train, y_train), (x_test, y_test) = \
        datasets.define_dataset(args.dataset, args.datapth)

    # bound check for the inputs (to compare the results with DP-training)
    assert (x_train.min() >= 0.) and (x_train.max() <= 1.) \
        and (x_test.min() >= 0.) and (x_test.max() <= 1.)

    # create an interpolated sample from two samples and a ratio
    (x_inter, y_inter) = _do_interpolation( \
        x_train, y_train, _dataindex[args.dataset], args.imode, args.alpha)

    # convert the data into float32/int32
    x_train = x_train.astype('float32')
    y_train = y_train.astype('int32')
    x_test  = x_test.astype('float32')
    y_test  = y_test.astype('int32')
    x_inter = x_inter.astype('float32')
    y_inter = y_inter.astype('int32')

    # [DEBUG]
    print (' : Construct the analysis data')
    print ('   Train : {} in [{}, {}]'.format(x_train.shape, x_train.min(), x_train.max()))
    print ('   Test  : {} in [{}, {}]'.format(x_test.shape, x_test.min(), x_test.max()))
    print ('   Interp: {} in [{}, {}]'.format(x_inter.shape, x_inter.min(), x_inter.max()))

    # compose into the tensorflow datasets
    clean_validset = datasets.convert_to_tf_dataset(x_test, y_test)

    # load the baseline acc
    baseline_acc = _validate(baseline_model, clean_validset)
    print (' : Baseline accuracy is [{}]'.format(baseline_acc))


    # --------------------------------------------------------------------------
    #   Substitute the numpy module used by JAX (when privacy)
    # --------------------------------------------------------------------------
    import jax.numpy as np


    # --------------------------------------------------------------------------
    #   Set the location to store...
    # --------------------------------------------------------------------------

    # extract the setup
    if 'one' == args.imode:
        current_task = 'a_pair_{}'.format( \
            '_'.join(map(str, _dataindex[args.dataset]['one'].values())))
    elif 'multi' == args.imode:
        current_task = 'pairs_of_{}'.format(len(_dataindex[args.dataset]['multi'][0]))
    else:
        assert False, ('Error: unknown mode - {}'.format(args.imode))

    # extract the current data
    if 'subtask' == args.dataset:
        current_data = args.datapth.split('/')[-1].replace('.pkl', '')
    else:
        current_data = args.dataset

    # compose
    store_base  = os.path.join( \
        'results', 'analysis', 'collison', \
        current_task, current_data, 'alpha_{}'.format(args.alpha))

    # fix store locations for each
    netname_pfix = 'vanilla_{}_{}_{}_{}'.format( \
            args.network, batch_size, epochs, learn_rate)

    results_model = os.path.join(store_base, netname_pfix)
    if not os.path.exists(results_model): os.makedirs(results_model)
    results_update= os.path.join(results_model, 'param_updates')
    if not os.path.exists(results_update): os.makedirs(results_update)
    results_data  = os.path.join(results_model, 'analysis_results.csv')

    # [DEBUG]
    print (' : Store locations are:')
    print ('  - Model folder : {}'.format(results_model))
    print ('  - Updates file : {}'.format(results_update))
    print ('  - Analysis data: {}'.format(results_data))


    # --------------------------------------------------------------------------
    #   Store the interpolated data
    # --------------------------------------------------------------------------
    if 'one' == args.imode:
        io.store_to_image( \
            os.path.join(results_model, 'base_0.png'), \
            x_train[_dataindex[args.dataset]['one'][0]].reshape(1, 28, 28), format='L')
        io.store_to_image( \
            os.path.join(results_model, 'base_1.png'), \
            x_train[_dataindex[args.dataset]['one'][1]].reshape(1, 28, 28), format='L')
        io.store_to_image( \
            os.path.join(results_model, 'interpolated.png'), \
            x_inter[0].reshape(1, 28, 28), format='L')
        print (' : Store the interpolated images to: {}'.format(results_model))


    # --------------------------------------------------------------------------
    #   Compose the poison dataset
    # --------------------------------------------------------------------------
    # total classes
    tot_cls = len(set(y_train))

    # convert the class information as one-hot vectors
    y_train = _convert_to_onehot(y_train, total=tot_cls)
    y_test  = _convert_to_onehot(y_test,  total=tot_cls)
    y_inter = _convert_to_onehot(y_inter, total=tot_cls)
    print (' : Labels converted to one-hot vectors - Y-train: {}'.format(y_train.shape))

    x_total = np.concatenate((x_train, x_inter), axis=0)
    y_total = np.concatenate((y_train, y_inter), axis=0)
    poison_trainsize= x_total.shape[0]
    poison_ncbatch, leftover = divmod(poison_trainsize, batch_size)
    poison_numbatch = poison_ncbatch + bool(leftover)
    poison_trainset = _data_loader( \
        x_total, y_total, batch_size, poison_numbatch)
    print (' : Insert the interpolated data into JAX datasets')


    # --------------------------------------------------------------------------
    #   Load the new model
    # --------------------------------------------------------------------------
    del baseline_model

    # initialize sequence for JAX
    prand_keys   = random.PRNGKey(_rand_fix)
    poison_lrate = learn_rate

    # init a JAX model
    if 'lr' == args.network:
        fn_pmodel_init, fn_pmodel_apply = LinearRegressionJAX(tot_cls)
    elif 'shallow-mlp' == args.network:
        fn_pmodel_init, fn_pmodel_apply = ShallowMLPJAX(256, tot_cls)
    else:
        assert False, ('Error: undefined network - {}'.format(args.network))
    if not _fn_holder: _fn_holder = fn_pmodel_apply

    # init parameters
    _, pminit_params = fn_pmodel_init(prand_keys, (-1, 28, 28, 1))

    # prepare the optimizer
    if 'lr' == args.network:
        fn_optim_init, fn_optim_update, fn_load_params = optimizers.adam(learn_rate)
    elif 'shallow-mlp' == args.network:
        fn_optim_init, fn_optim_update, fn_load_params = optimizers.sgd(learn_rate)
    else:
        assert False, ('Error: undefined network {} (optim error)'.format(args.network))
    optim_state = fn_optim_init(pminit_params)
    optim_count = itertools.count()
    print (' : Load a model [{}]'.format(args.network))


    # --------------------------------------------------------------------------
    #   Run in the inspection mode
    # --------------------------------------------------------------------------
    # data holder
    attack_results = []

    # compute how many updates happened
    total_cupdates = 0
    total_pupdates = 0

    # do training
    steps_per_epoch = poison_trainsize // batch_size
    for epoch in range(1, epochs+1):

        # : train the model for an epoch
        for mbatch in range(poison_numbatch):
            data, labels  = _shape_data(*next(poison_trainset), dummy_dim=True)

            """
                Dummy: this procedure is only for computing gradients
            """
            # :: data holder for the parameter updates
            clean_updates  = []
            poison_updates = []

            # :: check this batch includes the poisons or not.
            clean_data, clean_labels, poison_data, poison_labels = \
                _split_poisons_JAX(x_inter, y_inter, data, labels, verbose=_verbose)

            # :: check this batch includes the poisons or not.
            if _verbose:
                print (' :: The batch [{}] includes [{}] poisons...'.format(mbatch, len(poison_data)))

            # :: load the parameters and random number
            pmodel_params  = fn_load_params(optim_state)

            # :: [Poison] compute the gradient with the poisoned data
            if len(poison_data) != 0:

                # ::: increase the total updates
                total_pupdates += 1

                # ::: compute the gradients
                poison_gradient = grad(_loss)( \
                    pmodel_params, (poison_data, poison_labels))

                # ::: store the poison updates
                if not poison_updates:
                    for each_gradient in poison_gradient[len(poison_gradient)-1]:
                        cur_poison_ups = each_gradient
                        poison_updates.append(cur_poison_ups)
                else:
                    for gvidx, each_gradient in enumerate( \
                        poison_gradient[len(poison_gradient)-1]):
                        cur_poison_ups = each_gradient
                        poison_updates[gvidx] += cur_poison_ups

            # :: end if len(poison...)

            # :: increase the total updates
            total_cupdates += 1

            # :: compute the gradients
            clean_gradient = grad(_loss)( \
                pmodel_params, (clean_data, clean_labels))

            # :: store the clean updates
            if not clean_updates:
                for each_gradient in clean_gradient[len(clean_gradient)-1]:
                    cur_clean_ups = each_gradient
                    clean_updates.append(cur_clean_ups)
            else:
                for gvidx, each_gradient in enumerate( \
                    clean_gradient[len(clean_gradient)-1]):
                    cur_clean_ups = each_gradient
                    clean_updates[gvidx] += cur_clean_ups


            """
                Real procedure for optimizing the parameters
            """
            # :: compute gradients with DP-SGD
            pmodel_params  = fn_load_params(optim_state)
            current_count  = next(optim_count)
            current_random = random.fold_in(prand_keys, current_count)
            optim_state    = fn_optim_update(
                current_count, grad(_loss)(pmodel_params, (data, labels)), optim_state)


            """
                Save the updates in this epoch and batch to dir
            """
            # :: [Cleans] loop over the parameters (0th kernel, 1st bias, ...)
            if clean_updates:
                for uidx, updates in enumerate(clean_updates):
                    update_clfile  = os.path.join( \
                        results_update, '{}_{}_clean_{}.csv'.format(epoch, mbatch, uidx))
                    flatten_update = updates.flatten()
                    store_updates_to_csvfile(update_clfile, flatten_update)
                    print (' :: Store the [{}] update to [{}]'.format(uidx, update_clfile))

            # :: [Poisons] loop over the parameters (0th kernel, 1st bias, ...)
            if poison_updates:
                for uidx, updates in enumerate(poison_updates):
                    update_pofile  = os.path.join( \
                        results_update, '{}_{}_poison_{}.csv'.format(epoch, mbatch, uidx))
                    flatten_update = updates.flatten()
                    store_updates_to_csvfile(update_pofile, flatten_update)
                    print (' :: Store the [{}] update to [{}]'.format(uidx, update_pofile))

            # :: cleanup the data-holders
            clean_updates, poison_updates = [], []

        # : end for mbatch ...

        # : evaluate the test time accuracy
        pmodel_params = fn_load_params(optim_state)
        current_acc   = _validate_JAX(pmodel_params, fn_pmodel_apply, x_test, y_test)

        # : report the current state (cannot compute the total eps, as we split the ....)
        print (' : Epoch {} - acc {:.4f} (base) / {:.4f} (curr)'.format( \
            epoch, baseline_acc, current_acc))

        # : store the attack result
        attack_results.append([epoch, x_inter.shape[0], baseline_acc, current_acc])

        # : flush the stdouts
        sys.stdout.flush()

        # : info
        print (' : Poison {}, Clean {}'.format(total_pupdates, total_cupdates))

    # end for epoch...

    # report the attack results...
    print (' : [Result] epoch {}, alpha {}, base {:.4f}, curr {:.4f}'.format( \
        epoch, args.alpha, baseline_acc, current_acc))

    # store the attack results
    io.store_to_csv(results_data, attack_results)

    # finally
    print (' : Done, don\'t store the model')
    # done.
