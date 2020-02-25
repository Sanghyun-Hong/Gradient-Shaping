"""
    Do indiscriminate poisoning (w. Eager Execution of TF)
"""
import csv, os, sys
# suppress tensorflow errors -- too many, who's the developer?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import time
import pickle
import argparse
import itertools
import numpy as np
from tqdm import tqdm

# JAX models (for privacy analysis)
from jax import grad, partial, random, tree_util, vmap
from jax.lax import stop_gradient
from jax.experimental import optimizers, stax
from networks.linears import LinearRegressionJAX

# tensorflow modules
import tensorflow as tf
from tensorflow.compat.v1.logging import set_verbosity, ERROR
from tensorflow.compat.v1.estimator.inputs import numpy_input_fn
from tensorflow.compat.v1.train import GradientDescentOptimizer, AdamOptimizer

# tensorflow-privacy (since we use the bleeding-edge version)
try:
    from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
    from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
    from tensorflow_privacy.privacy.optimizers import dp_optimizer
except:
    from privacy.analysis.rdp_accountant import compute_rdp
    from privacy.analysis.rdp_accountant import get_privacy_spent
    from privacy.optimizers import dp_optimizer

# custom libs
from utils import io
from utils import datasets, models, optims


# ------------------------------------------------------------
#  Global variables
# ------------------------------------------------------------
_rand_fix = 215
_verbose  = True
_fn_holder= None


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

def _convert_to_onehot(labels):
    # use the original numpy functions
    from numpy import zeros as nzeros
    from numpy import arange as narange
    # to one-hot
    new_labels = nzeros((labels.size, labels.max()+1))
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

def _dp_compute_gradients(params, batch, rng, norm_clip, noise_level, batch_size):
    """
        Return differentially private gradients for params, evaluated on batch
    """
    def _clipped_grad(params, single_example_batch):
        # Evaluate gradient for a single-example batch and clip its grad norm
        grads = grad(_loss)(params, single_example_batch)

        nonempty_grads, tree_def = tree_util.tree_flatten(grads)
        total_grad_norm = np.linalg.norm( \
            [np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
        divisor = stop_gradient(np.amax((total_grad_norm / norm_clip, 1.)))
        normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
        return tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)

    px_clipped_grad_fn = vmap(partial(_clipped_grad, params))
    std_dev = norm_clip * noise_level
    noise_ = lambda n: n + std_dev * random.normal(rng, n.shape)
    normalize_ = lambda n: n / float(batch_size)
    tree_map = tree_util.tree_map
    sum_ = lambda n: np.sum(n, 0)  # aggregate
    aggregated_clipped_grads = tree_map(sum_, px_clipped_grad_fn((data, labels)))
    noised_aggregated_clipped_grads = tree_map(noise_, aggregated_clipped_grads)
    normalized_noised_aggregated_clipped_grads = (
        tree_map(normalize_, noised_aggregated_clipped_grads)
    )
    return normalized_noised_aggregated_clipped_grads

def _split_poisons_lflip_JAX( \
    poison_data, poison_labels, total_data, total_labels, verbose=False):
    """
        Identify whether the batch includes poisons
    """
    # reduce one dimension from the total data
    total_data = total_data.reshape( \
        total_data.shape[0], total_data.shape[2], \
        total_data.shape[3], total_data.shape[4])

    # data-holder
    poison_indexes = []

    # iterate over the total data, and see if any data is in poisons
    for pidx, each_poison in enumerate(poison_data):
        search_result = (each_poison == total_data).all((1, 2, 3))
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
    total_data = total_data.reshape( \
        total_data.shape[0], 1, \
        total_data.shape[1], total_data.shape[2], total_data.shape[3])

    # deal with the no-poison cases
    if (poison_indexes.size == 0):
        return total_data, total_labels, np.array([]), np.array([])

    # sane cases
    return total_data[clean_indexes], total_labels[clean_indexes], \
            total_data[poison_indexes], total_labels[poison_indexes]

def _split_poisons_slab_JAX( \
    poison_data, poison_labels, total_data, total_labels, verbose=False):
    """
        Identify whether the batch includes poisons
    """
    # reduce one dimension from the total data
    total_data = total_data.reshape( \
        total_data.shape[0], total_data.shape[2], \
        total_data.shape[3], total_data.shape[4])

    # data-holder
    poison_indexes = []

    # iterate over the total data, and see if any data is in poisons
    for pidx, each_poison in enumerate(poison_data):
        search_result = (each_poison == total_data).all((1, 2, 3))
        search_tindex = [i for i, tfval in enumerate(search_result) if tfval]
        # : skip, if the index is the same
        if not search_tindex: continue
        # : check the same label one, which is not included
        for each_tindex in search_tindex:
            # :: already in...
            if (each_tindex in poison_indexes): continue
            # :: include when the label is the same
            if (poison_labels[pidx] == total_labels[each_tindex]).any():
                poison_indexes.append(each_tindex)

    # split into two ...
    poison_indexes = np.array(poison_indexes)
    clean_indexes  = np.array([ \
        didx for didx in range(len(total_data)) if didx not in poison_indexes])

    # expand the data back
    total_data = total_data.reshape( \
        total_data.shape[0], 1, \
        total_data.shape[1], total_data.shape[2], total_data.shape[3])

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
    Main: to select the target and the poisons
"""
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    #   Arguments for this script: command line compatibility
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser( \
        description='Analyze individual gradients in indiscriminate attacks.')

    # load arguments (use -es to fit the # of characters)
    parser.add_argument('--dataset', type=str, default='subtask',
                        help='the name of a dataset (default: subtask)')
    parser.add_argument('--datapth', type=str, default='...',
                        help='the location of a dataset (default: ...)')
    parser.add_argument('--poisonp', type=str, default='...',
                        help='the location of a poison data (default: ...)')

    # model parameters
    parser.add_argument('--network', type=str, default='lr',
                        help='the name of a network (default: simple)')
    parser.add_argument('--netbase', type=str, default='',
                        help='the location of baseline model (default: ...)')
    parser.add_argument('--privacy', action='store_true',
                        help='set the privacy when it is in use')

    # privacy-parameters
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='epsilon as a privacy budget (default: 0.0)')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='delta as a privacy guarantee (default: 0.0)')
    parser.add_argument('--nclip', type=float, default=0.0,
                        help='l2 value for clipping the norm (default: 0.0)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='noise-level that adds to queries - sigma (default: 0.0)')

    # load...
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
    if args.poisonp.endswith('.pkl'):
        (x_train, y_train), (x_test, y_test), (x_poison, y_poison) = \
            datasets.load_lfip_poisons(args.poisonp)
    elif args.poisonp.endswith('.mat'):
        (x_train, y_train), (x_test, y_test), (x_poison, y_poison) = \
            datasets.load_slab_poisons(args.poisonp)
    else:
        assert False, ('Error: unknown format file - {}'.format(args.poisonp))

    # preprocess the fmnist 3/4 dataset
    # (change the shapes for the analysis code)
    if ('fmnist_34' in args.poisonp) \
        and (args.poisonp.endswith('.mat')):
        # convert the shapes
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test  = x_test.reshape((x_test.shape[0], 28, 28, 1))
        x_poison= x_poison.reshape((x_poison.shape[0], 28, 28, 1))

    # convert the data into float32/int32
    x_train  = x_train.astype('float32')
    y_train  = y_train.astype('int32')
    x_test   = x_test.astype('float32')
    y_test   = y_test.astype('int32')
    x_poison = x_poison.astype('float32')
    y_poison = y_poison.astype('int32')

    # enforce the poisons to be within [0, 1] range
    x_poison = np.clip(x_poison, 0., 1.)

    # [DEBUG]
    print (' : Load the poison data from [{}]'.format(args.poisonp))
    print ('   Train : {} in [{}, {}]'.format(x_train.shape, x_train.min(), x_train.max()))
    print ('   Test  : {} in [{}, {}]'.format(x_test.shape, x_test.min(), x_test.max()))
    print ('   Poison: {} in [{}, {}]'.format(x_poison.shape, x_poison.min(), x_poison.max()))

    # compose into the tensorflow datasets
    clean_validset = datasets.convert_to_tf_dataset(x_test, y_test)

    # load the baseline acc
    baseline_acc = _validate(baseline_model, clean_validset)
    print (' : Baseline model\'s accuracy is [{}]'.format(baseline_acc))


    # --------------------------------------------------------------------------
    #   Substitute the numpy module used by JAX (when privacy)
    # --------------------------------------------------------------------------
    import jax.numpy as np


    # --------------------------------------------------------------------------
    #   Set the location to store...
    # --------------------------------------------------------------------------
    # extract the setup
    poison_task = args.poisonp.split('/')[3]
    poison_data = args.poisonp.split('/')[4].replace('.pkl', '')

    # : compose
    store_base  = os.path.join( \
        'results', 'analysis', 'ipoisoning', poison_task, poison_data)

    # fix store locations for each
    if not args.privacy:
        netname_pfix = 'vanilla_{}_{}_{}_{}'.format( \
                args.network, batch_size, epochs, learn_rate)
    else:
        netname_pfix = 'dp_{}_{}_{}_{}_{}_{}_{}_{}'.format( \
            args.network, batch_size, epochs, learn_rate, \
            args.epsilon, args.delta, args.nclip, args.noise)

    results_model = os.path.join(store_base, netname_pfix)
    if not os.path.exists(results_model): os.makedirs(results_model)
    results_update= os.path.join(results_model, 'param_updates')
    if not os.path.exists(results_update): os.makedirs(results_update)
    results_data  = os.path.join(results_model, 'attack_results.csv')

    # [DEBUG]
    print (' : Store locations are:')
    print ('  - Model folder: {}'.format(results_model))
    print ('  - Updates file: {}'.format(results_update))
    print ('  - Attack data : {}'.format(results_data))


    # --------------------------------------------------------------------------
    #   Compose the poison dataset
    # --------------------------------------------------------------------------
    # convert the class information as one-hot vectors
    y_train = _convert_to_onehot(y_train)
    y_test  = _convert_to_onehot(y_test)
    y_poison= _convert_to_onehot(y_poison)
    print (' : Labels converted to one-hot vectors - Y-train: {}'.format(y_train.shape))

    # compose the poisonsed training set
    x_total = np.concatenate((x_train, x_poison), axis=0)
    y_total = np.concatenate((y_train, y_poison), axis=0)
    poison_trainsize= x_total.shape[0]
    poison_ncbatch, leftover = divmod(poison_trainsize, batch_size)
    poison_numbatch = poison_ncbatch + bool(leftover)
    poison_trainset = _data_loader( \
        x_total, y_total, batch_size, poison_numbatch)
    print (' : Convert the label-flipped dataset into JAX datasets')


    # --------------------------------------------------------------------------
    #   Load the new model
    # --------------------------------------------------------------------------
    del baseline_model

    # initialize sequence for JAX
    prand_keys   = random.PRNGKey(_rand_fix)
    poison_lrate = learn_rate

    # init a JAX model
    fn_pmodel_init, fn_pmodel_apply = LinearRegressionJAX(2)
    if not _fn_holder: _fn_holder = fn_pmodel_apply

    # init parameters
    _, pminit_params = fn_pmodel_init(prand_keys, (-1, 28, 28, 1))

    # prepare the optimizer
    fn_optim_init, fn_optim_update, fn_load_params = optimizers.adam(learn_rate)
    optim_state = fn_optim_init(pminit_params)
    optim_count = itertools.count()
    print (' : Load a model that will be poisoned [{}, {}]'.format(args.nclip, args.noise))


    # --------------------------------------------------------------------------
    #   Run in the inspection mode
    # --------------------------------------------------------------------------

    # best accuracy holder
    best_at  = 0
    best_acc = 0.0

    # compute how many updates happened
    total_cupdates = 0
    total_pupdates = 0

    # do training
    steps_per_epoch = poison_trainsize // batch_size
    for epoch in range(1, epochs+1):

        # ----------------------------------------------------------------------
        #  : No privacy
        # ----------------------------------------------------------------------
        if not args.privacy:

            # :: train the model for an epoch
            for mbatch in range(poison_numbatch):
                data, labels  = _shape_data(*next(poison_trainset), dummy_dim=True)

                """
                    Dummy: this procedure is only for computing gradients
                """
                # ::: data holder for the parameter updates
                clean_updates  = []
                poison_updates = []

                # ::: check this batch includes the poisons or not.
                if 'label-flip' in args.poisonp:
                    clean_data, clean_labels, poison_data, poison_labels = \
                        _split_poisons_lflip_JAX(x_poison, y_poison, data, labels, verbose=_verbose)
                elif 'slab' in args.poisonp:
                    clean_data, clean_labels, poison_data, poison_labels = \
                        _split_poisons_slab_JAX(x_poison, y_poison, data, labels, verbose=_verbose)
                else:
                    assert False, ('Error: undefined indiscriminate attacks - {}'.format(args.poisonp))

                # ::: check this batch includes the poisons or not.
                if _verbose:
                    print (' :: The batch [{}] includes [{}] poisons...'.format(mbatch, len(poison_data)))

                # ::: load the parameters and random number
                pmodel_params  = fn_load_params(optim_state)

                # ::: [Poison] compute the gradient with the poisoned data
                if len(poison_data) != 0:

                    # ::::: increase the total updates
                    total_pupdates += 1

                    # ::::: compute the gradients
                    poison_gradient = grad(_loss)( \
                        pmodel_params, (poison_data, poison_labels))

                    # ::::: store the poison updates
                    if not poison_updates:
                        for each_gradient in poison_gradient[1]:
                            cur_poison_ups = each_gradient
                            poison_updates.append(cur_poison_ups)
                    else:
                        for gvidx, each_gradient in enumerate(poison_gradient[1]):
                            cur_poison_ups = each_gradient
                            poison_updates[gvidx] += cur_poison_ups

                # ::: end if len(poison...)

                # ::: increase the total updates
                total_cupdates += 1

                # ::: compute the gradients
                clean_gradient = grad(_loss)( \
                    pmodel_params, (clean_data, clean_labels))

                # ::: store the clean updates
                if not clean_updates:
                    for each_gradient in clean_gradient[1]:
                        cur_clean_ups = each_gradient
                        clean_updates.append(cur_clean_ups)
                else:
                    for gvidx, each_gradient in enumerate(clean_gradient[1]):
                        cur_clean_ups = each_gradient
                        clean_updates[gvidx] += cur_clean_ups


                """
                    Real procedure for optimizing the parameters
                """
                # ::: compute gradients with DP-SGD
                pmodel_params  = fn_load_params(optim_state)
                current_count  = next(optim_count)
                current_random = random.fold_in(prand_keys, current_count)
                optim_state    = fn_optim_update(
                    current_count, grad(_loss)(pmodel_params, (data, labels)), optim_state)

                """
                    Save the updates in this epoch and batch to dir
                """
                # ::: [Cleans] loop over the parameters (0th kernel, 1st bias, ...)
                if clean_updates:
                    for uidx, updates in enumerate(clean_updates):
                        update_clfile  = os.path.join( \
                            results_update, '{}_{}_clean_{}.csv'.format(epoch, mbatch, uidx))
                        flatten_update = updates.flatten()
                        store_updates_to_csvfile(update_clfile, flatten_update)
                        print (' :: Store the [{}] update to [{}]'.format(uidx, update_clfile))

                # ::: [Poisons] loop over the parameters (0th kernel, 1st bias, ...)
                if poison_updates:
                    for uidx, updates in enumerate(poison_updates):
                        update_pofile  = os.path.join( \
                            results_update, '{}_{}_poison_{}.csv'.format(epoch, mbatch, uidx))
                        flatten_update = updates.flatten()
                        store_updates_to_csvfile(update_pofile, flatten_update)
                        print (' :: Store the [{}] update to [{}]'.format(uidx, update_pofile))

                # ::: cleanup the data-holders
                clean_updates, poison_updates = [], []

            # :: end for mbatch ...

            # :: evaluate the test time accuracy
            pmodel_params = fn_load_params(optim_state)
            current_acc   = _validate_JAX(pmodel_params, fn_pmodel_apply, x_test, y_test)

            # :: record the best accuracy
            if best_acc < current_acc:
                best_at  = epoch
                best_acc = current_acc

            # :: report the current state (cannot compute the total eps, as we split the ....)
            print (' : Epoch {} - acc {:.4f} (base) / {:.4f} (curr) / {:.4f} (best @ {})'.format( \
                epoch, baseline_acc, current_acc, best_acc, best_at))

            # :: flush the stdouts
            sys.stdout.flush()

            # :: info
            print (' : Poison {}, Clean {}'.format(total_pupdates, total_cupdates))

        # ----------------------------------------------------------------------
        #  : With privacy
        # ----------------------------------------------------------------------
        else:

            # :: train the model for an epoch
            for mbatch in range(poison_numbatch):
                data, labels  = _shape_data(*next(poison_trainset), dummy_dim=True)

                """
                    Dummy: this procedure is only for computing gradients
                """
                # ::: data holder for the parameter updates
                clean_updates  = []
                poison_updates = []

                # ::: check this batch includes the poisons or not.
                if 'label-flip' in args.poisonp:
                    clean_data, clean_labels, poison_data, poison_labels = \
                        _split_poisons_lflip_JAX(x_poison, y_poison, data, labels, verbose=_verbose)
                elif 'slab' in args.poisonp:
                    clean_data, clean_labels, poison_data, poison_labels = \
                        _split_poisons_slab_JAX(x_poison, y_poison, data, labels, verbose=_verbose)
                else:
                    assert False, ('Error: undefined indiscriminate attacks - {}'.format(args.poisonp))

                # ::: check this batch includes the poisons or not.
                if _verbose:
                    print (' :: The batch [{}] includes [{}] poisons...'.format(mbatch, len(poison_data)))

                # ::: load the parameters and random number
                pmodel_params  = fn_load_params(optim_state)
                current_count  = next(optim_count)
                current_random = random.fold_in(prand_keys, current_count)

                # ::: [Poison] compute the gradient with the poisoned data
                if len(poison_data) != 0:

                    # ::::: increase the total updates
                    total_pupdates += 1

                    # ::::: compute the gradients
                    poison_gradient = _dp_compute_gradients( \
                        pmodel_params, (poison_data, poison_labels), current_random,
                        args.nclip, args.noise, poison_labels.shape[0])

                    # ::::: store the poison updates
                    if not poison_updates:
                        for each_gradient in poison_gradient[1]:
                            cur_poison_ups = each_gradient
                            poison_updates.append(cur_poison_ups)
                    else:
                        for gvidx, each_gradient in enumerate(poison_gradient[1]):
                            cur_poison_ups = each_gradient
                            poison_updates[gvidx] += cur_poison_ups

                # ::: end if len(poison...)

                # ::: increase the total updates
                total_cupdates += 1

                # ::: compute the gradients
                clean_gradient = _dp_compute_gradients( \
                    pmodel_params, (clean_data, clean_labels), current_random,
                    args.nclip, args.noise, clean_labels.shape[0])

                # ::: store the clean updates
                if not clean_updates:
                    for each_gradient in clean_gradient[1]:
                        cur_clean_ups = each_gradient
                        clean_updates.append(cur_clean_ups)
                else:
                    for gvidx, each_gradient in enumerate(clean_gradient[1]):
                        cur_clean_ups = each_gradient
                        clean_updates[gvidx] += cur_clean_ups


                """
                    Real procedure for optimizing the parameters
                """
                # ::: compute gradients with DP-SGD
                pmodel_params  = fn_load_params(optim_state)
                current_count  = next(optim_count)
                current_random = random.fold_in(prand_keys, current_count)
                optim_state    = fn_optim_update(
                    current_count,
                    _dp_compute_gradients(
                        pmodel_params, (data, labels), current_random,
                        args.nclip, args.noise, batch_size),
                    optim_state)

                """
                    Save the updates in this epoch and batch to dir
                """
                # ::: [Cleans] loop over the parameters (0th kernel, 1st bias, ...)
                if clean_updates:
                    for uidx, updates in enumerate(clean_updates):
                        update_clfile  = os.path.join( \
                            results_update, '{}_{}_clean_{}.csv'.format(epoch, mbatch, uidx))
                        flatten_update = updates.flatten()
                        store_updates_to_csvfile(update_clfile, flatten_update)
                        print (' :: Store the [{}] update to [{}]'.format(uidx, update_clfile))

                # ::: [Poisons] loop over the parameters (0th kernel, 1st bias, ...)
                if poison_updates:
                    for uidx, updates in enumerate(poison_updates):
                        update_pofile  = os.path.join( \
                            results_update, '{}_{}_poison_{}.csv'.format(epoch, mbatch, uidx))
                        flatten_update = updates.flatten()
                        store_updates_to_csvfile(update_pofile, flatten_update)
                        print (' :: Store the [{}] update to [{}]'.format(uidx, update_pofile))

                # ::: cleanup the data-holders
                clean_updates, poison_updates = [], []

            # :: end for mbatch ...

            # :: evaluate the test time accuracy
            pmodel_params = fn_load_params(optim_state)
            current_acc   = _validate_JAX(pmodel_params, fn_pmodel_apply, x_test, y_test)

            # :: record the best accuracy
            if best_acc < current_acc:
                best_at  = epoch
                best_acc = current_acc

            # :: report the current state (cannot compute the total eps, as we split the ....)
            print (' : Epoch {} - acc {:.4f} (base) / {:.4f} (curr) / {:.4f} (best @ {})'.format( \
                epoch, baseline_acc, current_acc, best_acc, best_at))

            # :: flush the stdouts
            sys.stdout.flush()

            # :: info
            print (' : Poison {}, Clean {}'.format(total_pupdates, total_cupdates))

    # end for epoch...

    # report the attack results...
    print (' : [Result] epoch {}, poison {}, base {:.4f}, best {:.4f} @ {}'.format( \
        epoch, x_poison.shape[0], baseline_acc, best_acc, best_at))

    # store the attack results
    attack_results = [[best_at, best_acc, baseline_acc, x_poison.shape[0]]]
    io.store_to_csv(results_data, attack_results)

    # finally
    print (' : Done, don\'t store the model')
    # done.
