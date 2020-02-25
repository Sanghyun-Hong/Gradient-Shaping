"""
    A script that examine the gradient-level signatures.
"""
import csv, os, sys
# suppress tensorflow errors -- too many, what's the purpose?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import argparse
import itertools
import numpy as np

# JAX models (for privacy analysis)
from jax import grad, partial, random, tree_util, vmap, device_put
from jax.lax import stop_gradient
from jax.experimental import optimizers, stax
from networks.linears import LinearRegressionJAX

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
_verbose  = False
_fn_holder= None


# ------------------------------------------------------------
#  Valid
# ------------------------------------------------------------
def _validate(model, dataset):
    corrects = []
    for (_, (data, labels)) in enumerate(dataset.take(-1)):
        logits, penultimate = model(data, training=False)
        predicts = tf.argmax(logits, axis=1)
        predicts = tf.dtypes.cast(predicts, tf.int32)
        corrects.append(tf.equal(predicts, labels).numpy())
    cur_acc = np.mean(corrects)
    return cur_acc

def _check_attack_success(params, applyfn, data, oracle):
    # sanity check...
    assert (len(data) == 1), \
        ('Error: only for a target, but wtf {}'.format(data.shape))
    predict = applyfn(params, data)
    predict = np.argmax(predict, axis=1)[0]
    print (' :: [{}] - predict, [{}] - target'.format(predict, oracle))
    return (predict == oracle)


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

def _define_optimizer_JAX(dataset, learn_rate):
    if 'purchases' == dataset:
        return optimizers.adam(learn_rate)
        # return optimizers.sgd(learn_rate)       # temporarily error....
    elif 'fashion_mnist' == dataset:
        return optimizers.sgd(learn_rate)
    elif 'cifar10' == dataset:
        return optimizers.sgd(learn_rate)
    else:
        assert False, ('Error: unknown dataset - {}'.format(dataset))

def _shape_data(data, labels, dummy_dim=False):
    if len(data.shape) == 2:
        orig_shape = (-1, 1, 100) if dummy_dim else (-1, 100)
    elif len(data.shape) == 4:
        orig_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
    else:
        assert False, ('Error: unexpected dimensions - {}'.format(data.shape))
    return np.reshape(data, orig_shape), labels

def _convert_to_onehot(labels, total=100):
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

def _pminit_w_baseline(pminit_params, baseline_vars, dataset, network):
    if 'purchases' == dataset:
        if 'lr' == network:
            pminit_params = [()]
            pminit_params.append(( \
                device_put(baseline_vars['linear_regression/dense/kernel:0']),
                device_put(baseline_vars['linear_regression/dense/bias:0']),
            ))
            return (pminit_params)
        else:
            assert False, ('Error: unknown network {} for {}'.format(network, dataset))
    else:
        assert False, ('Error: unknown dataset - {}'.format(dataset))
    # done.


# ------------------------------------------------------------
#  Misc. function
# ------------------------------------------------------------
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
#  Misc. functions
# ------------------------------------------------------------
def store_updates_to_csvfile(filename, data):
    with open(filename, 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for each in data:
            csv_writer.writerow([each])
    # done.


"""
    Main: to analyze the gradients from poisons and cleans
"""
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    #   Arguments for this script: command line compatibility
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser( \
        description='Analyze individual gradients in targeted attacks.')

    # load arguments (use -es to fit the # of characters)
    parser.add_argument('--dataset', type=str, default='synthetic-moons',
                        help='the name of a dataset (default: synthetic-moons)')
    parser.add_argument('--datapth', type=str, default='...',
                        help='the location of a dataset (default: ...)')
    parser.add_argument('--poisond', type=str, default='...',
                        help='the dir. where the poisons are')
    parser.add_argument('--samples', type=str, default='...',
                        help='the list of target indexes sampled (default: ...)')
    parser.add_argument('--poisonn', type=int, default=-1,
                        help='the max. number of poisons to use (default: -1)')
    parser.add_argument('--b-class', type=int, default=3,
                        help='the class that an attacker wants (default: 3)')
    parser.add_argument('--t-class', type=int, default=4,
                        help='the class that an attacker wants (default: 4)')

    # load models
    parser.add_argument('--network', type=str, default='simple',
                        help='the name of the network (ex. simple)')
    parser.add_argument('--netpath', type=str, default='...',
                        help='the location where the model is stored (ex. ..)')
    parser.add_argument('--privacy', action='store_true', default=False,
                        help='set when the network is trained with DP')

    # load the target and the poison indexes to analyze
    parser.add_argument('--t-index', type=int, default=-1,
                        help='the index of the target (default: -1)')
    parser.add_argument('--p-index', type=int, default=-1,
                        help='the index of the poison (default: -1)')

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


    # --------------------------------------------------------------------------
    #   Load the trained model
    # --------------------------------------------------------------------------
    # extract the basic information from the baseline model (always vanilla)
    net_tokens = args.netpath.split('/')
    net_tokens = net_tokens[2].split('_')

    # model parameters
    batch_size = int(net_tokens[2])
    epochs     = int(net_tokens[3])
    learn_rate = float(net_tokens[4])

    # privacy parameters
    if args.privacy:
        epsilon = float(net_tokens[5])
        delta   = float(net_tokens[6])
        nclip   = float(net_tokens[7])
        noise   = float(net_tokens[8])

    # load the model
    baseline_vars  = models.extract_tf_model_parameters(args.network, args.netpath)
    baseline_model = models.load_model( \
        args.dataset, args.datapth, args.network, vars=baseline_vars)
    print (' : Load the baseline model [{}] from [{}]'.format(args.network, args.netpath))


    # ------------------------------------------------------------
    #  Load the dataset (Data + Poisons)
    # ------------------------------------------------------------

    # load the clean dataset
    (x_train, y_train), (x_test, y_test) = \
        datasets.define_dataset(args.dataset, args.datapth)
    print (' : Load the dataset [{}] from [{}]'.format(args.dataset, args.datapth))

    # compose the filename to load
    poison_file = os.path.join( \
        args.poisond, 'poisons_for_{}.pkl'.format(args.t_index))
    print (' : Load the poisons from [{}]'.format(poison_file))

    # load the attack poison / target
    (x_poison, y_poison), (x_target, y_target) = \
        datasets.load_poisons(poison_file, x_test, y_test, sort=True)
    (x_poison, y_poison) = \
        (x_poison[(args.p_index-1):args.p_index], \
         y_poison[(args.p_index-1):args.p_index])
    print (' : Pick the target [{}], using a poison [{}]'.format(args.t_index, args.p_index))

    # compose into the tensorflow datasets
    clean_validset = datasets.convert_to_tf_dataset(x_test, y_test)
    target_dataset = datasets.convert_to_tf_dataset(x_target, y_target)

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
    poison_task = args.poisond.split('/')[3]
    poison_data = args.poisond.split('/')[4]

    # : compose
    store_base  = os.path.join( \
        'results', 'analysis', 'tpoisoning', poison_task, poison_data)

    # fix store locations for each
    if not args.privacy:
        netname_pfix = 'vanilla_{}_{}_{}_{}'.format( \
                args.network, batch_size, epochs, learn_rate)
    else:
        netname_pfix = 'dp_{}_{}_{}_{}_{}_{}_{}_{}'.format( \
            args.network, batch_size, epochs, learn_rate, \
            epsilon, delta, nclip, noise)

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
    # total classes
    tot_cls = len(set(y_train))

    # convert the class information as one-hot vectors
    y_train = _convert_to_onehot(y_train,  total=tot_cls)
    y_test  = _convert_to_onehot(y_test,   total=tot_cls)
    y_poison= _convert_to_onehot(y_poison, total=tot_cls)
    y_target= _convert_to_onehot(y_target, total=tot_cls)
    print (' : Labels converted to one-hot vectors - Y-train: {}'.format(y_train.shape))

    x_total = np.concatenate((x_train, x_poison), axis=0)
    y_total = np.concatenate((y_train, y_poison), axis=0)
    poison_trainsize= x_total.shape[0]
    poison_ncbatch, leftover = divmod(poison_trainsize, batch_size)
    poison_numbatch = poison_ncbatch + bool(leftover)
    poison_trainset = _data_loader( \
        x_total, y_total, batch_size, poison_numbatch)
    print (' : Convert the label-flipped dataset into JAX datasets')


    # --------------------------------------------------------------------------
    #   Prepare for re-training...
    # --------------------------------------------------------------------------
    # define the re-training epochs
    if 'purchases' == args.dataset:
        poison_epochs = 40
    else:
        poison_epochs = 20 if (epochs > 20) else (epochs // 2)

    # initialize sequence for JAX
    prand_keys   = random.PRNGKey(_rand_fix)
    poison_lrate = learn_rate

    # init a JAX model
    fn_pmodel_init, fn_pmodel_apply = LinearRegressionJAX(tot_cls)
    if not _fn_holder: _fn_holder = fn_pmodel_apply

    # init parameters
    pmodel_indims    = (-1,) + tuple(x_train.shape[1:])
    _, pminit_params = fn_pmodel_init(prand_keys, pmodel_indims)

    # init parameter [insert the baseline model's params]
    pminit_params = _pminit_w_baseline( \
        pminit_params, baseline_vars, args.dataset, args.network)

    # prepare the optimizer
    fn_optim_init, fn_optim_update, fn_load_params = \
        _define_optimizer_JAX(args.dataset, learn_rate)
    optim_state = fn_optim_init(pminit_params)
    optim_count = itertools.count()
    print (' : Load a model trained with poisons')


    # --------------------------------------------------------------------------
    #   Run in the inspection mode
    # --------------------------------------------------------------------------

    # data holder
    attack_results = []

    # a flag that decides to store or not
    attack_success = False

    # do training
    steps_per_epoch = poison_trainsize // batch_size
    for epoch in range(1, poison_epochs+1):

        # ----------------------------------------------------------------------
        #  : No privacy
        # ----------------------------------------------------------------------
        if not args.privacy:

            # :: data hold per epoch
            clean_updates  = []
            poison_updates = []

            # :: train the model for an epoch
            for mbatch in range(poison_numbatch):
                data, labels  = _shape_data(*next(poison_trainset), dummy_dim=True)

                """
                    Dummy: this procedure is only for computing gradients
                """
                # ::: check this batch includes the poisons or not.
                clean_data, clean_labels, poison_data, poison_labels = \
                    _split_poisons_JAX(x_poison, y_poison, data, labels, verbose=_verbose)

                # ::: check this batch includes the poisons or not.
                if _verbose:
                    print (' :: The batch [{}] includes [{}] poisons...'.format(mbatch, len(poison_data)))

                # ::: load the parameters and random number
                pmodel_params  = fn_load_params(optim_state)
                current_count  = next(optim_count)
                current_random = random.fold_in(prand_keys, current_count)

                # ::: [Poison] compute the gradient with the poisoned data
                if len(poison_data) != 0:

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

            # :: end for mbatch ...

            # :: evaluate the test time accuracy
            pmodel_params = fn_load_params(optim_state)
            current_acc   = _validate_JAX(pmodel_params, fn_pmodel_apply, x_test, y_test)
            attack_success= _check_attack_success( \
                pmodel_params, fn_pmodel_apply, x_target, args.t_class)

            # : report the current state (cannot compute the total eps, as we split the ....)
            print (' : Epoch {} - acc {:.4f} (base) / {:.4f} (curr) [{}]'.format( \
                epoch, baseline_acc, current_acc, attack_success))

            # : store the current attack results
            attack_results.append([ \
                epoch, \
                args.b_class, args.t_class, args.t_index, args.p_index, \
                baseline_acc, current_acc, attack_success])

            # : flush the stdouts
            sys.stdout.flush()


            """
                Save the updates in this epoch and batch to dir
            """
            # ::: [Cleans] loop over the parameters (0th kernel, 1st bias, ...)
            if clean_updates:
                for uidx, updates in enumerate(clean_updates):
                    update_clfile  = os.path.join( \
                        results_update, '{}_clean_{}.csv'.format(epoch, uidx))
                    flatten_update = updates.flatten()
                    store_updates_to_csvfile(update_clfile, flatten_update)
                    if _verbose: print (' :: Store the [{}] update to [{}]'.format(uidx, update_clfile))

            # ::: [Poisons] loop over the parameters (0th kernel, 1st bias, ...)
            if poison_updates:
                for uidx, updates in enumerate(poison_updates):
                    update_pofile  = os.path.join( \
                        results_update, '{}_poison_{}.csv'.format(epoch, uidx))
                    flatten_update = updates.flatten()
                    store_updates_to_csvfile(update_pofile, flatten_update)
                    if _verbose: print (' :: Store the [{}] update to [{}]'.format(uidx, update_pofile))

            # : stop condition (poisoning was successful)
            if attack_success:
                print (' : Poisoning successful, store the updates'); break


        # ----------------------------------------------------------------------
        #  : With privacy
        # ----------------------------------------------------------------------
        else:

            # :: data hold per epoch
            clean_updates  = []
            poison_updates = []

            # :: train the model for an epoch
            for mbatch in range(poison_numbatch):
                data, labels  = _shape_data(*next(poison_trainset), dummy_dim=True)

                """
                    Dummy: this procedure is only for computing gradients
                """
                # ::: check this batch includes the poisons or not.
                clean_data, clean_labels, poison_data, poison_labels = \
                    _split_poisons_JAX(x_poison, y_poison, data, labels, verbose=_verbose)

                # ::: check this batch includes the poisons or not.
                if _verbose:
                    print (' :: The batch [{}] includes [{}] poisons...'.format(mbatch, len(poison_data)))

                # ::: load the parameters and random number
                pmodel_params  = fn_load_params(optim_state)
                current_count  = next(optim_count)
                current_random = random.fold_in(prand_keys, current_count)

                # ::: [Poison] compute the gradient with the poisoned data
                if len(poison_data) != 0:

                    # ::::: compute the gradients
                    poison_gradient = _dp_compute_gradients( \
                        pmodel_params, (poison_data, poison_labels), current_random,
                        nclip, noise, poison_labels.shape[0])

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

                # ::: compute the gradients
                clean_gradient = _dp_compute_gradients( \
                    pmodel_params, (clean_data, clean_labels), current_random,
                    nclip, noise, clean_labels.shape[0])

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
                        nclip, noise, batch_size),
                    optim_state)

            # :: end for mbatch ...

            # :: evaluate the test time accuracy
            pmodel_params = fn_load_params(optim_state)
            current_acc   = _validate_JAX(pmodel_params, fn_pmodel_apply, x_test, y_test)
            attack_success= _check_attack_success( \
                pmodel_params, fn_pmodel_apply, x_target, args.t_class)

            # : report the current state (cannot compute the total eps, as we split the ....)
            print (' : Epoch {} - acc {:.4f} (base) / {:.4f} (curr) [{}]'.format( \
                epoch, baseline_acc, current_acc, attack_success))

            # : store the current attack results
            attack_results.append([ \
                epoch, \
                args.b_class, args.t_class, args.t_index, args.p_index, \
                baseline_acc, current_acc, attack_success])

            # : flush the stdouts
            sys.stdout.flush()


            """
                Save the updates in this epoch and batch to dir
            """
            # ::: [Cleans] loop over the parameters (0th kernel, 1st bias, ...)
            if clean_updates:
                for uidx, updates in enumerate(clean_updates):
                    update_clfile  = os.path.join( \
                        results_update, '{}_clean_{}.csv'.format(epoch, uidx))
                    flatten_update = updates.flatten()
                    store_updates_to_csvfile(update_clfile, flatten_update)
                    if _verbose: print (' :: Store the [{}] update to [{}]'.format(uidx, update_clfile))

            # ::: [Poisons] loop over the parameters (0th kernel, 1st bias, ...)
            if poison_updates:
                for uidx, updates in enumerate(poison_updates):
                    update_pofile  = os.path.join( \
                        results_update, '{}_poison_{}.csv'.format(epoch, uidx))
                    flatten_update = updates.flatten()
                    store_updates_to_csvfile(update_pofile, flatten_update)
                    if _verbose: print (' :: Store the [{}] update to [{}]'.format(uidx, update_pofile))

            # : stop condition (poisoning was successful)
            if attack_success:
                print (' : Poisoning successful, store the updates'); break

    # end for epoch...

    # report the attack results
    print (' : [Result:{}] epoch {}, poison {}, base {:.4f}, curr {:.4f}'.format( \
        attack_success, epoch, x_poison.shape[0], baseline_acc, current_acc))

    # store the attack result
    attack_results.append([ \
        epoch, \
        args.b_class, args.t_class, args.t_index, args.p_index, \
        baseline_acc, current_acc, attack_success])
    io.store_to_csv(results_data, attack_results)
    # done.
