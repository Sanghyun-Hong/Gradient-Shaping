"""
    Analyze feature insertion (w. Eager Execution of TF)
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
from networks.cnns import BadNetJAX

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


# ------------------------------------------------------------
#  Valiadation datasets
# ------------------------------------------------------------
def _validate(model, validset):
    corrects = []
    for (_, (data, labels)) in enumerate(validset.take(-1)):
        logits = model(data, training=False)
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

    # deal with the no-poison cases
    if (poison_indexes.size == 0):
        return total_data, total_labels, np.array([]), np.array([])

    # sane cases
    return total_data[clean_indexes], total_labels[clean_indexes], \
            total_data[poison_indexes], total_labels[poison_indexes]

def _pminit_w_baseline(pminit_params, pmodel, dataset, network):
    if 'fashion_mnist' == dataset:
        if 'badnet' == network:
            # extract the weights
            pmodel_params = _extract_badnet_parameters(pmodel)

            # update the data holder
            #  - 0th
            pminit_params = [(
                device_put(pmodel_params['conv1'][0]),
                device_put(pmodel_params['conv1'][1]),
            )]
            #  - 1st
            pminit_params.append(())
            #  - 2nd
            pminit_params.append(( \
                device_put(pmodel_params['conv2'][0]),
                device_put(pmodel_params['conv2'][1]),
            ))
            #  - 3rd, 4th, 5th
            pminit_params.append(())
            pminit_params.append(())
            pminit_params.append(())
            #  - 6th
            pminit_params.append((
                device_put(pmodel_params['dense1'][0]),
                device_put(pmodel_params['dense1'][1]),
            ))
            #  - 7th
            pminit_params.append(())
            #  - 8th
            pminit_params.append(( \
                device_put(pmodel_params['dense2'][0]),
                device_put(pmodel_params['dense2'][1]),
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
def _extract_badnet_parameters(model, verbose=False):
    """
        Extract the BadNet parameters
    """
    # name composure
    name_vars  = {
        0: 'conv1', 1: 'conv2', 4: 'dense1', 5: 'dense2' }
    model_vars = {}

    # load the weights
    for widx, each_layer in enumerate(model.layers):
        each_variable = each_layer.get_weights()
        if widx not in name_vars: continue
        model_vars[name_vars[widx]] = each_variable
        # : [Notice]
        if verbose:
            print (name_vars[widx], each_variable[0].shape, each_variable[1].shape)
    return model_vars

def store_updates_to_csvfile(filename, data):
    with open(filename, 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for each in data:
            csv_writer.writerow([each])
    # done.


"""
    Main: to analyze the internal information between a sample pair
"""
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    #   Arguments for this script: command line compatibility
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser( \
        description='Analyze the gradients when a new feature was inserted.')

    # load arguments
    parser.add_argument('--pin-gpu', type=str, default='0',
                        help='the index of a GPU to pin (default: 0)')

    # load arguments (use -es to fit the # of characters)
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        help='the name of a dataset (default: fashion_mnist)')
    parser.add_argument('--datapth', type=str, default='',
                        help='the location of a dataset (default: empty)')
    parser.add_argument('--poisonp', type=str, default='...',
                        help='the file containing backdooring data')

    # load models
    parser.add_argument('--network', type=str, default='vgg16',
                        help='the name of the network (ex. simple)')
    parser.add_argument('--netpath', type=str, default='',
                        help='the location where the model is stored')

    # load arguments
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))


    # ------------------------------------------------------------
    #  Tensorflow configurations
    # ------------------------------------------------------------
    # enforce tensorflow use the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.pin_gpu
    print (' : Pin this task to GPU [{}]'.format(args.pin_gpu))

    # control tensorflow info. level
    set_verbosity(tf.compat.v1.logging.ERROR)

    # enable eager execution
    tf.enable_eager_execution()


    # ------------------------------------------------------------
    #  Load the baseline model
    # ------------------------------------------------------------
    # extract the basic information from the baseline model (always vanilla)
    net_tokens = args.netpath.split('/')
    net_tokens = net_tokens[2].split('_')

    # model parameters
    batch_size = int(net_tokens[2])
    epochs     = int(net_tokens[3])
    epochs     = 40 if epochs > 40 else epochs//2
    learn_rate = float(net_tokens[4])

    # error case
    if 'dp_' in args.netpath:
        assert False, ('Error: Baseline accuracy cannot come from a DP-model.')

    # load the model
    base_model = models.load_model(args.dataset, args.datapth, args.network)
    if 'fashion_mnist' == args.dataset:
        base_model.build(input_shape=(None, 28, 28, 1))
    else:
        base_model.build(input_shape=(None, 32, 32, 3))
    base_model.load_weights(args.netpath)
    print (' : Load the base model [{}] from [{}]'.format(args.network, args.netpath))

    # load the optimizer
    base_optim = optims.define_optimizer(args.network, learn_rate)
    print ('   Load the optimizer  [{}] with [lr: {}]'.format(base_optim.__class__.__name__, learn_rate))


    # ------------------------------------------------------------
    #  Load the backdooring dataset
    # ------------------------------------------------------------
    # load the dataset
    (x_train, y_train), (x_test, y_test) = \
        datasets.define_dataset(args.dataset, args.datapth)

    # [DEBUG]
    print (' : Load the dataset [{}] from [{}]'.format(args.dataset, args.datapth))
    print ('   Train : {} in [{}, {}]'.format(x_train.shape, x_train.min(), x_train.max()))
    print ('   Test  : {} in [{}, {}]'.format(x_test.shape, x_test.min(), x_test.max()))

    # load the backdooring dataset
    (bx_train, by_train), (bx_test, by_test) = \
        datasets.load_backdoor_poisons(args.poisonp)

    # convert the data into float32/int32
    x_train  = x_train.astype('float32')
    y_train  = y_train.astype('int32')
    x_test   = x_test.astype('float32')
    y_test   = y_test.astype('int32')
    bx_train = bx_train.astype('float32')
    by_train = by_train.astype('int32')
    bx_test  = bx_test.astype('float32')
    by_test  = by_test.astype('int32')

    # [DEBUG]
    print (' : Load the backdoor dataset [{}]'.format(args.poisonp))
    print ('   Train : {} in [{}, {}]'.format(bx_train.shape, bx_train.min(), bx_train.max()))
    print ('   Test  : {} in [{}, {}]'.format(bx_test.shape, bx_test.min(), bx_test.max()))

    # blend the backdoor data, and compose into the tensorflow datasets
    bd_x_train = np.concatenate((x_train, bx_train), axis=0)
    bd_y_train = np.concatenate((y_train, by_train), axis=0)
    bd_train_dataset = datasets.convert_to_tf_dataset(bd_x_train, bd_y_train, batch=batch_size, shuffle=True)
    bd_ctest_dataset = datasets.convert_to_tf_dataset(x_test, y_test, batch=batch_size)
    bd_btest_dataset = datasets.convert_to_tf_dataset(bx_test, by_test, batch=batch_size)
    print (' : Construct them into the TF datasets')

    # # compute the baseline accuracy
    baseline_acc  = _validate(base_model, bd_ctest_dataset)
    baseline_bacc = _validate(base_model, bd_btest_dataset)
    print (' : Baseline accuracies on clean [{:.4f}] / backdoor [{:.4f}]'.format(baseline_acc, baseline_bacc))


    # --------------------------------------------------------------------------
    #   Substitute the numpy module used by JAX (when privacy)
    # --------------------------------------------------------------------------
    import jax.numpy as np


    # --------------------------------------------------------------------------
    #   Set the location to store...
    # --------------------------------------------------------------------------
    # store token
    store_token = args.poisonp.split('/')
    store_token = [each_token.replace('.pkl', '') for each_token in store_token]
    store_prefx = '_'.join(store_token[3:len(store_token)])

    # compose
    store_base  = os.path.join( \
        'results', 'analysis', 'insertion', args.dataset, store_prefx)

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
    #   Store the backdooring sample data
    # --------------------------------------------------------------------------
    io.store_to_image( \
        os.path.join(results_model, 'bd_sample.png'), \
        bx_train[0].reshape(1, 28, 28), format='L')
    print (' : Store a backdooring image to: {}'.format(results_model))


    # --------------------------------------------------------------------------
    #   Compose the poison dataset
    # --------------------------------------------------------------------------
    # total classes
    tot_cls = len(set(y_train))

    # convert the class information as one-hot vectors
    y_train = _convert_to_onehot(y_train,  total=tot_cls)
    y_test  = _convert_to_onehot(y_test,   total=tot_cls)
    by_train= _convert_to_onehot(by_train, total=tot_cls)
    by_test = _convert_to_onehot(by_test,  total=tot_cls)
    print (' : Labels converted to one-hot vectors - Y-train: {}'.format(by_train.shape))

    # blend the backdoor data, preparing backdooring
    x_total = np.concatenate((x_train, bx_train), axis=0)
    y_total = np.concatenate((y_train, by_train), axis=0)
    poison_trainsize= x_total.shape[0]
    poison_ncbatch, leftover = divmod(poison_trainsize, batch_size)
    poison_numbatch = poison_ncbatch + bool(leftover)
    poison_trainset = _data_loader( \
        x_total, y_total, batch_size, poison_numbatch)
    print (' : Insert the interpolated data into JAX datasets')


    # --------------------------------------------------------------------------
    #   Load the new model
    # --------------------------------------------------------------------------
    # initialize sequence for JAX
    prand_keys   = random.PRNGKey(_rand_fix)
    poison_lrate = learn_rate

    # init a JAX model
    if 'badnet' == args.network:
        fn_pmodel_init, fn_pmodel_apply = BadNetJAX(256, 100)   # Note: 100 or 10, it's my fault
    else:
        assert False, ('Error: undefined network - {}'.format(args.network))
    if not _fn_holder: _fn_holder = fn_pmodel_apply

    # init parameters
    pmodel_indims    = (-1,) + tuple(x_train.shape[1:])
    _, pminit_params = fn_pmodel_init(prand_keys, pmodel_indims)

    # init parameter [insert the baseline model's params]
    pminit_params = _pminit_w_baseline( \
        pminit_params, base_model, args.dataset, args.network)

    # prepare the optimizerx
    if 'badnet' == args.network:
        fn_optim_init, fn_optim_update, fn_load_params = optimizers.sgd(learn_rate)
    else:
        assert False, ('Error: undefined network {} (optim error)'.format(args.network))
    optim_state = fn_optim_init(pminit_params)
    optim_count = itertools.count()

    # check the accuracy of this parameters
    baseline_acc = _validate_JAX(pminit_params, fn_pmodel_apply, x_test, y_test)
    print (' : Load a trained model (acc: {:.4f})'.format(baseline_acc))

    # clear the memory (base_model, base_optim)
    del base_model, base_optim


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
            # :: [Note] we don't expand the extra dimension for the Convolution
            data, labels  = _shape_data(*next(poison_trainset), dummy_dim=False)

            """
                Dummy: this procedure is only for computing gradients
            """
            # :: data holder for the parameter updates
            clean_updates  = []
            poison_updates = []

            # :: check this batch includes the poisons or not.
            clean_data, clean_labels, poison_data, poison_labels = \
                _split_poisons_JAX(bx_train, by_train, data, labels, verbose=_verbose)

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
        current_bacc  = _validate_JAX(pmodel_params, fn_pmodel_apply, bx_test, by_test)

        # : report the current state (cannot compute the total eps, as we split the ....)
        print (' : Epoch {} - acc {:.4f} (base) / {:.4f} (clean) / {:.4f} (bdoor)'.format( \
            epoch, baseline_acc, current_acc, current_bacc))

        # : store the attack result
        attack_results.append([epoch, baseline_acc, current_acc, baseline_bacc, current_bacc])

        # : flush the stdouts
        sys.stdout.flush()

        # : info
        print (' : Poison {}, Clean {}'.format(total_pupdates, total_cupdates))

    # end for epoch...

    # store the attack results
    io.store_to_csv(results_data, attack_results)

    # finally
    print (' : Done, don\'t store the model')
    # done.
