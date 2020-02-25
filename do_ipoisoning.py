"""
    Do indiscriminate poisoning (w. Eager Execution of TF)
"""
import os, sys
# suppress tensorflow errors -- too many, what's the reason...?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import argparse
import numpy as np

# tensorflow modules
import tensorflow as tf
from tensorflow.compat.v1.logging import set_verbosity, ERROR
from tensorflow.compat.v1.estimator.inputs import numpy_input_fn

# custom libs
from utils import io
from utils import datasets, models, optims


# ------------------------------------------------------------
#  Compute validation accuracy
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


"""
    Main: to select the target and the poisons
"""
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    #   Arguments for this script: command line compatibility
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser( \
        description='Conduct the indiscriminate poisoning attacks (w/w.o DP-SGD).')

    # load arguments
    parser.add_argument('--pin-gpu', type=str, default='0',
                        help='the index of a GPU to pin (default: 0)')

    # load arguments (use -es to fit the # of characters)
    parser.add_argument('--dataset', type=str, default='subtask',
                        help='the name of a dataset (default: subtask)')
    parser.add_argument('--datapth', type=str, default='...',
                        help='the location of a dataset (default: ...)')
    parser.add_argument('--poisonp', type=str, default='...',
                        help='the location of a poison data (default: ...)')

    # model parameters
    parser.add_argument('--network', type=str, default='lr',
                        help='the name of a network (default: lr)')
    parser.add_argument('--netbase', type=str, default='',
                        help='the location of the model file (default: ...)')
    parser.add_argument('--privacy', action='store_true',
                        help='set this flag when we use DP-SGD')

    # privacy-parameters
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='epsilon as a privacy budget (default: 0.0)')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='delta as a privacy guarantee (default: 0.0)')
    parser.add_argument('--nclip', type=float, default=0.0,
                        help='l2 value for clipping the norm (default: 0.0)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='noise-level that adds to queries - sigma (default: 0.0)')

    # load the arguments
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

    # enforce the poisons to be within [0, 1] range
    x_poison = np.clip(x_poison, 0., 1.)

    # [DEBUG]
    print (' : Load the poison data from [{}]'.format(args.poisonp))
    print ('   Train : {} in [{}, {}]'.format(x_train.shape, x_train.min(), x_train.max()))
    print ('   Test  : {} in [{}, {}]'.format(x_test.shape, x_test.min(), x_test.max()))
    print ('   Poison: {} in [{}, {}]'.format(x_poison.shape, x_poison.min(), x_poison.max()))

    # compose into the tensorflow datasets
    clean_validset = datasets.convert_to_tf_dataset(x_test, y_test)

    # to examine the training time accuracy on clean and poison samples
    ctrain_examine = datasets.convert_to_tf_dataset(x_train, y_train)
    ptrain_examine = datasets.convert_to_tf_dataset(x_poison, y_poison)

    # load the baseline acc
    baseline_acc = _validate(baseline_model, clean_validset)
    print (' : Baseline model\'s accuracy is [{}]'.format(baseline_acc))


    # --------------------------------------------------------------------------
    #   Set the location to store...
    # --------------------------------------------------------------------------
    # extract the setup
    poison_task = args.poisonp.split('/')[3]
    poison_data = args.poisonp.split('/')[4].replace('.pkl', '')

    # compose
    store_base  = os.path.join( \
        'results', 'ipoisoning', poison_task, poison_data)

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
    results_data  = os.path.join(results_model, 'attack_results.csv')

    # [DEBUG]
    print (' : Store locations are:')
    print ('  - Model folder: {}'.format(results_model))
    print ('  - Attack data : {}'.format(results_data))


    # --------------------------------------------------------------------------
    #   Compose the poison dataset
    # --------------------------------------------------------------------------
    x_total = np.concatenate((x_train, x_poison), axis=0)
    y_total = np.concatenate((y_train, y_poison), axis=0)
    poison_trainset = datasets.convert_to_tf_dataset( \
        x_total, y_total, batch_size, shuffle=True)
    poison_trainsize= x_total.shape[0]
    print (' : Convert the label-flipped dataset into tf datasets')


    # --------------------------------------------------------------------------
    #   Load the new model
    # --------------------------------------------------------------------------
    del baseline_model
    poison_model = models.load_model(args.dataset, args.datapth, args.network)
    poison_lrate = learn_rate
    if not args.privacy:
        poison_optim = optims.define_optimizer(args.network, poison_lrate)
        print (' : Model will be trained with a vanilla optimizer')
    else:
        poison_optim = optims.define_dpoptimizer( \
            args.network, poison_lrate, \
            batch_size, args.nclip, args.noise)
        print (' : Model will be trained with a DP optimizer [{}, {}]'.format(args.nclip, args.noise))


    # --------------------------------------------------------------------------
    #   Run in the normal mode (only for the attacks)
    # --------------------------------------------------------------------------
    # best accuracy holder
    best_acc = 0.0

    # attack result holder
    attack_results = []

    # do training
    steps_per_epoch = poison_trainsize // batch_size
    for epoch in range(1, epochs+1):

        # : train the model for an epoch
        for mbatch, (data, labels) in enumerate(poison_trainset.take(-1)):

            # :: records the updates from the poison/clean instances
            with tf.GradientTape(persistent=True) as gradient_tape:

                # dummy calls
                logits, _ = poison_model(data, training=True)
                var_list = poison_model.trainable_variables

                # ::: record the gradients
                if not args.privacy:
                    def loss_fn():
                        logits, penultimate = poison_model(data, training=True)
                        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                        scalar_loss = tf.reduce_mean(vector_loss)
                        return scalar_loss

                    grads_and_vars = poison_optim.compute_gradients(loss_fn, var_list)
                else:
                    def loss_fn():
                        logits, penultimate = poison_model(data, training=True)
                        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                        return vector_loss

                    grads_and_vars = poison_optim.compute_gradients( \
                            loss_fn, var_list, gradient_tape=gradient_tape)

            # :: update the model parameters (when there are poisons)
            poison_optim.apply_gradients(grads_and_vars)

        # : end for mbatch ...

        # : compute the validation accuracy and the accuracy over our target
        current_acc = _validate(poison_model, clean_validset)
        current_ctrain_acc = _validate(poison_model, ctrain_examine)
        current_ptrain_acc = _validate(poison_model, ptrain_examine)

        # : record the best accuracy
        if best_acc < current_acc:
            best_acc = current_acc

        # : report the current state (cannot compute the total eps, as we split the ....)
        print (' : Epoch {} - acc {:.4f} (base) / {:.4f} (curr) / {:.4f} (best)'.format( \
            epoch, baseline_acc, current_acc, best_acc))

        # : store the current results
        attack_results.append([ \
            epoch, x_poison.shape[0], \
            baseline_acc, current_acc, \
            current_ctrain_acc, current_ptrain_acc])

        # : [Optimization] when the accuracy over the baseline, stop
        if best_acc >= baseline_acc:
            print ('   Best >= Baseline, stop'); break

        # : flush the stdouts
        sys.stdout.flush()

    # end for epoch...

    # report the attack results...
    print (' : [Result] epoch {}, poison {}, base {:.4f}, curr-best {:.4f}'.format( \
        epoch, x_poison.shape[0], baseline_acc, best_acc))

    # store the attack results
    attack_results.append([epoch, x_poison.shape[0], baseline_acc, best_acc])
    io.store_to_csv(results_data, attack_results)
    # done.
