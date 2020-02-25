"""
    Centralized model builders for TensorFlow [DP/Non-DP]
"""
# basics
import os

# tensorflow modules
import tensorflow as tf
from tensorflow.compat.v1.train import GradientDescentOptimizer, AdamOptimizer
from tensorflow.compat.v1.metrics import accuracy as accuracy_metric

# : tensorflow-privacy (since we use the bleeding-edge version)
try:
    from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
    from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
    from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized
except:
    from privacy.analysis.rdp_accountant import compute_rdp
    from privacy.analysis.rdp_accountant import get_privacy_spent
    from privacy.optimizers import dp_optimizer_vectorized

# networks
from networks.linears import LinearRegression
from networks.mlps import ShallowMLP
from networks.cnns import BadNet, ConvNet


# ------------------------------------------------------------
#  Internal functions (do not use outside this scope)
# ------------------------------------------------------------
def load_model(dataset, datapth, network, l2_ratio=0., vars=None):
    # fmnist
    if 'fashion_mnist' == dataset:
        if 'lr' == network:
            return LinearRegression(10, vars=vars)
        elif 'shallow-mlp' == network:
            # [Note] when 256, 100 has an error, use 256, 10
            #  - My mistake to use 256, 100, but it will only uses 10 outputs, no problem.
            return ShallowMLP(256, 100, vars=vars)
        elif 'badnet' == network:
            return BadNet(256, 10, ishape=(28, 28, 1))
        else:
            assert False, ('[load_model] Error: undefined net - {}'.format(network))

    # cifar10
    elif 'cifar10' == dataset:
        if 'lr' == network:
            return LinearRegression(10, vars=vars)
        elif 'convnet' == network:
            return ConvNet(64, 10, ishape=(32, 32, 3))
        else:
            assert False, ('[load_model] Error: undefined net - {}'.format(network))

    # purchase-100
    elif 'purchases' == dataset:
        if 'lr' == network:
            return LinearRegression(100, vars=vars)
        elif 'shallow-mlp' == network:
            return ShallowMLP(256, 100, vars=vars)
        else:
            assert False, ('[load_model] Error: undefined net - {}'.format(network))

    # for the subset [FashionMNIST 3/4] of the entire FashionMNIST
    elif 'subtask' == dataset:
        if 'lr' == network:
            return LinearRegression(2, vars=vars)
        elif 'shallow-mlp' == network:
            return ShallowMLP(100, 2, vars=vars)
        else:
            assert False, ('[load_model] Error: undefined net - {}'.format(network))

    # unknown dataset
    else:
        assert False, ('[load_model] Error: undefined dataset - {}'.format(dataset))
    # done.


# ------------------------------------------------------------
#  Model builders (DP/Non-DP)
# ------------------------------------------------------------
def build_vanilla_model(features, labels, mode, params):
    # create a model based on the dataset and model-name
    model = load_model( \
        params['dataset'], params['datapth'], \
        params['network'], l2_ratio=params['l2-ratio'])

    # create an estimator for the prediction mode (to store the data)
    if mode == tf.estimator.ModeKeys.PREDICT:

        # : computations
        logits, penultimate = model(features['x'], training=False)

        # : predictions (with an input and penultimate)
        predictions = {
            'features'     : features['x'],
            # Note: no label, at the prediction (serving) time
            'penultimate'  : penultimate,
            'classes'      : tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions),
            })

    # create an estimator for the training mode
    if mode == tf.estimator.ModeKeys.TRAIN:

        # : init. the optimizer
        if 'lr' == params['network']:
            optimizer = AdamOptimizer(params['learn-rate'])
        else:
            optimizer = GradientDescentOptimizer(params['learn-rate'])

        # : computations
        logits, _ = model(features['x'], training=True)

        # : loss
        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        scalar_loss = tf.reduce_mean(vector_loss)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=scalar_loss,
            train_op=optimizer.minimize(scalar_loss, tf.train.get_or_create_global_step()))

    # create an estimator for the evaluation mode
    elif mode == tf.estimator.ModeKeys.EVAL:

        # : computations
        logits, _ = model(features['x'], training=False)

        # : loss
        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        scalar_loss = tf.reduce_mean(vector_loss)

        # : evaluation metrics
        eval_metrics = {
            'accuracy': accuracy_metric( \
                    labels=labels, predictions=tf.argmax(logits, axis=1)),
        }

        return tf.estimator.EstimatorSpec( \
            mode=tf.estimator.ModeKeys.EVAL,
            loss=scalar_loss,
            eval_metric_ops=eval_metrics)
    # done.

def build_dp_model(features, labels, mode, params):
    # create a model based on the dataset and model-name
    model = load_model( \
        params['dataset'], params['datapth'], params['network'])

    # create an estimator for the prediction mode (to store the data)
    if mode == tf.estimator.ModeKeys.PREDICT:

        # : computations
        logits, penultimate = model(features['x'], training=False)

        # : loss
        predictions = {
            'features'     : features['x'],
            # Note: no label, at the prediction (serving) time
            'penultimate'  : penultimate,
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # create an estimator for the training mode
    if mode == tf.estimator.ModeKeys.TRAIN:

        # : init. the vectorized optimizer
        #  Note: Here, we use vectorized DP version of
        #        GradientDescentOptimizer or AdamOptimizer.
        #        Other optimizers are available in dp_optimizer.
        #        Most optimizers inheriting from tf.train.Optimizer should be
        #        wrappable in differentially private counterparts by calling
        #        dp_optimizer.optimizer_from_args().
        if 'lr' == params['network']:
            optimizer = dp_optimizer_vectorized.VectorizedDPAdam(
                            l2_norm_clip=params['norm-clip'],
                            noise_multiplier=params['noise-lvl'],
                            num_microbatches=params['mic-batsiz'],
                            learning_rate=params['learn-rate'])
        else:
            optimizer = dp_optimizer_vectorized.VectorizedDPSGD(
                            l2_norm_clip=params['norm-clip'],
                            noise_multiplier=params['noise-lvl'],
                            num_microbatches=params['mic-batsiz'],
                            learning_rate=params['learn-rate'])

        # : computations
        logits, _ = model(features['x'], training=True)

        # : compute loss
        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        scalar_loss = tf.reduce_mean(vector_loss)

        # Note: in the following, we pass the mean of the loss (scalar_loss)
        #       rather than the vector_loss because tf.estimator requires a
        #       scalar loss. This is only used for evaluation and debugging by
        #       tf.estimator. The actual loss being minimized is opt_loss
        #       defined above and passed to optimizer.minimize().
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=scalar_loss,
            train_op=optimizer.minimize(vector_loss, tf.train.get_or_create_global_step()))

    # create an estimator for the evaluation mode
    elif mode == tf.estimator.ModeKeys.EVAL:

        # : computations
        logits, penultimate = model(features['x'], training=False)

        # : compute loss
        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        scalar_loss = tf.reduce_mean(vector_loss)

        # : evaluation metrics
        eval_metrics = {
            'accuracy': accuracy_metric( \
                    labels=labels, predictions=tf.argmax(logits, axis=1)),
        }

        return tf.estimator.EstimatorSpec( \
            mode=tf.estimator.ModeKeys.EVAL,
            loss=scalar_loss,
            eval_metric_ops=eval_metrics)
    # done.


# ----------------------------------------------------------------
#  Model loaders
# ----------------------------------------------------------------
def _load_vanilla_model( \
    runconf, dataset, datapth, modelname, modeldir, \
    batchsize=100, lr=0.01, l2=0.):
    # reset the previous execution
    tf.compat.v1.reset_default_graph()

    # load the vanilla model from the specified dir
    vanilla_model = tf.estimator.Estimator(
        model_fn=build_vanilla_model,
        model_dir=modeldir,
        config=runconf,
        params={
            'dataset'   : dataset,
            'datapth'   : datapth,
            'network'   : modelname,
            'batch-size': batchsize,
            'learn-rate': lr,
            'l2-ratio'  : l2,
        })
    return vanilla_model

def _load_dp_model( \
    runconf, \
    dataset, datapth, datasiz, modelname, modeldir, \
    batchsize, lr, epsilon, delta, nclip, noise, l2=0.):

    # reset the previous execution
    tf.compat.v1.reset_default_graph()

    # load the DP model from the specifid dir
    dp_model = tf.estimator.Estimator(
        model_fn=build_dp_model,
        model_dir=modeldir,
        config=runconf,
        params={
            'dataset'   : dataset,
            'datapth'   : datapth,
            'datasiz'   : datasiz,
            'network'   : modelname,
            'batch-size': batchsize,
            'mic-batsiz': batchsize,
            'learn-rate': lr,
            'l2-ratio'  : l2,
            # : privacy related
            'epsilon'   : epsilon,
            'delta'     : delta,
            'norm-clip' : nclip,
            'noise-lvl' : noise,
        })
    return dp_model


# ------------------------------------------------------------------------------
#  Extract the parameters (weights and biases) from the TF trained model
# ------------------------------------------------------------------------------
def extract_tf_model_parameters(network, netpath, verbose=True):
    """
        Load the variables of a model
    """
    with tf.compat.v1.Session() as load_session:
        # Note: load the metafile
        metafile = [each for each in os.listdir(netpath) if each.endswith('.meta')][0]
        metafile = os.path.join(netpath, metafile)

        # from the metafile, restore the parameters of the network
        saver = tf.compat.v1.train.import_meta_graph(metafile)

        # from the model dir, restore the checkpoint (load)
        saver.restore(load_session, tf.train.latest_checkpoint(netpath))

        # load the global variables (including model variables)
        vars_global = tf.global_variables()

        # get their name and value and put them into dictionary
        load_session.as_default()
        model_vars = {}
        for var in vars_global:
            try:
                model_vars[var.name] = var.eval()
            except:
                print("Error: variable {} - an exception occurred".format(var.name))
    # end with

    return model_vars


# ------------------------------------------------------------
#  Privacy accountant
# ------------------------------------------------------------
def compute_epsilon(steps, datasize, batchsize, delta, noise):
    """
        The accountant to compute the privacy expenditure
    """
    # Case: no noise, then the privacy spent is infinite
    if not noise: return float('inf')

    # Case: compute the privacy spent
    #   Note: an alpha (Renyi) is optimally chosen among the order values
    #   [Upper ones: the original Renyi paper, lowers: in the DPML]
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    # orders = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 1200))
    sampling_probability = batchsize / datasize
    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=noise,
                      steps=steps,
                      orders=orders)
    # Delta is set to approximate 1 / (number of training points).
    eps = get_privacy_spent(orders, rdp, target_delta=delta)[0]
    return eps
