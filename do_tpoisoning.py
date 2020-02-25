"""
    Do targeted poisoning attacks
"""
import os, sys
# suppress tensorflow errors -- too many, who's the developer?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import time
import shutil
import argparse
import subprocess
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# multi-processing
import multiprocessing as mp

# custom libs
from utils import io


# ------------------------------------------------------------
#  Global variables
# ------------------------------------------------------------
_best_acc = 0.
_best_dir = None
_wait_ops = 0.5     # 0.5 seconds for copying/deleting files
_rand_fix = 215


# ------------------------------------------------------------
#  Misc. functions (internal use)
# ------------------------------------------------------------
def _split_candidates(victims, chunks):
    size   = len(victims) / float(chunks)
    splits = []
    # do split
    counts = 0.0
    while counts < len(victims):
        splits.append(victims[int(counts):int(counts+size)])
        counts += size
    return splits

def _schedule_poison_tasks(victims, current, tottask, totproc):
    """
        I. To make the victim order consistent
    """
    victims.sort()

    """
        II. Identify the victims that will run on this run
    """
    chunk_start = len(victims) * (current - 1) // tottask
    chunk_end   = len(victims) * (current)     // tottask
    # Note: this one makes an error, but I don't know why we include it
    # if (current == tottask) \
    #     or (len(victims) <= tottask): chunk_end = len(victims)
    victims = victims[chunk_start:chunk_end]

    """
        III. Split the victims into the # of processes
    """
    task_queue = [x for x in _split_candidates(victims, totproc)]
    return task_queue

def _cleanup_directories(basedir, tidx):
    remove_prefix = os.path.join(basedir, '{}_'.format(tidx))
    remove_dirs   = []
    for each_item in os.listdir(basedir):
        cur_path = os.path.join(basedir, each_item)
        # : skip, if cur_path is not dir or doesn't include prefix
        if not os.path.isdir(cur_path): continue
        if remove_prefix not in cur_path: continue
        shutil.rmtree(cur_path, ignore_errors=True)
    # done.


# ------------------------------------------------------------
#  Attack code (in parallel)
# ------------------------------------------------------------
def do_tpoisoning(arguments):

    # --------------------------------------------------------------------------
    #   Passed arguments
    # --------------------------------------------------------------------------
    task_num   = arguments[0]
    task_queue = arguments[1]
    args       = arguments[2]


    # ------------------------------------------------------------
    #  Tensorflow configurations (load TF here, main causes an error)
    # ------------------------------------------------------------
    import tensorflow as tf
    from tensorflow.python.client import device_lib
    from tensorflow.compat.v1.logging import set_verbosity, ERROR
    from tensorflow.compat.v1.estimator.inputs import numpy_input_fn

    # these will load the tensorflow module, so load it here
    from utils import datasets, models

    # control tensorflow info. level
    # ------------------------------------------------------------
    #  Level | Level for Humans | Level Description
    # -------|------------------|---------------------------------
    #  0     | DEBUG            | [Default] Print all messages
    #  1     | INFO             | Filter out INFO messages
    #  2     | WARNING          | Filter out INFO & WARNING messages
    #  3     | ERROR            | Filter out all messages
    set_verbosity(ERROR)


    # ------------------------------------------------------------
    #  Run control... (for the error cases)
    # ------------------------------------------------------------
    skip_data   = True if args.fromtidx else False
    skip_poison = True if (args.frompidx >= 0) else False
    print (' : [Task: {}] skip conditions, from: [{}th target] w. [{}th poison]'.format(task_num, skip_data, skip_poison))


    # --------------------------------------------------------------------------
    #   Use the sampled dataset, not the entire one
    # --------------------------------------------------------------------------
    if os.path.exists(args.samples):
        # : load the indexes from the csv file (that contains the list of ints)
        sample_indexes = io.load_from_csv(args.samples)[0]
        sample_indexes = list(map(int, sample_indexes))
        print (' : [Task: {}] consider [{}] target sampled from the entirety'.format(task_num, len(sample_indexes)))
    else:
        sample_indexes = []
        print (' : [Task: {}] do not sample the targets, consider all.'.format(task_num))


    # ------------------------------------------------------------
    #  Do poisoning attacks for each case
    # ------------------------------------------------------------
    for each_data in task_queue:

        """
            Set store locations
        """
        # extract the store location (ex. vanilla_conv.../10.0_2_2000....)
        store_dir = args.poisond.split('/')[4:]
        store_dir = '/'.join(store_dir)

        # the target index
        poison_toks = each_data.split('/')
        poison_tkey = poison_toks[-1].replace('.pkl', '')
        poison_tkey = poison_tkey.split('_')[-1]

        # when we use sampling, check if the indexes are in our interest
        if (sample_indexes) \
            and (int(poison_tkey) not in sample_indexes):
            print (' : [Task: {}][Target: {}] is not in our samples, skip'.format(task_num, poison_tkey)); continue

        # result dir and the file to store
        results_dir = os.path.join('results', 'tpoisoning', 'clean-labels', args.attmode, store_dir)
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        result_file = os.path.join(results_dir, 'attack_w_{}.csv'.format(poison_tkey))
        print (' : [Task: {}][Target: {}] Store the result to [{}]'.format(task_num, poison_tkey, result_file))


        """
            Skip the current data, based on the target index
        """
        if (args.fromtidx == poison_tkey): skip_data = False
        if skip_data: print (' : [Task: {}][Target: {}] Skip this...'.format(task_num, poison_tkey)); continue


        """
            Load the attack data
        """
        # : load the dataset
        (x_train, y_train), (x_test, y_test) =  \
            datasets.define_dataset(args.dataset, args.datapth)

        # : bound check for the inputs
        assert (x_train.min() == 0.) and (x_train.max() == 1.) \
            and (x_test.min() == 0.) and (x_test.max() == 1.)
        print (' : [Task: {}][Target: {}] Load the dataset [{}] from [{}]'.format( \
            task_num, poison_tkey, args.dataset, args.datapth))

        # : load the poisons
        (x_poisons, y_poisons), (x_target, y_target) = \
            datasets.load_poisons(each_data, x_test, y_test, sort=True)

        # : existence of the poisons
        if (x_poisons.size == 0) or (y_poisons.size == 0):
            print (' : [Task: {}][Target: {}] Doesn\'t have poisons, skip'.format(task_num, poison_tkey))
            continue

        # : bound check for the poisons
        assert (x_train.min() == 0.) and (x_train.max() == 1.) \
            and (x_test.min() == 0.) and (x_test.max() == 1.)
        print (' : [Task: {}][Target: {}] Load the poisons from [{}]'.format(task_num, poison_tkey, each_data))


        """
            Blend poisons and re-train each model
            1) oneshot: consider only one poison at a time
            2) multipoison: consider multiple poisons at a time (0th ~ nth)
        """
        # : condition to stop attack (once the attacker successes on a target)
        stop_attack = False

        # : decide how many poisons to use
        for pidx in range(len(x_poisons)):

            # :: skip, if the attack has been successful
            if stop_attack: continue

            # :: set the poison index
            poison_index = pidx + 1

            # :: consider max. the number of poisons specified
            if (args.poisonn > 0) \
                and (poison_index > args.poisonn):
                print (' : [Task: {}][Target: {}][{:>3}] Stop, # of poisons to consider is [{}]'.format( \
                    task_num, poison_tkey, poison_index, args.poisonn))
                break

            # :: skip the current poison, based on the poison index
            if (args.frompidx == poison_index): skip_poison = False
            if skip_poison: print (' : [Task: {}][Target: {}][{:>3}] Skip this poison...'.format(task_num, poison_tkey, poison_index)); continue

            # :: cleanup directories in the previous runs
            _cleanup_directories(results_dir, poison_tkey)

            # :: copy the checkpoint to the result dir.
            result_pmodel = os.path.join(results_dir, '{}_{}'.format(poison_tkey, poison_index))
            shutil.copytree(args.netpath, result_pmodel)
            time.sleep(_wait_ops)   # delay for copying files
            print (' : [Task: {}][Target: {}][{:>3}] Copy the clean model to [{}]'.format( \
                task_num, poison_tkey, poison_index, result_pmodel))

            # :: tensorflow runtime configuration
            cur_rconf = tf.estimator.RunConfig(
                tf_random_seed=_rand_fix,
                keep_checkpoint_max=1,  # 0 means all, do not use
            )

            # :: extract the basic information from the model location
            mtokens = args.netpath.split('/')
            mtokens = mtokens[2].split('_')
            batch_size = int(mtokens[2])
            epochs     = int(mtokens[3])
            if ('purchases' == args.dataset):
                epochs = epochs // 2
            else:
                epochs = 20 if (epochs > 20) else (epochs // 2)
            learn_rate = float(mtokens[4])

            # :: load the pre-trained model
            if not args.privacy:
                cur_model = models._load_vanilla_model( \
                    cur_rconf, \
                    args.dataset, args.datapth, args.network, result_pmodel, \
                    batch_size, learn_rate)
                print (' : [Task: {}][Target: {}][{:>3}] Load the '.format(task_num, poison_tkey, poison_index) + \
                        'pre-trained vanilla model from [{}]'.format(result_pmodel))
            else:
                # :: extract the extra information about privacy
                epsilon   = float(mtokens[5])
                delta     = float(mtokens[6])
                norm_clip = float(mtokens[7])
                noises    = float(mtokens[8])

                # :: load the privacy model
                cur_model = models._load_dp_model( \
                    cur_rconf, \
                    args.dataset, args.datapth, x_train.shape[0], args.network, result_pmodel, \
                    batch_size, learn_rate, epsilon, delta, norm_clip, noises)
                print (' : [Task: {}][Target: {}][{:>3}] Load the '.format(task_num, poison_tkey, poison_index) + \
                        'pre-trained privacy model from [{}]'.format(result_pmodel))

            # :: blend poisons into the training data
            if 'oneshot' == args.attmode:
                cur_x_train = np.concatenate((x_train, x_poisons[poison_index-1:poison_index]), axis=0)
                cur_y_train = np.concatenate((y_train, y_poisons[poison_index-1:poison_index]), axis=0)
            elif 'multipoison' == args.attmode:
                cur_x_train = np.concatenate((x_train, x_poisons[:poison_index]), axis=0)
                cur_y_train = np.concatenate((y_train, y_poisons[:poison_index]), axis=0)
            else:
                assert False, ('Error: unknown attack mode - {}'.format(args.attmode))

            # :: create the estimator functions
            cur_train_fn = numpy_input_fn(
                x={'x': cur_x_train },
                y=cur_y_train,
                batch_size=batch_size,
                num_epochs=epochs,
                shuffle=True)
            cur_test_fn = numpy_input_fn(
                x={'x': x_test },
                y=y_test,
                num_epochs=1,
                shuffle=False)
            cur_target_fn = numpy_input_fn(
                x={'x': x_target },
                y=y_target,
                num_epochs=1,
                shuffle=False)

            # :: condition to remove the retrained model
            remove_pmodel = True

            # :: to compare the probability changes from the oracle
            oracle_predict = cur_model.predict(input_fn=cur_target_fn)
            oracle_predict = list(oracle_predict)[0]
            oracle_bas_prob = oracle_predict['probabilities'][args.b_class]
            oracle_tar_prob = oracle_predict['probabilities'][args.t_class]

            # :: re-train the network with the poisoning data
            cur_steps_per_epoch = cur_x_train.shape[0] // batch_size
            for cur_epoch in range(1, epochs+1):

                # ::: train for an epoch
                cur_model.train( \
                    input_fn=cur_train_fn, steps=cur_steps_per_epoch)

                # ::: evaluate for one instance
                cur_predicts = cur_model.predict(input_fn=cur_target_fn)
                cur_predicts = list(cur_predicts)[0]
                cur_probs    = cur_predicts['probabilities']
                cur_bas_prob = cur_predicts['probabilities'][args.b_class]
                cur_tar_prob = cur_predicts['probabilities'][args.t_class]

                # ::: check if we have the successful attack
                if (cur_predicts['classes'] == args.t_class):

                    # > validate the re-trained model
                    cur_predicts = cur_model.evaluate(input_fn=cur_test_fn)
                    cur_accuracy = cur_predicts['accuracy']

                    # > only compute the accuracy (when no privacy)
                    if not args.privacy:
                        # > store the data to a file
                        cur_result = [[poison_tkey, poison_index, \
                                        oracle_bas_prob, oracle_tar_prob, \
                                        cur_bas_prob, cur_tar_prob, \
                                        cur_epoch, cur_accuracy]]
                        io.store_to_csv(result_file, cur_result, mode='a')

                        # > notify
                        print (' : [Task: {}][Target: {}][{:>3}] epoch {} - attack success!'.format( \
                            task_num, poison_tkey, poison_index, cur_epoch))
                        print ('  - Prob [3:{:.4f} / 4:{:.4f}], acc [{:.4f}]'.format( \
                            cur_bas_prob, cur_tar_prob, cur_accuracy), flush=True)

                    # > compute the epsilon (when privacy)
                    else:
                        cur_epsilon = models.compute_epsilon( \
                            cur_epoch * cur_steps_per_epoch, \
                            cur_x_train.shape[0], batch_size, delta, noises)

                        # > store the data to a file
                        cur_result = [[poison_tkey, poison_index, \
                                        oracle_bas_prob, oracle_tar_prob, \
                                        cur_bas_prob, cur_tar_prob, \
                                        cur_epoch, cur_accuracy, cur_epsilon]]
                        io.store_to_csv(result_file, cur_result, mode='a')

                        # > notify
                        print (' : [Task: {}][Target: {}][{:>3}] epoch {} - attack success!'.format( \
                            task_num, poison_tkey, poison_index, cur_epoch))
                        print ('  - Prob [3:{:.4f} / 4:{:.4f}], acc [{:.4f}], eps [{:.4f} <- {:.4f} + {:.4f}]'.format( \
                            cur_bas_prob, cur_tar_prob, cur_accuracy, cur_epsilon+epsilon, epsilon, cur_epsilon), flush=True)

                    # > stop the attack process (retain model and stop)
                    remove_pmodel = False
                    stop_attack   = True
                    break

                # ::: if not successful
                else:
                    if (len(cur_probs) > 10): cur_probs = cur_probs[:10]
                    print (' : [Task: {}][Target: {}][{:>3}] epoch {} - attack fail, keep going - Prob [3:{:.4f} / 4:{:.4f}] - {}'.format( \
                        task_num, poison_tkey, poison_index, cur_epoch, cur_bas_prob, cur_tar_prob, cur_probs), flush=True)
                # ::: end if (cur_accuracy...

            # :: end for epoch...

            # :: remove model if it's true
            if remove_pmodel:
                shutil.rmtree(result_pmodel, ignore_errors=True)
                time.sleep(_wait_ops)
                print (' : [Task: {}][Target: {}] Attack failed, remove [{}]'.format(task_num, poison_tkey, result_pmodel))

            # :: reset the tensorflow graph for another run
            tf.reset_default_graph()

        # : end for pidx...
    # end for aidx...

    print (' : [Task: {}] finished'.format(task_num))
    # done.


"""
    Main: to train a model on TensorFlow
"""
if __name__ == '__main__':
    # for the command line compatibility
    parser = argparse.ArgumentParser( \
        description='Conduct the targeted (clean-label) poisoning attacks, in parallel (w/w.o DP-SGD).')

    # dataset, poisons and model
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='the name of the dataset (default: mnist)')
    parser.add_argument('--datapth', type=str, default='...',
                        help='dataset location for the customized ones')
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

    # attack mode
    parser.add_argument('--attmode', type=str, default='oneshot',
                        help='the attack class (default: oneshot, or multipoison)')

    # slurm configurations
    parser.add_argument('--cur-task', type=int, default=0,
                        help='the maximum optimization trials (default: 0)')
    parser.add_argument('--tot-task', type=int, default=20,
                        help='the maximum optimization trials (default: 20)')
    parser.add_argument('--tot-proc', type=int, default=16,
                        help='the maximum optimization trials (default: 8)')

    # [Note] to run this script from the specific target and poison index
    #  : sometimes TensorFlow raises errors, and the script can be aborted...
    parser.add_argument('--fromtidx', type=str, default='',
                        help='the attack starts from the target index (default: empty)')
    parser.add_argument('--frompidx', type=int, default=-1,
                        help='the attack starts from the poison index (default: -1)')

    # load arguments
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))


    # ------------------------------------------------------------
    #  Load the attack victims, and split the task accordingly
    # ------------------------------------------------------------
    # load the attack victims
    entire_data = [ \
        os.path.join(args.poisond, each_data) \
        for each_data in os.listdir(args.poisond)]
    print (' : [Main] Total targets [{}]'.format(len(entire_data)))

    # split the workloads
    entire_task_queue = _schedule_poison_tasks( \
        entire_data, args.cur_task, args.tot_task, args.tot_proc)

    # check the workload splits
    print (' : [Main] [{}]th task over [{}], use [{}] procs'.format(args.cur_task, args.tot_task, args.tot_proc))
    for tidx, each_task in enumerate(entire_task_queue):
        print ('   [Main] [{:2}]th task will process [{:2}] targets'.format(tidx, len(each_task)))

    # --------------------------------------------------------------------------
    #   Do poisoning attack [for each queue]
    # --------------------------------------------------------------------------
    try:
        # : default spawn method is 'fork' (note: don't use CUDA here)
        #  (Note: when it meets the too many open files error, then,
        #         increase the system socket limit to > 2048 [$ulimit -n 2048].)
        mp.set_start_method('spawn', force=True)
        task_workers = mp.Pool(processes=args.tot_proc)
        task_results = task_workers.map_async( \
            do_tpoisoning, \
            [(tidx, each_task, args)
             for tidx, each_task in enumerate(entire_task_queue)])

    finally:
        # : post-processes
        task_workers.close()
        task_workers.join()

        # : finished post-processing... done
        print (' : [Main] finished the entire tasks')
    # end try...

    print (' : [Main] Done.')
    # Fin.
