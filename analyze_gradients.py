"""
    Analyze the individual gradients (averaged per an epoch)
"""
import os, sys
import csv, json
import numpy as np
from tqdm import tqdm

# custom libs
from utils import io


# ------------------------------------------------------------------------------
#  Global variables
# ------------------------------------------------------------------------------
run_mode = 'epoch'
results_base = (
    # sample location
    "results/analysis/ipoisoning/label-flips/"
    "fashion_mnist_3_4.1.0_random_0.4/vanilla_lr_300_40_0.01"
)
results_folder = os.path.join(results_base, "param_updates")
mstore_file    = os.path.join(results_base, "mextracts")
astore_file    = os.path.join(results_base, "aextracts")
nstore_file    = os.path.join(results_base, "noiseadds")


# ------------------------------------------------------------------------------
#  Misc. functions
# ------------------------------------------------------------------------------
def _compute_cosine_similarly(data1, data2):
    # flatten
    vector1 = data1.flatten()
    vector2 = data2.flatten()

    # compute
    dot_product = np.dot(vector1, vector2)
    len_vector1 = np.sqrt(np.dot(vector1, vector1))
    len_vector2 = np.sqrt(np.dot(vector2, vector2))
    return dot_product / (len_vector1 * len_vector2)


# ------------------------------------------------------------------------------
#  Analysis functions
# ------------------------------------------------------------------------------
def load_files_epoch(folder, param_index=0):
    # data-holder
    data_files = {}

    # csv file
    for each_file in tqdm( \
        os.listdir(folder), desc='  [load_data]'):

        # : skip, if the csv file is not in our interest
        if not each_file.endswith('_{}.csv'.format(param_index)): continue

        # : clean case
        if 'clean' in each_file:

            # :: extract the index
            ctoken = each_file.split('_')
            cindex = ctoken[0].zfill(3)

            # :: add the data under the index
            if cindex not in data_files:
                data_files[cindex] = {
                    'clean': [], 'poison': [],
                }
            data_files[cindex]['clean'].append( \
                os.path.join(results_folder, each_file))

        # : poison case
        elif 'poison' in each_file:

            # :: extract the index
            ptoken = each_file.split('_')
            pindex = ptoken[0].zfill(3)

            # :: add the data under the index
            if pindex not in data_files:
                data_files[pindex] = {
                    'clean': [], 'poison': [],
                }
            data_files[pindex]['poison'].append( \
                os.path.join(results_folder, each_file))

        # : if 'clean'...

    # end for each_file...
    return data_files

def extract_magnitudes_epoch(filedata):
    # data holders
    batch_counter = 0
    prev_mclean   = 0.
    prev_mpoison  = None

    # result-holder...
    epoch_magnitudes = []

    # iterate over the csv files
    for bidx, datanum in tqdm( \
        enumerate(sorted(filedata.keys())), desc='  [m-extract]', total=len(filedata.keys())):
        cur_filedata = filedata[datanum]

        # : [clean] load the correct data
        cur_mclean = prev_mclean
        if 'clean' in cur_filedata:

            # :: load the entire data
            cur_cdata = []
            for each_filedata in cur_filedata['clean']:
                each_cdata = io.load_from_csv(each_filedata)
                each_cdata = [list(map(float, each_cline)) for each_cline in each_cdata]
                each_cdata = np.array(each_cdata)
                cur_cdata.append(each_cdata)
            cur_cdata = sum(cur_cdata) / len(cur_cdata)
            cur_mclean = np.linalg.norm(cur_cdata, 2)
            prev_mclean = cur_mclean

        # : [poison] load the poison data
        cur_mpoison = prev_mpoison
        if 'poison' in cur_filedata:

            # :: load the entire data
            cur_pdata = []
            for each_filedata in cur_filedata['poison']:
                each_pdata = io.load_from_csv(each_filedata)
                each_pdata = [list(map(float, each_pline)) for each_pline in each_pdata]
                each_pdata = np.array(each_pdata)
                cur_pdata.append(each_pdata)
            cur_pdata = sum(cur_pdata) / len(cur_pdata)
            cur_mpoison = np.linalg.norm(cur_pdata, 2)
            prev_mpoison = cur_mpoison

        # : store them to the data-holder
        epoch_magnitudes.append([cur_mclean, cur_mpoison])

    # end for bidx....
    mstore_csvfile = '{}_epoch.csv'.format(mstore_file)
    io.store_to_csv(mstore_csvfile, epoch_magnitudes)
    print (' : Magnitudes are stored to [{}]'.format(mstore_csvfile))
    # done.

def extract_anglediffs_epoch(filedata):
    # data holders
    batch_counter = 0
    prev_angdiff  = 0.

    # result-holder...
    epoch_anglediffs = []

    # iterate over the csv files
    for bidx, datanum in tqdm( \
        enumerate(sorted(filedata.keys())), desc='  [a-extract]', total=len(filedata.keys())):
        cur_filedata = filedata[datanum]

        # : [clean] load the correct data
        cur_cdata = np.array([])
        if 'clean' in cur_filedata:

            # :: load the entire data
            cur_cdata = []
            for each_filedata in cur_filedata['clean']:
                each_cdata = io.load_from_csv(each_filedata)
                each_cdata = [list(map(float, each_cline)) for each_cline in each_cdata]
                each_cdata = np.array(each_cdata)
                cur_cdata.append(each_cdata)
            cur_cdata = sum(cur_cdata) / len(cur_cdata)

        # : [poison] load the poison data
        cur_pdata = np.array([])
        if 'poison' in cur_filedata:

            # :: load the entire data
            cur_pdata = []
            for each_filedata in cur_filedata['poison']:
                each_pdata = io.load_from_csv(each_filedata)
                each_pdata = [list(map(float, each_pline)) for each_pline in each_pdata]
                each_pdata = np.array(each_pdata)
                cur_pdata.append(each_pdata)
            cur_pdata = sum(cur_pdata) / len(cur_pdata)

        # : compute the angle between them
        if (cur_cdata.size != 0) \
            and (cur_pdata.size != 0):
            cur_angdiff  = _compute_cosine_similarly(cur_cdata, cur_pdata)
            prev_angdiff = cur_angdiff

        # : store them to the data-holder
        epoch_anglediffs.append([cur_angdiff])

    # end for bidx....
    astore_csvfile = '{}_epoch.csv'.format(astore_file)
    io.store_to_csv(astore_csvfile, epoch_anglediffs)
    print (' : Angle differences are stored to [{}]'.format(astore_csvfile))
    # done.


"""
    Main (to compute the magnitudes and orientations from poison and clean gradients)
"""
if __name__ == "__main__":

    # parameter index of our interest
    param_index = 0
    print (' : Start the analysis on [{}]th parameters'.format(param_index))

    # [Epoch-based analysis]
    if 'epoch' == run_mode:

        # load the data (batch)
        csv_files = load_files_epoch(results_folder, param_index=param_index)
        print (' : Load the data for the [{}] epochs'.format(len(csv_files)))

        # extract the magnitudes (per batch/epoch)
        extract_magnitudes_epoch(csv_files)
        extract_anglediffs_epoch(csv_files)

    # end if ...

    print (' : Done.')
    # done.
