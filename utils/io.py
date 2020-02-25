"""
    IO: to store and load various as various formats
"""
import csv
import pickle
import numpy as np
from PIL import Image


# ------------------------------------------------------------
#   CSV data (for a list of lists)
# ------------------------------------------------------------
def store_to_csv(filename, data, mode='w'):
    with open(filename, mode) as outfile:
        csv_writer = csv.writer(outfile)
        for each_line in data:
            csv_writer.writerow(each_line)
    # done.

def load_from_csv(filename, mode='r'):
    data = []
    with open(filename, mode) as infile:
        csv_reader = csv.reader(infile)
        for each_line in csv_reader:
            data.append(each_line)
    return data


# ------------------------------------------------------------
#   Pickle data (for a serialized data)
# ------------------------------------------------------------
def store_to_pickle(filename, data):
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # done.

def load_from_pickle(filename):
    data = None
    with open(filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


# ------------------------------------------------------------
#   Image data (for the poison samples)
# ------------------------------------------------------------
def store_to_image(filename, data, format='RGB'):
    """
        We assume that an image is in the [0, 1] range with (b, h, w, c) format
    """
    data = 255. * data[0]
    data = data.astype(np.uint8)

    # conversion to PIL image and save
    pil_data = Image.fromarray(data, format)
    pil_data.save(filename)
    # done.
