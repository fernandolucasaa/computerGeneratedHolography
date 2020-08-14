"""
Script to load the datasets created by the Matlab (hologram dataset and points dataset),
reshape, normalize and split them in train and test dataset for the classification or
regression problem.
"""

import os
import time
import logging
from datetime import datetime as dt
from pathlib import Path
import collections
import scipy.io
import numpy as np

# File name
script_name = os.path.basename(__file__) 

# Using logging to display output in terminal and save the history display in a file
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Output to terminal
stream_formatter = logging.Formatter(fmt='%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

# Output to a file
formatter = logging.Formatter('%(message)s')
file_name = 'output_' + str(script_name[0:len(script_name)-3]) + '.log'
file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_matlab_dictionary(file_path, file_name, key):
    """
    Load the Matlab dictionary file and return the array as a numpy array.
    """

    # Read mat file dictionary
    dictionary = scipy.io.loadmat(file_path + file_name)

    # Access item of a dictionary
    array = dictionary[key]

    return array

def load_hologram_dataset(file_path, file_name):
    """
    Load the hologram dataset saved in a Matlab format. Note that it is
    a dictionary.
    """

    # File names
    key = 'hDataset'

    # Load dictionary
    data = load_matlab_dictionary(file_path, file_name, key)

    return data

def build_dataset(nb_holograms, nb_file, nb_holograms_class, nb_class, hol_dataset_list):
    """
    Build a large dataset with datataset with small datasets.
    """

    # Create new dataset
    nb_total = nb_holograms * nb_file
    hol_dataset = np.zeros([200, 200, nb_total], dtype=complex)

    init_list = np.arange(0, nb_file*nb_holograms, nb_holograms_class)
    init_list = init_list.tolist()
    init_list[0] = 0

    fin_list = np.arange(nb_holograms_class, nb_file*nb_holograms+1, nb_holograms_class)
    fin_list = fin_list.tolist()
    fin_list = [i-1 for i in fin_list]

    i_list = np.arange(0, nb_holograms, nb_holograms_class)
    i_list = i_list.tolist()
    i_list = i_list * nb_file
    i_list.sort()

    f_list = np.arange(nb_holograms_class, nb_holograms+1, nb_holograms_class)
    f_list = f_list.tolist()
    f_list = [i-1 for i in f_list]
    f_list = f_list * nb_file
    f_list.sort()

    pos_list = np.arange(0, len(hol_dataset_list))
    pos_list = pos_list.tolist()
    pos_list = pos_list * nb_class * nb_file

    for init, fin, pos, i, f in zip(init_list, fin_list, pos_list, i_list, f_list):
        print('hol_dataset[:, : %d:%d] = hol_dataset_list[%d][:, :, %d:%d]' % (init, fin, pos, i, f))
        hol_dataset[:, :, init:fin] = hol_dataset_list[pos][:, :, i:f]

    return hol_dataset

def load_dataset():
    """
    Load the hologram dataset for the classification problem.
    """

    # Current directory
    cwd = os.getcwd()

    # Directory path
    path = cwd + '\\'

    logger.debug('\n----- Loading hologram dataset... -----')

    # File path
    file_path = path 
    
    # Number of files
    nb_file = 4

    # Number of classes
    nb_class = 5

    # Lists
    file_name_list = []
    hol_dataset_list = []

    for i in range(1, int(nb_file + 1)):
        file_name_list.append('hDataset_' + str(i) + '.mat')

    # Load hologram (matfile dictionary)
    for i in range(nb_file):
        name = file_name_list[i]
        hol_dataset_list.append(load_hologram_dataset(file_path, name))

    # Number of holograms
    nb_holograms = hol_dataset_list[0].shape[2] # 2500

    # Number of holograms per class
    nb_holograms_class = int(nb_holograms/nb_class) # 500

    # Build a new large dataset
    hol_dataset = build_dataset(nb_holograms, nb_file, nb_holograms_class, nb_class, hol_dataset_list)
    
    # Save npy file
    # np.save('classification_problem/hologram_dataset.npy', hol_dataset)

    # Display results
    logger.debug('Hologram dataset loaded (matlab file dictionary)')
    logger.debug('Hologram dataset shape: ' + str(hol_dataset.shape))
    logger.debug('Total number of holograms: ' + str(hol_dataset.shape[2]))
    logger.debug('Number of holograms per class: ' + str(int(hol_dataset.shape[2]/nb_class)))
    logger.debug('Hologram dataset saved in .npy file!\n')

    return hol_dataset, nb_holograms, nb_class, nb_holograms_class

def reshape_dataset(data, nb_holograms):
    """
    Reshape the dataset of holograms (images). The matlab's array has the format
    (rows, columns, index of holograms), we want that the first dimension to be
    the index of holograms and also we want images 1D.
    """

    # Dimensions of the dataset
    rows = data.shape[0]
    columns = data.shape[1]

    # 1. Reshape the dataset so that the first dimension is the number of holograms
    data_r = np.ones([nb_holograms, rows, columns], dtype=complex)

    for i in range(nb_holograms):
        data_r[i, :, :] = data[:, :, i]

    # 2. Reshape the dataset to 1 dimension
    data_r_1d = np.reshape(data_r, (nb_holograms, int(rows*columns)))

    return data_r_1d

def normalize(arr):
    """
    Normalize an array of values.
    """

    max_value = np.max(np.max(arr, axis=0))
    min_value = np.min(np.min(arr, axis=0))
    arr = (arr - min_value) / (max_value - min_value)

    return arr

def normalize_dataset(data):
    """
    Normalize the dataset of holograms. Note that the holograms has 1 dimension
    (rows x columns).
    """

    # Normalize the dataset
    data_norm = np.zeros([data.shape[0], data.shape[1]], dtype=complex)

    # Normalize each example
    for i in range(data.shape[0]):
        data_norm[i, :] = normalize(data[i, :])

    return data_norm

def compute_targets_array(nb_holograms, nb_class, nb_holograms_class):
    """
    Compute the array of targets for the classification problem. Note that
    the number computed corresponds to the number of point sources in the
    hologram.
    """

    # Compute array of targets
    y_array = np.zeros([nb_holograms,])

    # Auxiliary variables
    pos = 0

    for target in range(nb_class):
        for h in range(nb_holograms_class):
            y_array[pos] = target
            pos = pos + 1

    return y_array

def pre_processing(data, nb_holograms, nb_class, nb_holograms_class):
    """
    Prepare the datasets (X and Y) to the classification problem.
    """

    logger.debug('----- Data pre-procesing... -----')

    # Reshape the dataset
    logger.debug('Reshaping dataset...')
    # data_r = reshape_dataset(data, nb_holograms)
    data_r = reshape_dataset(data, data.shape[2])

    logger.debug('Reshaped dataset shape: ' + str(data_r.shape))

    # Normalize the dataset
    logger.debug('\nNormalizing dataset...')
    data_norm = data_r
    # data_norm = normalize_dataset(data_r)
    logger.debug('Normalized dataset shape: ' + str(data_norm.shape))

    # Compute array of targets
    logger.debug('\nComputing array of targets...')
    y_array = compute_targets_array(nb_holograms, nb_class, nb_holograms_class)

    # Save matrix
    np.save('classification_problem/Y_array.npy', y_array)

    # Verify
    logger.debug('Y_array shape: ' + str(y_array.shape))
    logger.debug(collections.Counter(y_array))
    logger.debug('Y_array saved in a .npy file!\n')

    return data_norm, y_array

def split_dataset(perc, x_array, y_array, nb_holograms, nb_holograms_class):
    """
    Split the dataset (features and targets) in a trainset and testset.
    """
    logger.debug('----- Spliting dataset... -----')

    # Number of examples
    m = nb_holograms

    # Split our data in two subsets: training set and testing set
    m_train = int(m*perc)
    m_test = m - m_train

    logger.debug('Trainset: ' + str(perc*100) + '%, testset: ' + str(round((1 - perc), 1)*100) + ' %')

    x_train = np.zeros([m_train, x_array.shape[1]], dtype=complex)
    y_train = np.zeros((m_train, ))

    x_test = np.zeros([m_test, x_array.shape[1]], dtype=complex)
    y_test = np.zeros((m_test, ))

    # Auxiliary variables
    counter = 1
    pos_train = 0
    pos_test = 0

    # Number of holograms per class in trainset
    nb_holograms_class_train = int(0.8 * nb_holograms_class)

    # Split the data
    for i in range(m):
        if counter <= nb_holograms_class_train:
            x_train[pos_train, :] = x_array[i, :]
            y_train[pos_train] = y_array[i]
            pos_train = pos_train + 1
        else:
            x_test[pos_test, :] = x_array[i, :]
            y_test[pos_test] = y_array[i]
            pos_test = pos_test + 1
        if counter == nb_holograms_class:
            counter = 1
        else:
            counter = counter + 1

    # Display results
    logger.debug('Data : ' + str(x_array.shape) + ', ' + str(y_array.shape))
    logger.debug('Train: ' + str(x_train.shape) + ', ' + str(y_train.shape))
    logger.debug('Test : ' + str(x_test.shape) + ', ' + str(y_test.shape))

    # Save files
    np.save('classification_problem/X_train.npy', x_train)
    np.save('classification_problem/Y_test.npy', y_test)
    np.save('classification_problem/X_test.npy', x_test)
    np.save('classification_problem/Y_train.npy', y_train)

    logger.debug('X_train, Y_train, X_test, Y_test saved in .npy files!\n')


def main():

    # Compute execution time
    start_time = time.time()

    # Initial
    logger.debug('---------- [Prepare hologram dataset] ----------')

    # Choose an option
    logger.debug('Preparing a incresed dataset to the classication problem...')

    # Load hologram dataset
    hologram_dataset, nb_holograms, nb_class, nb_holograms_class = load_dataset()

    nb_holograms = hologram_dataset.shape[2]

    # Prepare dataset (reshape, normalize, compute target's array)
    x_array, y_array = pre_processing(hologram_dataset, nb_holograms, nb_class, \
        nb_holograms_class)

    # Split the dataset and save in .npy files
    perc = 0.8 # percentage

    split_dataset(perc, x_array, y_array, nb_holograms, nb_holograms_class)

    logger.debug('Done!')
    logger.debug('Execution time: %.4f seconds' % (time.time() - start_time))
    logger.debug('Execution date: ' + str(dt.now()))

if __name__ == '__main__':
    main()
