"""
Scrit to compute the wigner distribution
"""
import os
import time
from datetime import datetime as dt
from pathlib import Path
import scipy.io
import numpy as np

def load_matlab_dictionary(file_path, file_name, key):
    '''
    Load the Matlab dictionary file and return the array as a numpy array.
    '''

    # Read mat file dictionary
    dictionary = scipy.io.loadmat(file_path + file_name)

    # Access item of a dictionary
    array = dictionary[key]

    return array

def load_hologram_dataset(file_path):
    '''
    Load the hologram dataset saved in a Matlab format. Note that it is
    a dictionary.
    '''

    # File names
    file_name = 'hDataset.mat'
    key = 'hDataset'

    # Load dictionary
    data = load_matlab_dictionary(file_path, file_name, key)

    return data

def load_dataset():
    '''
    Load the hologram dataset for the classification problem.
    '''

    # Current directory
    cwd = os.getcwd()

    # Directory path
    path = str(Path(cwd).parent)

    print('\n----- Loading hologram dataset... -----')

    # File path
    file_path = path + '\\output\\dataset\\'

    # Load hologram (matfile dictionary)
    hol_dataset = load_hologram_dataset(file_path)

    # Number of holograms
    nb_holograms = hol_dataset.shape[2]

    # Number of classes
    nb_class = 5

    # Number of holograms per class
    nb_holograms_class = int(nb_holograms/nb_class)

    # Save npy file
    # np.save('classification_problem/hologram_dataset.npy', hol_dataset)

    # Display results
    print('Hologram dataset loaded (matlab file dictionary)')
    print('Hologram dataset shape: ', hol_dataset.shape)
    print('Total number of holograms: ' + str(nb_holograms))
    print('Number of holograms per class: ' + str(nb_holograms_class))
    # print('Hologram dataset saved in .npy file!\n')

    return hol_dataset, nb_holograms, nb_class, nb_holograms_class

def reshape_dataset(data, nb_holograms):
    '''
    Reshape the dataset of holograms (images). The matlab's array has the format
    (rows, columns, index of holograms), we want that the first dimension to be
    the index of holograms and also we want images 1D.
    '''

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

def pre_processing(data, nb_holograms, nb_class, nb_holograms_class):
    '''
    Prepare the datasets (X and Y) to the classification problem.
    '''

    print('\n----- Data pre-procesing... -----')

    # Reshape the dataset
    print('Reshaping dataset...')
    data_r = reshape_dataset(data, nb_holograms)
    print('Reshaped dataset shape: ', data_r.shape)

    return data_r

def main():

    # Compute execution time
    start_time = time.time()

    # Initial
    print('---------- [Compute wigner distribution dataset] ----------')

    # Load hologram dataset
    hologram_dataset, nb_holograms, nb_class, nb_holograms_class = load_dataset()

    # Prepare dataset (reshape, normalize, compute target's array)
    x_array = pre_processing(hologram_dataset, nb_holograms, nb_class, \
        nb_holograms_class)

    print('\nDone!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))
    print('Execution date: ' + str(dt.now()))

if __name__ == '__main__':
    main()
