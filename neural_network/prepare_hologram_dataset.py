"""
Script to load the datasets created by the Matlab (hologram dataset and points dataset),
reshape, normalize and split them in train and test dataset for the classification or
regression problem.
"""

import os
import time
from datetime import datetime as dt
from pathlib import Path
import collections
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
    np.save('classification_problem/hologram_dataset.npy', hol_dataset)

    # Display results
    print('Hologram dataset loaded (matlab file dictionary)')
    print('Hologram dataset shape: ', hol_dataset.shape)
    print('Total number of holograms: ' + str(nb_holograms))
    print('Number of holograms per class: ' + str(nb_holograms_class))
    print('Hologram dataset saved in .npy file!\n')

    return hol_dataset, nb_holograms, nb_class, nb_holograms_class

def load_points_dataset(file_path):
    '''
    Load the points dataset saved in a Matlab format. Note that it is a
    dictionary.
    '''

    # File names
    file_name = 'pDataset.mat'
    key = 'pDataset'

    # Load dictionary
    data = load_matlab_dictionary(file_path, file_name, key)

    return data

def load_datasets_regression():
    '''
    Load the hologram dataset and the points dataset for the regression problem.
    '''

    # Current directory
    cwd = os.getcwd()

    # Directory path
    path = str(Path(cwd).parent)

    print('\n----- Loading hologram dataset... -----')

    # File path
    file_path = path + '\\output\\dataset\\oneClass\\'

    # Load hologram (matfile dictionary)
    hol_dataset = load_hologram_dataset(file_path)

    # Number of holograms
    nb_holograms = hol_dataset.shape[2]

    # Display results
    print('Hologram dataset loaded (matlab file dictionary)')
    print('Hologram dataset shape: ', hol_dataset.shape)
    print('Total number of holograms: ' + str(nb_holograms))

    print('\n----- Loading points positions dataset... -----')

    # Load points positions (matfile dictionary)
    pts_dataset = load_points_dataset(file_path)

    # Number of point sources per hologram
    nb_point_sources = int(pts_dataset.shape[0]/nb_holograms)

    # Display results
    print('Points positions dataset loaded (matlab file dictionary)')
    print('Points positions dataset shape: ', pts_dataset.shape)
    print('Number of point sources per hologram: ', nb_point_sources)

    return hol_dataset, pts_dataset

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

def normalize(arr):
    '''
    Normalize an array of values.
    '''

    max_value = np.max(np.max(arr, axis=0))
    min_value = np.min(np.min(arr, axis=0))
    arr = (arr - min_value) / (max_value - min_value)

    return arr

def normalize_dataset(data):
    '''
    Normalize the dataset of holograms. Note that the holograms has 1 dimension
    (rows x columns).
    '''

    # Normalize the dataset
    data_norm = np.zeros([data.shape[0], data.shape[1]], dtype=complex)

    # Normalize each example
    for i in range(data.shape[0]):
        data_norm[i, :] = normalize(data[i, :])

    return data_norm

def compute_targets_array(nb_holograms, nb_class, nb_holograms_class):
    '''
    Compute the array of targets for the classification problem. Note that
    the number computed corresponds to the number of point sources in the
    hologram.
    '''

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
    '''
    Prepare the datasets (X and Y) to the classification problem.
    '''

    print('----- Data pre-procesing... -----')

    # Reshape the dataset
    print('Reshaping dataset...')
    data_r = reshape_dataset(data, nb_holograms)
    print('Reshaped dataset shape: ', data_r.shape)

    # Normalize the dataset
    print('\nNormalizing dataset...')
    data_norm = normalize_dataset(data_r)
    print('Normalized dataset shape: ', data_norm.shape)

    # Compute array of targets
    print('\nComputing array of targets...')
    y_array = compute_targets_array(nb_holograms, nb_class, nb_holograms_class)

    # Save matrix
    np.save('classification_problem/Y_array.npy', y_array)

    # Verify
    print('Y_array shape: ', y_array.shape)
    print(collections.Counter(y_array))
    print('Y_array saved in a .npy file!\n')

    return data_norm, y_array

def compute_targets_array_regression(data):
    '''
    Compute the array of targets for the regression problem by changing
    the dimensions of the values.
    '''

    # 3D position in (mm, mm, m)
    for i in range(data.shape[0]):
        data[i, 0] = data[i, 0] * 1000
        data[i, 1] = data[i, 1] * 1000

    return data

def pre_processing_regression(hol_dataset, pts_dataset):
    '''
    Prepara the datasets (hologram and points) to the regression problem.
    '''

    print('\n----- Data pre-procesing... -----')

    # Reshape the dataset
    print('Reshaping dataset...')
    data_r = reshape_dataset(hol_dataset, hol_dataset.shape[2])
    print('Reshaped dataset shape: ', data_r.shape)

    # Normalize the dataset
    print('\nNormalizing dataset...')
    data_norm = normalize_dataset(data_r)
    print('Normalized dataset shape: ', data_norm.shape)

    # Compute array of targets
    print('\nComputing array of targets...')
    # y_array = compute_targets_array_regression(pts_dataset)
    y_array = pts_dataset
    print('Y_array shape: ', y_array.shape)

    # Save matrix
    np.save('regression_problem/Y_array.npy', y_array)

    return data_norm, y_array

def split_dataset(perc, x_array, y_array, nb_holograms, nb_holograms_class):
    '''
    Split the dataset (features and targets) in a trainset and testset.
    '''
    print('----- Spliting dataset... -----')

    # Number of examples
    m = nb_holograms

    # Split our data in two subsets: training set and testing set
    m_train = int(m*perc)
    m_test = m - m_train

    print('Trainset: ' + str(perc*100) + '%, testset: ' + str(round((1 - perc), 1)*100) + ' %')

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
    print('Data : ', x_array.shape, y_array.shape)
    print('Train: ', x_train.shape, y_train.shape)
    print('Test : ', x_test.shape, y_test.shape)

    # Save files
    np.save('classification_problem/X_train.npy', x_train)
    np.save('classification_problem/Y_test.npy', y_test)
    np.save('classification_problem/X_test.npy', x_test)
    np.save('classification_problem/Y_train.npy', y_train)

    print('X_train, Y_train, X_test, Y_test saved in .npy files!\n')

def split_dataset_regression(perc, x_array, y_array):
    '''
    Split the dataset in a trainset and testset.
    '''

    print('\n----- Spliting dataset... -----')

    # Number of holograms
    m = x_array.shape[0]

    # Split our data in two subsets: training set and testing set
    m_train = int(m*perc)
    m_test = m - m_train

    print('Trainset: ' + str(perc*100) + '%, testset: ' + str(round((1 - perc), 1)*100) + ' %')

    # Training dataset
    x_train = np.zeros([m_train, x_array.shape[1]], dtype=complex)
    y_train = np.zeros((m_train, 3))

    x_train[:, :] = x_array[0:m_train, :]
    y_train[:, :] = y_array[0:m_train, :]

    # Testing set
    x_test = np.zeros([m_test, x_array.shape[1]], dtype=complex)
    y_test = np.zeros((m_test, 3))

    x_test[:, :] = x_array[m_train:len(x_array), :]
    y_test[:, :] = y_array[m_train:len(x_array), :]

    # Display results
    print('Data : ', x_array.shape, y_array.shape)
    print('Train: ', x_train.shape, y_train.shape)
    print('Test : ', x_test.shape, y_test.shape)

    # Save files
    np.save('regression_problem/X_train.npy', x_train)
    np.save('regression_problem/Y_train.npy', y_train)
    np.save('regression_problem/X_test.npy', x_test)
    np.save('regression_problem/Y_test.npy', y_test)

    print('X_train, Y_train, X_test, Y_test saved in .npy files!\n')

def main():

    # Compute execution time
    start_time = time.time()

    # Initial
    print('---------- [Prepare hologram dataset] ----------')

    # Choose an option
    option = int(input('Do you want to prepare the hologram dataset to the classication problem ' \
        '(5 classes) [1] or to the regression problem (1 source) [2] ?\n'))

    if option == 1:

        # Load hologram dataset
        hologram_dataset, nb_holograms, nb_class, nb_holograms_class = load_dataset()

        # Prepare dataset (reshape, normalize, compute target's array)
        x_array, y_array = pre_processing(hologram_dataset, nb_holograms, nb_class, \
            nb_holograms_class)

        # Split the dataset and save in .npy files
        perc = 0.8 # percentage

        split_dataset(perc, x_array, y_array, nb_holograms, nb_holograms_class)

    elif option == 2:

        # Load datasets
        hologram_dataset, points_dataset = load_datasets_regression()

        # Prepare dataset (reshape, normalize )
        x_array, y_array = pre_processing_regression(hologram_dataset, points_dataset)

        # Split dataset
        perc = 0.8 # percentage
        split_dataset_regression(perc, x_array, y_array)

    else:

        print('Invalid entry!')

    print('Done!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))
    print('Execution date: ' + str(dt.now()))

if __name__ == '__main__':
    main()
