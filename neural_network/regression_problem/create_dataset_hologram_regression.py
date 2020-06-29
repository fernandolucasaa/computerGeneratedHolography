from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io
import collections
import pandas as pd


def load_matlab_dictionary(file_path, file_name, key):

    # Read mat file dictionary
    dictionary = scipy.io.loadmat(file_path + file_name)

    # Access item of a dictionary
    array = dictionary[key]

    return array

def load_hologram_dataset(path):
    
    # File path
    file_path = path + '\\output\\dataset\\oneClass\\'
    
    # File names
    file_name = 'hDataset.mat'
    key = 'hDataset'
    
    # Load dictionary
    data = load_matlab_dictionary(file_path, file_name, key)
    
    return data

def load_points_dataset(path):
    
    # File path
    file_path = path + '\\output\\dataset\\oneClass\\'
    
    # File names
    file_name = 'pDataset.mat'
    key = 'pDataset'
    
    # Load dictionary
    data = load_matlab_dictionary(file_path, file_name, key)
    
    return data

def load_datasets():
    
    # Current directory
    cwd = os.getcwd()

    # Directory path
    path = str((Path(cwd).parent).parent)

    print('----- Loading hologram dataset... -----')
    
    # Load hologram (matfile dictionary)
    hologram_dataset = load_hologram_dataset(path)
    
    # Number of holograms
    nb_holograms = hologram_dataset.shape[2]
    
    # Display results
    print('Hologram dataset loaded (matlab file dictionary)')
    print('Hologram dataset shape: ', hologram_dataset.shape)
    print('Total number of holograms: ' + str(nb_holograms))
    
    print('\n----- Loading points positions dataset... -----')
    
    # Load points positions (matfile dictionary)
    points_dataset = load_points_dataset(path)
    
    # Number of point sources per hologram
    nb_point_sources = int(points_dataset.shape[0]/nb_holograms)
    
    # Display results
    print('Points positions dataset loaded (matlab file dictionary)')
    print('Points positions dataset shape: ', points_dataset.shape)
    print('Number of point sources per hologram: ', nb_point_sources)
    
    return hologram_dataset, points_dataset

def reshape_dataset(data, nb_holograms):

    # Dimensions
    rows = data.shape[0]
    columns = data.shape[1]

    # Reshape the dataset so that the first dimension is the number of holograms
    data_r = np.ones([nb_holograms, rows, columns], dtype=complex)
    
    for i in range(nb_holograms):
        data_r[i, :, :] = data[:, :, i]

    # Reshape the dataset to 1 dimension
    data_1D = np.reshape(data_r, (nb_holograms, int(rows*columns)))

    return data_1D

def compute_targets_array(data): # FIX, BUG

    # All the positions (x, y, z) in meters

    for i in range(data.shape[0]):
        data[i, 0] = data[i, 0] * 1000
        data[i, 1] = data[i, 1] * 1000

    return data

def pre_processing(hol_dataset, points_dataset):
    
    print('\n----- Data pre-procesing... -----')

    # Reshape the dataset to 1 dimension
    print('Reshaping hologram dataset to 1 dimension...')
    data_1D = reshape_dataset(hol_dataset, hol_dataset.shape[2])
    print('Dataset 1D shape: ', data_1D.shape)
    
    # Compute array of targets
    print('Computing Y_array...')
    #Y_array = compute_targets_array(points_dataset)
    Y_array = points_dataset
    print('Y_array shape: ', Y_array.shape)

    return data_1D, Y_array

def split_dataset(perc, x_array, y_array):

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
    np.save('X_train.npy', x_train)
    np.save('Y_train.npy', y_train)
    np.save('X_test.npy', x_test)
    np.save('Y_test.npy', y_test)
    
    print('X_train, Y_train, X_test, Y_test saved in .npy files!\n')
    
    return x_train, y_train, x_test, y_test

def main():

    # Compute execution time
    start_time = time.time()

    # Load datasets
    hologram_dataset, points_dataset = load_datasets()

    # Prepare dataset (reshape)
    X_array, Y_array = pre_processing(hologram_dataset, points_dataset)

    # Split dataset
    perc = 0.8
    X_train, Y_train, X_test, Y_test = split_dataset(perc, X_array, \
        Y_array)

    print('Done!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))

    return

if __name__ == '__main__':
    main()