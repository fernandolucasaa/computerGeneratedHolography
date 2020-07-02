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
    """
    Load the Matlab dictionary file and return the array as a numpy array.
    """

    # Read mat file dictionary
    dictionary = scipy.io.loadmat(file_path + file_name)

    # Access item of a dictionary
    array = dictionary[key]

    return array

def load_hologram_dataset(file_path):
    """
    Load the hologram dataset saved in a Matlab format. Note that it is
    a dictionary.
    """

    # File names
    file_name = 'hDataset.mat'
    key = 'hDataset'

    # Load dictionary
    data = load_matlab_dictionary(file_path, file_name, key)

    return data

def load_dataset():
    """
    Load the hologram dataset for the classification problem.
    """

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

    return hol_dataset, nb_holograms

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

def pre_processing(data, nb_holograms):
    """
    Prepare the datasets (X and Y) to the classification problem.
    """

    print('\n----- Data pre-procesing... -----')

    # Reshape the dataset
    print('Reshaping dataset...')
    data_r = reshape_dataset(data, nb_holograms)
    print('Reshaped dataset shape: ', data_r.shape)

    return data_r


def wigner_distribution_1d(array, seq_length, k_list):
    """
    Calculate the 1D pseudo-Wigner distribution, seq_length is the length in pixels of the
    operating windowsa and k_list is a list with the spatial frequencies.
    """
    # Array shape
    dim = array.shape[0] # (40000 = 200 x 200)

    # Determine the pixels to frame the array
    h = int(seq_length/2)
    N = seq_length

    # Determine the framing background array
    z_array = np.ones([dim + 2 * h], dtype=complex)
    
    # Insert array into the frame
    z_array[h:dim+h] = array

    # 1D Wigner distribution
    wigner = np.ones([dim, len(k_list)], dtype=complex)

    # Loop through the array
    for i in range(dim):

        # Adapt the position for the frame background array
        n_pixel = i + int(N/2)

        # Auxiliary variable (reset to zero)
        pos_y = 0

        # Loop through the spatial frequencies
        for k in k_list:

            # Spatial interval limits
            m_spatial_interval = np.arange(-int(N/2), int(N/2 - 1) + 1)

            # Computation
            summation = (z_array[n_pixel + m_spatial_interval] * np.conj(z_array[n_pixel - m_spatial_interval])) * \
                (np.exp(-1j * 2 * np.pi * k * (2*np.pi/N)))

            # Summation
            wigner[i, pos_y] = 2*sum(summation)
            pos_y += 1

    return wigner

def compute_wigner_distribution(data):
    """
    Compute the wigner distribution.
    """
    print('\n----- Computing 1D wigner distribution... -----')

    # Windows' length
    N = 9
    print('Window length: ' + str(N))

    # Spatial frequencies
    k_list = np.arange(-5, 5, 1) ##### ATTENTION!!!!!!!
    print('Spatial frequency array: ' + str(k_list))
    print('Spatial frequency array shape: ' + str(k_list.shape))

    # 1D wigner distribution
    wigner_distribution = np.zeros([data.shape[0], data.shape[1], len(k_list)], dtype=complex)
    print('Wigner distribution shape: ' + str(wigner_distribution.shape))

    print('\nComputing...')
    # for i in range(data.shape[0]):
    for i in range(10):
        if np.mod(i, 5) == 0:
            print('example ' + str(i))
        hol = data[i, :]
        wigner_distribution[i, :, :] = wigner_distribution_1d(hol, N, k_list)

    # Save .npy file
    # np.save('wigner_distribution.npy', wigner_distribution)

def main():

    # Compute execution time
    start_time = time.time()

    # Initial
    print('---------- [Compute wigner distribution dataset] ----------')

    # Load hologram dataset
    hologram_dataset, nb_holograms = load_dataset()

    # Prepare dataset (reshape, normalize, compute target's array)
    x_array = pre_processing(hologram_dataset, nb_holograms)

    # Compute the wigner distribution
    compute_wigner_distribution(x_array)

    print('\nDone!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))
    print('Execution date: ' + str(dt.now()))

if __name__ == '__main__':
    main()
