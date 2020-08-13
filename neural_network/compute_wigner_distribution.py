"""
Scrit to compute the 1D peusdo wigner distribution in an one-dimensional hologram
dataset.
"""

import os
import time
import logging
from datetime import datetime as dt
from pathlib import Path
import scipy.io
import numpy as np

import multiprocessing as mp

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

    logger.debug('\n----- Loading hologram dataset... -----')

    # File path
    # file_path = path + '\\output\\dataset\\oneClass\\'
    file_path = path + '\\output\\dataset\\'

    # Load hologram (matfile dictionary)
    hol_dataset = load_hologram_dataset(file_path)

    # Number of holograms
    nb_holograms = hol_dataset.shape[2]

    # Number of classes
    nb_class = 1

    # Number of holograms per class
    nb_holograms_class = int(nb_holograms/nb_class)

    # Save npy file
    # np.save('classification_problem/hologram_dataset.npy', hol_dataset)

    # Display results
    logger.debug('Hologram dataset loaded (matlab file dictionary)')
    logger.debug('Hologram dataset shape: ' + str(hol_dataset.shape))
    logger.debug('Total number of holograms: ' + str(nb_holograms))
    logger.debug('Number of holograms per class: ' + str(nb_holograms_class))
    # logger.debug('Hologram dataset saved in .npy file!\n')

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

    logger.debug('\n----- Data pre-procesing... -----')

    # Reshape the dataset
    logger.debug('Reshaping dataset...')
    data_r = reshape_dataset(data, nb_holograms)
    logger.debug('Reshaped dataset shape: ' + str(data_r.shape))

    return data_r

def pseudo_wigner_distribution(z_array, n_pixel, k_freq, N):
    """
    Compute the 1D pseudo-Wigner distribution (PWD). The WD has been mathematically
    defined as:

        W(n, k) = 2 . summation {m = -N/2, N/2} (z(n + m) . z*(n - m) . exp(-i.2.pi.k.(2.m/N)))

    where, the variable z(n) represents the gray value of pixel n in a given image z. Here z*
    indicates the complex-conjugate of signal z. The sum is limited to a spatial interval
    (-N/2, N/2-1). In the equation, n and k represent the space and frequency discrete variables
    respectively and m is a shifting parameter, which is also discrete. Note that we are performing
    the computation through the local frequency, so it's a simplification.
    """

    # Spatial interval
    m_spatial_interval = np.arange(-int(N/2), int(N/2 - 1) + 1)

    # Computation
    summation = z_array[n_pixel + m_spatial_interval] * np.conj(z_array[n_pixel - m_spatial_interval]) * \
        np.exp(-1j * 2 * np.pi * k_freq * (2 * np.pi / N))

    # Summation
    return 2*sum(summation)

def wigner_distribution_1d(array, seq_length, k_list):
    """
    Compute the 1D pseudo-Wigner distribution of a 1D array, seq_length is the length in pixels
    of the operating window and k_list is a list with the spatial frequencies.
    """
    # Array shape
    size = array.shape[0] # (40000 = 200 x 200)

    # Determine the pixels to frame the array
    h_len = int(seq_length/2)

    # Framing background array
    z_array = np.ones([size + 2*h_len], dtype=complex)
    z_array[h_len : size+h_len] = array

    # 1D Wigner distribution
    wigner = np.ones([size, len(k_list)], dtype=complex)

    # Loop through the array
    for i in range(size):

        # Adapt the position for the frame background array
        n_pixel = i + h_len

        # Auxiliary variable (reset to zero)
        pos_y = 0

        # Loop through the spatial frequencies
        for k in k_list:
            wigner[i, pos_y] = pseudo_wigner_distribution(z_array, n_pixel, k, seq_length)
            pos_y += 1

    return wigner

def wigner_distribution_1d_opt(array, seq_length, k_list):
    """
    Compute the 1D pseudo-Wigner distribution of a 1D array, seq_length is the length in pixels
    of the operating window and k_list is a list with the spatial frequencies.
    """
    # Array shape
    size = array.shape[0] # (40000 = 200 x 200)

    # Determine the pixels to frame the array
    h_len = int(seq_length/2)

    # Framing background array
    z_array = np.ones([size + 2*h_len], dtype=complex)
    z_array[h_len : size+h_len] = array

    # 1D Wigner distribution
    wigner = np.ones([size, len(k_list)], dtype=complex)

    # Loop through the array
    for i in range(size):

        # Adapt the position for the frame background array
        n_pixel = i + h_len

        # Auxiliary variable (reset to zero)
        pos_y = 0

        # Init Pool Class
        pool = mp.Pool(mp.cpu_count())
        
        result = [pool.apply(pseudo_wigner_distribution, args=(z_array, n_pixel, k, seq_length)) for k in k_list]
        
        # # Loop through the spatial frequencies
        # for k in k_list:
        #     wigner[i, pos_y] = pseudo_wigner_distribution(z_array, n_pixel, k, seq_length)
        #     pos_y += 1

    return wigner

def compute_wigner_distribution(data):
    """
    Compute the wigner distribution of the dataset.
    """
    logger.debug('\n----- Computing 1D wigner distribution... -----')

    # Window length
    window_len = 9
    logger.debug('Window length: ' + str(window_len))

    # Spatial frequencies
    k_list = np.arange(-4, 4, 1) # I DID NOT UNDERSTAND ???

    logger.debug('Spatial frequency array: ' + str(k_list))
    logger.debug('Spatial frequency array shape: ' + str(k_list.shape))

    lim = 500

    # 1D wigner distribution
    wigner_distribution = np.zeros([lim, data.shape[1], len(k_list)], dtype=complex)
    # wigner_distribution = np.zeros([data.shape[0], data.shape[1], len(k_list)], dtype=complex)
    logger.debug('Wigner distribution shape: ' + str(wigner_distribution.shape))

    logger.debug('\nComputing...')
    # for i in range(data.shape[0]):
    
    for i in range(lim):
        logger.debug(i)
        if np.mod(i, 5) == 0:
            logger.debug('example ' + str(i))
        hol_1d = data[i, :]
        # wigner_distribution[i, :, :] = wigner_distribution_1d_opt(hol_1d, window_len, k_list)
        wigner_distribution[i, :, :] = wigner_distribution_1d(hol_1d, window_len, k_list)

    # Save .npy file
    np.save('wigner_distribution/wigner_distribution.npy', wigner_distribution)

    logger.debug('Wigner distribution saved in .npy file!')

def main():
    """
    Compute the 1D wigner distribution.
    """

    # Compute execution time
    start_time = time.time()

    # Initial
    logger.debug('---------- [Compute wigner distribution dataset] ----------')

    # Load hologram dataset
    hologram_dataset, nb_holograms = load_dataset()

    # Prepare dataset (reshape, normalize, compute target's array)
    x_array = pre_processing(hologram_dataset, nb_holograms)

    # Compute the wigner distribution
    compute_wigner_distribution(x_array)

    logger.debug('\nDone!')
    logger.debug('Execution time: %.4f seconds' % (time.time() - start_time))
    logger.debug('Execution date: ' + str(dt.now()))

if __name__ == '__main__':
    main()
