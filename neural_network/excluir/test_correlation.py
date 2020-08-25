# # generate related variables
# from numpy import mean
# from numpy import std
# from numpy.random import randn
# from numpy.random import seed
# from matplotlib import pyplot
# # seed random number generator
# seed(1)
# # prepare data
# data1 = 20 * randn(1000) + 100
# data2 = data1 + (10 * randn(1000) + 50)
# # summarize
# print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
# print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# # plot
# pyplot.scatter(data1, data2)
# pyplot.show()

# # calculate the covariance between two variables
# from numpy.random import randn
# from numpy.random import seed
# from numpy import cov
# # seed random number generator
# seed(1)
# # prepare data
# data1 = 20 * randn(1000) + 100
# data2 = data1 + (10 * randn(1000) + 50)
# # calculate covariance matrix
# covariance = cov(data1, data2)
# print(covariance)

# # calculate the Pearson's correlation between two variables
# from numpy.random import randn
# from numpy.random import seed
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from numpy import cov
# # seed random number generator
# seed(1)
# # prepare data
# data1 = 20 * randn(1000) + 100
# print(data1.shape)
# data2 = data1 + (10 * randn(1000) + 50)
# print(data2.shape)
# # calculate Pearson's correlation
# corr, _ = pearsonr(data1, data2)
# print('Pearsons correlation: %.3f' % corr)

import os
import time
from datetime import datetime as dt
from pathlib import Path
import collections
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print('Hologram dataset saved in .npy file!\n')

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

def main():
    hol_dataset, nb_holograms, nb_class, nb_holograms_class = load_dataset()

    data_r = reshape_dataset(hol_dataset, nb_holograms)
    print(data_r.shape)

    data1 = data_r[0, :]
    data2 = data_r[1, :]

    # calculate covariance matrix
    covariance = cov(data1, data2)
    print(covariance)

    # calculate Pearson's correlation
    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: %.3f' % corr)

    # calculate spearman's correlation
    corr, _ = spearmanr(data1, data2)
    print('Spearmans correlation: %.3f' % corr)


    df = pd.DataFrame(data_r[0:10, :])

    corr_fct = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_fct,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(df.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.show()


if __name__ == '__main__':
    main()