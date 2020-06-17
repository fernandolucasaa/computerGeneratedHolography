import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import collections

def load_matlab_dictionary(file_path, file_name, key):
   	
	# Read mat file dictionary
	dictionary = scipy.io.loadmat(file_path + file_name)
	
	# Access item of a dictionary
	array = dictionary[key]
	
	return array

def load_hologram_dataset(path, file_path, file_name, key):

	# Load dictionary
	dat = load_matlab_dictionary(file_path, file_name, key)

	# Number of holograms
	nb_holograms = dat.shape[2]

	# Number of class
	nb_class = 5

	# Number of holograms per class
	nb_holograms_class = int(nb_holograms/nb_class)

	# Save npy file
	np.save('dat.npy', dat)

	return dat, nb_holograms, nb_class, nb_holograms_class

def load_dataset():

	# Path
	path = 'C:\\Users\\flucasamar\\Desktop\\Github\\computerGeneratedHolography\\'
	file_path = path + '\\output\\dataset\\'

	# File names
	file_name = 'hDataset.mat'
	key = 'hDataset'

	# Load hologram (matfile dictionary)
	dat, nb_holograms, nb_class, nb_holograms_class = load_hologram_dataset(path, file_path, file_name, key)

	# Display results
	print('----- Load hologram dataset (matlab file dictionary) -----')
	print('Hologram shape: ', dat.shape)
	print('Total number of holograms: ' + str(nb_holograms))
	print('Number of holograms per class: ' + str(nb_holograms_class))
	print('Hologram dataset saved in .npy file')

	return

def reshape_dataset(dat, nb_holograms):
	
	# Dimensions
	rows = dat.shape[0]
	columns = dat.shape[1]

	# Reshape the dataset so that the first dimension is the number of holograms
	dat_r = np.ones([nb_holograms, rows, columns], dtype = complex)
	for i in range(nb_holograms):
		dat_r[i,:,:] = dat[:,:,i]

	# Reshape the dataset to 1 dimension
	data_1D = np.reshape(dat_r, (1500, int(rows*columns)))

	return data_1D

def normalize(arr):

	max_value = np.max(np.max(arr, axis = 0))
	min_value = np.min(np.min(arr, axis = 0))
	arr = (arr - min_value) / (max_value - min_value)
	
	return arr

def normalize_dataset(nb_holograms, data_1D):

	# Normalize the examples
	data_1D_norm = np.zeros([nb_holograms, data_1D.shape[1]], dtype = complex)

	# Normalize each example
	for i in range(nb_holograms):
		data_1D_norm[i,:] = normalize(data_1D[i,:])

	return data_1D_norm

def compute_targets_array(nb_holograms, nb_class, nb_holograms_class):

	# Compute array of targets
	Y_array = np.ones([nb_holograms,])

	pos = 0
	for c in range(nb_class):
		for h in range(nb_holograms_class):
			Y_array[pos] = c
			pos = pos + 1
	
	# Save matrix
	np.save('Y_array.npy', Y_array)

	return Y_array

def split_dataset(perc, data_1D_norm, nb_holograms, nb_holograms_class, Y_array):

	# Dataset
	X_array = data_1D_norm

	# Number of examples
	m = nb_holograms

	# Split our data in two subsets: training set and testing set
	m_train = int(m*perc)
	m_test = m - m_train

	X_train = np.zeros([m_train, data_1D_norm.shape[1] ], dtype = complex)
	Y_train = np.zeros((m_train, ))

	X_test = np.zeros([m_test, data_1D_norm.shape[1]], dtype = complex)
	Y_test = np.zeros((m_test, ))

	# Auxiliary variables
	counter = 1
	pos_train = 0
	pos_test = 0

	# Number of holograms per class in trainset
	nb_holograms_class_train = int(0.8*nb_holograms_class)

	# Split the data
	for i in range(m):
		if (counter <= nb_holograms_class_train):
			X_train[pos_train,:] = X_array[i,:]
			Y_train[pos_train] = Y_array[i]
			pos_train = pos_train + 1
		else:
			X_test[pos_test,:] = X_array[i,:]
			Y_test[pos_test] = Y_array[i]
			pos_test = pos_test + 1
		if (counter == nb_holograms_class):
			counter = 1
		else:
			counter = counter + 1

	return X_train, Y_train, X_test, Y_test

def pre_processing():

	# Reshape the dataset to 1 dimension
	data_1D = reshape_dataset(dat, nb_holograms)
	print('Dataset 1D shape: ', data_1D.shape)

	# Normalize the data
	data_1D_norm = normalize_dataset(nb_holograms, data_1D)
	print('Normalized dataset shape: ', data_1D_norm.shape)

	# Compute array of targets
	Y_array = compute_targets_array(nb_holograms, nb_class, nb_holograms_class)

	# Verify
	print('Y_array shape: ', Y_array.shape)
	print(collections.Counter(Y_array))

	return

if __name__ == '__main__':

	load_dataset()

	pre_processing()
	
	