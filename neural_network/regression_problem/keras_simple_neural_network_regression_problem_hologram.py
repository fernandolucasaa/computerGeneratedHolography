from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io
import collections
import pandas as pd

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data():
    '''
    Load the datasets in .npy format.
    '''
    x_train = np.load('X_train.npy')
    y_train = np.load('Y_train.npy')

    x_test = np.load('X_test.npy')
    y_test = np.load('Y_test.npy')

    return x_train, y_train, x_test, y_test

def normalize(arr):

    max_value = np.max(np.max(arr, axis=0))
    min_value = np.min(np.min(arr, axis=0))
    arr = (arr - min_value) / (max_value - min_value)

    return arr

def normalize_dataset(data_1D):

    # Normalize the examples
    data_1D_norm = np.zeros([data_1D.shape[0], data_1D.shape[1]], dtype=complex)

    # Normalize each example
    for i in range(data_1D.shape[0]):
        data_1D_norm[i, :] = normalize(data_1D[i, :])

    return data_1D_norm

def pre_processing(x_train, x_test):

    print('\n----- Data pre-procesing... -----')

    # Normalize dataset
    print('Normalizing dataset...')
    x_train_norm = normalize_dataset(x_train)
    x_test_norm = normalize_dataset(x_test)

    print('Dataset normalized')
    print('x_train_norm : ' + str(x_train_norm.shape))
    print('x_test_norm: ' + str(x_test_norm.shape))

    return x_train_norm, x_test_norm

def create_model(nb_nodes1, input_dim1, nb_nodes2):
    '''
    Create the sequantial model and addded the layers.
    '''

    # Create the model
    model = Sequential() # build a model layer by layer

    # Add model layers

    # First layer (hidden layer)
    model.add(Dense(nb_nodes1, kernel_initializer='normal', input_dim=input_dim1, activation='relu'))

    # Second layer (hidden layer)
    model.add(Dense(nb_nodes2, kernel_initializer='normal', activation='relu'))

    # Third layer (output layer)
    model.add(Dense(3, kernel_initializer='normal', activation='linear'))

    return model

def plot_history(history):
    '''
    Plot the training results (loss and accuracy)
    '''
    plt.title('Training and validation loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Number of epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    plt.savefig('history.png')

    return

def show_results(model, history):
    
    # Sumarize model
    print('\n----- Model summary -----')
    model.summary()

    # Plot history
    plot_history(history)

def save_model(model):
    '''
    Save the weights and the model separately. Note that it must re-compile
    the model when loaded.
    '''
    
    # Files
    file_model = 'model.json'
    file_weights = 'model.h5'

    # Serialize model to JSON
    model_json = model.to_json()
    with open(file_model, "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(file_weights)

    return

def neural_network(x_train, y_train, x_test, y_test):
    '''
    Train a Multilayer Perceptron to solve a classification problem, the number of
    ponctual sources in the holograms.
    '''

    # Parameters to create the model
    nb_nodes_1 = 1000
    nb_nodes_2 = 400
    input_dim_1 = x_train.shape[1] # 40000 (200x200)
    
    # 1. Create the model (add layers)
    model = create_model(nb_nodes_1, input_dim_1, nb_nodes_2)

    # 2. Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # Parameters to train the model
    nb_epochs = 2

    # 3. Train the model 
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, verbose=1)    

    # Display the results
    show_results(model, history)

    # Save the model
    save_model(model)

def load_model():
    '''
    Load the keras model
    '''

    file_model = 'model.json'
    file_weights = 'model.h5'

    # Load json file
    json_file = open(file_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Create model
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights(file_weights)

    return loaded_model

def load_trained_neural_network():

    '''
    Load and re-compile the trained neural network
    '''

    # Load the trained model
    loaded_model = load_model()

    # Re-compile model
    loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # Sumarize model
    print('\n----- Model summary -----')
    loaded_model.summary()

    return 

def main():

    # Compute execution time
    start_time = time.time()

    # Inital
    print('---------- [Regression problem] ----------')


    # Choose an option
    option = int(input('Do you want to train the neural network [1] or load the \
        last trained model [2]?\n'))

    if option == 1:

        # Load .npy files
        print('\n----- Loading datasets... -----')

        X_train, Y_train, X_test, Y_test = load_data()

        print('Datasets loaded')
        print('X_train: ' + str(X_train.shape) + ', Y_train: ' + str(Y_train.shape))
        print('X_test: ' + str(X_test.shape) + ', Y_test: ' + str(Y_test.shape))

        # Train the neural network (MLP)
        print('\n----- Neural network -----')
        neural_network(X_train, Y_train, X_test, Y_test)

    elif option == 2:

        # Load and complile trained neural network (MLP)
        print('\n----- Neural network -----')
        load_trained_neural_network()

    else:

        print('Invalid entry!')

    print('\nDone!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))
    print('-----------------------------------------')

    return

if __name__ == '__main__':
    main()