import time
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Flatten
'''
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
'''
def load_data():
    '''
    Load the datasets in .npy format (reshaped, normalizedm splited).
    '''
    x_train = np.load('regression_problem/X_train.npy')
    y_train = np.load('regression_problem/Y_train.npy')
    x_test = np.load('regression_problem/X_test.npy')
    y_test = np.load('regression_problem/Y_test.npy')

    return x_train, y_train, x_test, y_test

def create_model(nodes_1, dim_1, nodes_2):
    '''
    Create the sequantial model adding the layers.
    '''

    # Create the model
    model = Sequential() # build a model layer by layer

    # Add model layers

    # First layer (hidden layer)
    model.add(Dense(nodes_1, kernel_initializer='normal', input_dim=dim_1, activation='relu'))

    # Second layer (hidden layer)
    model.add(Dense(nodes_2, kernel_initializer='normal', activation='relu'))

    # Third layer (output layer)
    model.add(Dense(3, kernel_initializer='normal', activation='linear'))

    return model

def plot_history(history):
    '''
    Plot the training results (loss)
    '''
    plt.title('Training and validation loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Number of epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    plt.savefig('regression_problem/history.png')

def show_results(model, history):
    '''
    Display the results (summary, history)
    '''

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
    file_model = 'regression_problem/model.json'
    file_weights = 'regression_problem/model.h5'

    # Serialize model to JSON
    model_json = model.to_json()
    with open(file_model, "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(file_weights)

def neural_network(x_train, y_train, x_test, y_test):
    '''
    Train a Multilayer Perceptron (MLP) to solve a regression problem, the positions of the
    sources in the 3D scene.
    '''

    # Parameters to create the model (number of nodes)
    nb_nodes_1 = 1000
    nb_nodes_2 = 400
    input_dim_1 = x_train.shape[1] # (40000 = 200x200)

    # 1. Create the model (add layers)
    model = create_model(nb_nodes_1, input_dim_1, nb_nodes_2)

    # 2. Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # Parameters to train the model
    nb_epochs = 20
    nb_batchs = 1000

    # 3. Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, \
        batch_size=nb_batchs, verbose=1)

    # Display the results
    show_results(model, history)

    # Save the model
    save_model(model)
    print('\nModel structure and weights saved!')

def load_model():
    '''
    Load the keras model
    '''

    file_model = 'regression_problem/model.json'
    file_weights = 'regression_problem/model.h5'

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

def main():

    # Compute execution time
    start_time = time.time()

    # Inital
    print('---------- [Regression problem] ----------')


    # Choose an option
    option = int(input('Do you want to train the neural network (MLP) [1] or load the ' \
        'last trained model [2]?\n'))

    if option == 1:

        # Load .npy files
        print('\n----- Loading datasets... -----')

        x_train, y_train, x_test, y_test = load_data()

        print('Datasets loaded')
        print('X_train: ' + str(x_train.shape) + ', Y_train: ' + str(y_train.shape))
        print('X_test: ' + str(x_test.shape) + ', Y_test: ' + str(y_test.shape))

        # Train the neural network (MLP)
        print('\n----- Neural network -----')
        neural_network(x_train, y_train, x_test, y_test)

    elif option == 2:

        # Load and complile trained neural network (MLP)
        print('\n----- Neural network -----')
        load_trained_neural_network(x_train, y_train, x_test, y_test)

    else:

        print('Invalid entry!')

    print('\nDone!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))
    print('-----------------------------------------')

if __name__ == '__main__':
    main()
