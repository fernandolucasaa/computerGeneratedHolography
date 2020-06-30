import time
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.utils import to_categorical

def load_data():
    '''
    Load the prepared datasets (reshaped, normalized, splited) in .npy format.
    '''
    x_train = np.load('classification_problem/X_train.npy')
    y_train = np.load('classification_problem/Y_train.npy')
    x_test = np.load('classification_problem/X_test.npy')
    y_test = np.load('classification_problem/Y_test.npy')

    return x_train, y_train, x_test, y_test

def create_model(nodes_1, dim_1, nodes_2, nb_class):
    '''
    Create a sequential model adding the layers.
    '''

    # Create the model
    model = Sequential()  # build a model layer by layer

    # Add model layers

    # First layer (hidden layer)
    model.add(Dense(nodes_1, input_dim=dim_1, activation='relu')) # Rectified linear unit (ReLU)

    # Second layer (hidden layer)
    model.add(Dense(nodes_2, activation='relu'))

    # Third layer (output layer)
    model.add(Dense(nb_class, activation='softmax'))

    return model

def categorical_target(nb_class, y_train, y_test):
    '''
    Convert an array of labeled data (targets) to one-hot vector.
    '''

    y_train = to_categorical(y_train, nb_class)
    y_test = to_categorical(y_test, nb_class)

    return y_train, y_test

def evaluate_model(model, x_train, y_train, x_test, y_test):
    '''
    Evaluate the model in accuracy terms.
    '''
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return train_acc, test_acc

def plot_history(history):
    '''
    Plot the training results (loss and accuracy)
    '''
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Training and validation loss/accuracy')

    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='test')
    ax1.set(ylabel='Loss')
    ax1.legend(loc='upper right')

    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='test')
    ax2.set(xlabel='Number of epochs', ylabel='Accuracy')
    ax2.legend(loc='upper left')

    # plt.show()
    fig.savefig('classification_problem/history.png')

def show_results(model, x_train, y_train, x_test, y_test):
    '''
    Display the model results (accuracy and model)
    '''

    # Evaluate the model
    train_acc, test_acc = evaluate_model(model, x_train, y_train, x_test, y_test)

    print('\n----- Accuracy -----')
    print('Train accuracy: %.2f%%, test accuracy: %.2f%%' % (train_acc*100, test_acc*100))

    # Sumarize model
    print('\n----- Model summary -----')
    model.summary()

def show_training_results(model, x_train, y_train, x_test, y_test, history):
    '''
    Display the training results (accuracy, summary, history).
    '''
    # Display the model results
    show_results(model, x_train, y_train, x_test, y_test)

    # Plot the history
    plot_history(history)

def save_model(model):
    '''
    Save the weights and the model separately. Note that it must re-compile
    the model when loaded.
    '''

    # Files
    file_model = 'classification_problem/model.json'
    file_weights = 'classification_problem/model.h5'

    # Serialize model to JSON
    model_json = model.to_json()
    with open(file_model, "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(file_weights)

def neural_network(x_train, y_train, x_test, y_test):
    '''
    Train a Multilayer Perceptron (MLP) to solve a classification problem, the number of
    point sources in the holograms.
    '''

    # Convert target classes to categorical ones
    nb_class = 5
    y_train, y_test = categorical_target(nb_class, y_train, y_test)

    # Parameters to create a model (number of nodes)
    nb_nodes_1 = 1000
    nb_nodes_2 = 400
    input_dim_1 = x_train.shape[1] # (40000 = 200x200)

    # 1. Create the model (add layers)
    model = create_model(nb_nodes_1, input_dim_1, nb_nodes_2, nb_class)

    # 2. Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Parameters to train the model
    nb_epochs = 50
    nb_batchs = 1000

    # 3. Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, \
        batch_size=nb_batchs, verbose=1)

    # Display training results
    show_training_results(model, x_train, y_train, x_test, y_test, history)

    # Save the model
    save_model(model)
    print('\nModel structure and weights saved!')

def load_model():
    '''
    Load the keras model.
    '''

    # Files
    file_model = 'classification_problem/model.json'
    file_weights = 'classification_problem/model.h5'

    # Load json file
    json_file = open(file_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Create model
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights(file_weights)

    return loaded_model

def load_trained_neural_network(x_train, y_train, x_test, y_test):
    '''
    Load and re-compile the trained neural network
    '''

    # Load the trained model
    loaded_model = load_model()

    # Re-compile model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Convert target classes to categorical ones
    nb_class = 5
    y_train, y_test = categorical_target(nb_class, y_train, y_test)

    # Display model results
    show_results(loaded_model, x_train, y_train, x_test, y_test)

def main():

    # Compute execution time
    start_time = time.time()

    # Inital
    print('---------- [Classification problem] ----------')

    # Choose an option
    option = int(input('Do you want to train the neural network (MLP) [1] or ' \
        'load the last trained model [2]?\n'))

    # Load .npy files
    print('\n----- Loading datasets... -----')

    # Load the .npy files (reshaped, normalized, splited)
    x_train, y_train, x_test, y_test = load_data()

    print('Datasets loaded')
    print('X_train: ' + str(x_train.shape) + ', Y_train: ' + str(y_train.shape))
    print('X_test: ' + str(x_test.shape) + ', Y_test: ' + str(y_test.shape))

    if option == 1:

        # Train the neural network (MLP)
        print('\n----- Neural network -----')
        neural_network(x_train, y_train, x_test, y_test)

    elif option == 2:

        # Load and re-compile the trained neural network (MLP)
        print('\n----- Neural network -----')
        load_trained_neural_network(x_train, y_train, x_test, y_test)

    else:

        print('Invalid entry!')

    print('\nDone!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))

if __name__ == '__main__':
    main()
