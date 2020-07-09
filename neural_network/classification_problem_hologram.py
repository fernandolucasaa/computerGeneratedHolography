"""
Script to train or load a model to a classification problem. The classification problem
is identify the number of point sources in a hologram dataset.
"""

import os
import time
from datetime import datetime as dt
import logging
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

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
# formatter = logging.Formatter('%(message)s')
# file_name = 'classification_problem/output_' + str(script_name[0:len(script_name)-3]) + '.log'
# file_handler = logging.FileHandler(file_name)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

def load_data():
    """
    Load the prepared datasets (reshaped, normalized, splited) in .npy format.
    """
    x_train = np.load('classification_problem/X_train.npy')
    y_train = np.load('classification_problem/Y_train.npy')
    x_test = np.load('classification_problem/X_test.npy')
    y_test = np.load('classification_problem/Y_test.npy')

    return x_train, y_train, x_test, y_test

def create_model(nodes_1, dim_1, nodes_2, nb_class):
    """
    Create a sequential model adding the layers.
    """

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
    """
    Convert an array of labeled data (targets) to one-hot vector.
    """

    y_train = to_categorical(y_train, nb_class)
    y_test = to_categorical(y_test, nb_class)

    return y_train, y_test

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Evaluate the model in accuracy terms.
    """
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return train_acc, test_acc

def plot_history(history):
    """
    Plot the training results (loss and accuracy)
    """
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

def save_summary_to_file(message):
    """
    Save the model summary once the logging does not save.
    """
    f_name = 'classification_problem/summary.txt'
    with open(f_name, 'a') as f_out:
        print(message, file=f_out)

def show_results(model, x_train, y_train, x_test, y_test):
    """
    Display the model results (accuracy and model)
    """

    # Evaluate the model
    train_acc, test_acc = evaluate_model(model, x_train, y_train, x_test, y_test)

    logger.debug('\n----- Accuracy -----')
    logger.debug('Train accuracy: %.2f%%, test accuracy: %.2f%%' % (train_acc*100, test_acc*100))

    # Sumarize model
    logger.debug('\n----- Model summary -----')
    model.summary()
    model.summary(print_fn=save_summary_to_file)

def show_training_results(model, x_train, y_train, x_test, y_test, history):
    """
    Display the training results (accuracy, summary, history).
    """
    # Display the model results
    show_results(model, x_train, y_train, x_test, y_test)

    # Plot the history
    plot_history(history)

def save_model(model):
    """
    Save the weights and the model separately. Note that it must re-compile
    the model when loaded.
    """

    # Files
    file_model = 'classification_problem/model.json'
    file_weights = 'classification_problem/model.h5'

    # Serialize model to JSON
    model_json = model.to_json()
    with open(file_model, "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(file_weights)

def predict_results(model, data):
    """
    Make predictions with the model trained in the a dataset and verify
    the accuracy of the predictions for each class.
    """

    # Make the predictions
    predictions = model.predict_classes(data)

    # Number of holograms
    nb_holograms = data.shape[0]

    # Number of classes
    nb_class = 5

    # Number of holograms per class
    nb_holograms_class = int(nb_holograms / nb_class)

    # Array for the predictions results per class
    results_class = np.zeros([nb_class, nb_holograms_class])

    # Auxiliary variables positions
    init = 0
    fin = int(nb_holograms / nb_class - 1)

    for i in range(nb_class):
        results_class[i, :] = predictions[init:(fin+1)]
        init = fin + 1
        fin = fin + nb_holograms_class

    # Display results
    counter = np.zeros((nb_class))

    for i in range(nb_class):
        counter[0] = np.count_nonzero(results_class[i, :] == 0)
        counter[1] = np.count_nonzero(results_class[i, :] == 1)
        counter[2] = np.count_nonzero(results_class[i, :] == 2)
        counter[3] = np.count_nonzero(results_class[i, :] == 3)
        counter[4] = np.count_nonzero(results_class[i, :] == 4)
        logger.debug('\n- Predictions for class ' + str(i) + ', accuracy: ' \
            '{:.2f}'.format((counter[i] / nb_holograms_class)  * 100) + '%')
        logger.debug('[0]: ' + str(counter[0]) + ', [1]: ' + str(counter[1]) + ', [2]: ' \
            + str(counter[2]) + ', [3]: ' + str(counter[3]) + ', [4]: ' + str(counter[4]))

def predict(model, train, test):
    """
    Make predictions with the model trianed in the train dateset and test dataset.
    """

    logger.debug('\n----- Predictions -----')

    logger.debug('\nTrain predictions:')
    predict_results(model, train)

    logger.debug('\nTest predictions:')
    predict_results(model, test)

def callback_functions():
    """
    Create callback functions to monitore the training procedure.
    """

    # Stops the training when there is not improvement in the validations loss for some
    # consectuve epochs and keeps the best weghts once stopped
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, \
        min_delta=0, restore_best_weights=True)

    # Save the log in a file
    # file_name = "classification-hologram" + str(dt.now())
    path_logs = os.getcwd() + '/classification_problem/logs'

    # Measures and visualizes the accuracy and the loss
    tensor = TensorBoard(log_dir=path_logs, histogram_freq=0, write_graph=0, \
        write_images=False)

    cb_list = [early_stop, tensor]

    return cb_list

def neural_network(x_train, y_train, x_test, y_test):
    """
    Train a Multilayer Perceptron (MLP) to solve a classification problem, the number of
    point sources in the holograms.
    """

    # Convert target classes to categorical ones
    nb_class = 5
    y_train, y_test = categorical_target(nb_class, y_train, y_test)

    # Parameters to create a model (number of nodes)
    nb_nodes_1 = 7000
    nb_nodes_2 = 700
    input_dim_1 = x_train.shape[1] # (40000 = 200x200)

    # 1. Create the model (add layers)
    model = create_model(nb_nodes_1, input_dim_1, nb_nodes_2, nb_class)

    # 2. Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Parameters to train the model
    nb_epochs = 50
    nb_batchs = 1000

    # Create the callbacks functions
    cb_list = callback_functions()

    # 3. Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, \
        batch_size=nb_batchs, callbacks=cb_list, verbose=1)

    # Display training results
    show_training_results(model, x_train, y_train, x_test, y_test, history)

    # Save the model
    save_model(model)
    logger.debug('\nModel structure and weights saved!')

    # Display predictions results
    predict(model, x_train, x_test)

def load_model():
    """
    Load the keras model.
    """

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
    """
    Load and re-compile the trained neural network
    """

    # Load the trained model
    loaded_model = load_model()

    # Re-compile model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Convert target classes to categorical ones
    nb_class = 5
    y_train, y_test = categorical_target(nb_class, y_train, y_test)

    # Display model results
    show_results(loaded_model, x_train, y_train, x_test, y_test)

    # Display predictions results
    predict(loaded_model, x_train, x_test)

def main():

    # Compute execution time
    start_time = time.time()

    # Inital
    logger.debug('---------- [Classification problem] ----------')

    # Choose an option
    option = int(input('Do you want to train the neural network (MLP) [1] or ' \
        'load the last trained model [2]?\n'))

    # Load .npy files
    logger.debug('\n----- Loading datasets... -----')

    # Load the .npy files (reshaped, normalized, splited)
    x_train, y_train, x_test, y_test = load_data()

    logger.debug('Datasets loaded')
    logger.debug('X_train: ' + str(x_train.shape) + ', Y_train: ' + str(y_train.shape))
    logger.debug('X_test: ' + str(x_test.shape) + ', Y_test: ' + str(y_test.shape))

    if option == 1:

        # Train the neural network (MLP)
        logger.debug('\n----- Neural network -----')
        neural_network(x_train, y_train, x_test, y_test)

    elif option == 2:

        # Load and re-compile the trained neural network (MLP)
        logger.debug('\n----- Neural network -----')
        load_trained_neural_network(x_train, y_train, x_test, y_test)

    else:

        logger.debug('Invalid entry!')

    logger.debug('\nDone!')
    logger.debug('Execution time: %.4f seconds' % (time.time() - start_time))
    logger.debug('Execution date: ' + str(dt.now()))

if __name__ == '__main__':
    main()
