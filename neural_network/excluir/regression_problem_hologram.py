"""
Script to train or load a model to a regression problem. The regression problem
is identify the positions x, y, z of the point sources in a hologram dataset.
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
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

from sklearn.metrics import mean_squared_error
"""
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
"""

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
# file_name = 'regression_problem/output_' + str(script_name[0:len(script_name)-3]) + '.log'
# file_handler = logging.FileHandler(file_name)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

def load_data():
    """
    Load the datasets in .npy format (reshaped, normalizedm splited).
    """
    x_train = np.load('regression_problem/X_train.npy')
    y_train = np.load('regression_problem/Y_train.npy')
    x_test = np.load('regression_problem/X_test.npy')
    y_test = np.load('regression_problem/Y_test.npy')

    return x_train, y_train, x_test, y_test

def create_model(nodes_1, dim_1, nodes_2):
    """
    Create the sequantial model adding the layers.
    """

    # Create the model
    model = Sequential() # build a model layer by layer

    # Add model layers

    # First layer (hidden layer)
    model.add(Dense(nodes_1, kernel_initializer='normal', input_dim=dim_1, activation='relu'))

    # Second layer (hidden layer)
    model.add(Dense(nodes_2, kernel_initializer='normal', activation='relu'))

    # Third layer (output layer)
    # model.add(Dense(3, kernel_initializer='normal', activation='linear'))
    # model.add(Dense(2, kernel_initializer='normal', activation='linear'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    return model

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Evaluate the performance of the model using Root Mean Squared Error.
    """
    # Make the predictions
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    # Compute the errors
    rmsr_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmsr_test = np.sqrt(mean_squared_error(y_test, pred_test))

    mse_train, mae_train = model.evaluate(x_train, y_train, verbose=0)
    mse_test, mae_test = model.evaluate(x_test, y_test, verbose=0)

    return rmsr_train, rmsr_test, mse_train, mae_train, mse_test, mae_test

def plot_history(history):
    """
    Plot the training results (loss)
    """
    plt.title('Training and validation loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Number of epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    plt.savefig('regression_problem/history.png')

def save_summary_to_file(message):
    """
    Save the model summary once the logging does not save.
    """
    f_name = 'regression_problem/summary.txt'
    with open(f_name, 'a') as f_out:
        print(message, file=f_out)

def show_results(model, x_train, y_train, x_test, y_test):
    """
    Display the results (summary, history)
    """
    # Evaluate the performance
    rmsr_train, rmsr_test, mse_train, mae_train, mse_test, mae_test = \
    evaluate_model(model, x_train, y_train, x_test, y_test)

    logger.debug('\n----- Performance-----')
    logger.debug('- Root Mean Squared Error:')
    logger.debug('Train: %.3f, test: %.3f' % (rmsr_train, rmsr_test))
    logger.debug('- Mean Squared Error:')
    logger.debug('Train: %.3f, test: %.3f' % (mse_train, mse_test))
    logger.debug('- Mean Absolute Error:')
    logger.debug('Train: %.3f, test: %.3f' % (mae_train, mae_test))

    # Sumarize model
    logger.debug('\n----- Model summary -----')
    model.summary()
    model.summary(print_fn=save_summary_to_file)

def show_training_results(model, x_train, y_train, x_test, y_test, history):
    """
    Display the training results.
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
    file_model = 'regression_problem/model.json'
    file_weights = 'regression_problem/model.h5'

    # Serialize model to JSON
    model_json = model.to_json()
    with open(file_model, "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(file_weights)


def plot_predictions(title, y_array, y_pred):
    """
    Plot the predicted values against the actual value.
    """
    target_x = y_array[:, 0]
    target_y = y_array[:, 1]

    pred_x = y_pred[:, 0]
    pred_y = y_pred[:, 1]

    fig, axs = plt.subplots(2)
    fig.suptitle(title)

    axs[0].scatter(target_x, pred_x)
    axs[0].plot([target_x.min(), target_x.max()], [target_x.min(), target_x.max()], 'k--', lw=4)
    axs[0].set_xlabel('x measured')
    axs[0].set_ylabel('x predicted')

    axs[1].scatter(target_y, pred_y)
    axs[1].plot([target_y.min(), target_y.max()], [target_y.min(), target_y.max()], 'k--', lw=4)
    axs[1].set_xlabel('y measured')
    axs[1].set_ylabel('y predicted')
    # plt.show()
    plt.savefig('regression_problem/' + str(title) + '.png')

def predict_results(model, data, y_array, title):
    """
    Make predictions with the model trained in the a dataset and verify the predicted
    positions of the sources.
    """

    # Make the predictions
    predictions = model.predict(data)
    logger.debug('Prediction shape: ' + str(predictions.shape))

    # Display the prediction for the 10 first examples
    for i in range(10):
        # point = y_array[i, :]
        # point_p = predictions[i, :]
        point = y_array[i]
        point_p = predictions[i, :]
        logger.debug('Example [' + str(i) + ']')
        # logger.debug('Real position:     (x, y, z) = (%.5f, %.5f, %.5f)' \
        #     % (point[0], point[1], point[2]))
        # logger.debug('Predicted position (x, y, z) = (%.5f, %.5f, %.5f)' \
        #     % (point_p[0], point_p[1], point_p[2]))
        # logger.debug('Real position:     (x, y) = (%.2f, %.2f)' \
        #     % (point[0], point[1]))
        # logger.debug('Predicted position (x, y) = (%.2f, %.2f)' \
        #     % (point_p[0], point_p[1]))
        logger.debug('Real position:     (z) = (%.2f)' \
            % (point))
        logger.debug('Predicted position (z) = (%.2f)' \
            % (point_p))

    # plot_predictions(title, y_array, predictions)

def predict(model, x_train, y_train, x_test, y_test):
    """
    Make predictions with the model trianed in the train dateset and test dataset.
    """

    logger.debug('\n----- Predictions -----')

    logger.debug('\nTrain predictions:')
    predict_results(model, x_train, y_train, 'Train predictions')

    logger.debug('\nTest predictions:')
    predict_results(model, x_test, y_test, 'Test predictions')

def callback_functions():
    """
    Create callback functions to monitore the training procedure.
    """

    # Stops the training when there is not improvement in the validations loss for some
    # consectuve epochs and keeps the best weghts once stopped
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1, \
        min_delta=0, restore_best_weights=True)

    # Save the log in a file
    path_logs = os.getcwd() + '/regression_problem/logs'

    # Measures and visualizes the accuracy and the loss
    tensor = TensorBoard(log_dir=path_logs, histogram_freq=0, write_graph=0, \
        write_images=False)

    cb_list = [early_stop, tensor]

    return cb_list

def neural_network(x_train, y_train, x_test, y_test):
    """
    Train a Multilayer Perceptron (MLP) to solve a regression problem, the positions of the
    sources in the 3D scene.
    """

    # ### Tests ###
    # # Stops the training when there is not improvement in the validations loss for 10
    # # consectuve epochs and keeps the best weghts once stopped
    # early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, \
    #     min_delta=0.5)
    # ##############

    # Parameters to create the model (number of nodes)
    nb_nodes_1 = 1000
    nb_nodes_2 = 400
    input_dim_1 = x_train.shape[1] # (40000 = 200x200)

    logger.debug('\nParameters to create the model:')
    logger.debug('- Dimension of the imput: ' + str(input_dim_1))
    logger.debug('- Number of nodes in the first layer (hidden layer): ' + str(nb_nodes_1))
    logger.debug('- Number of nodes in the second layer (hidden layer): ' + str(nb_nodes_2))

    # 1. Create the model (add layers)
    model = create_model(nb_nodes_1, input_dim_1, nb_nodes_2)

    # 2. Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # Parameters to train the model
    nb_epochs = 50
    nb_batchs = 1000

    logger.debug('\nParameters to train the model:')
    logger.debug('- Number of epochs: ' + str(nb_epochs))
    logger.debug('- Batch size: ' + str(nb_batchs))

     # Create the callbacks functions
    cb_list = callback_functions()

    # 3. Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, \
        batch_size=nb_batchs, callbacks=cb_list, verbose=1)

    # Display the results
    show_training_results(model, x_train, y_train, x_test, y_test, history)

    # Save the model
    save_model(model)
    logger.debug('\nModel structure and weights saved!')

    # Display predictions results
    predict(model, x_train, y_train, x_test, y_test)

def load_model():
    """
    Load the keras model
    """

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

def load_trained_neural_network(x_train, y_train, x_test, y_test):
    """
    Load and re-compile the trained neural network
    """

    # Load the trained model
    loaded_model = load_model()

    # Re-compile model
    loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # Sumarize model
    logger.debug('\n----- Model summary -----')
    loaded_model.summary()

    # Display model results
    show_results(loaded_model, x_train, y_train, x_test, y_test)

    # Display predictions results
    predict(loaded_model, x_train, y_train, x_test, y_test)

def main():

    # Compute execution time
    start_time = time.time()

    # Inital
    logger.debug('---------- [Regression problem] ----------')


    # Choose an option
    option = int(input('Do you want to train the neural network (MLP) [1] or load the ' \
        'last trained model [2]?\n'))

    # Load .npy files
    logger.debug('\n----- Loading datasets... -----')

    x_train, y_train, x_test, y_test = load_data()

    logger.debug('Datasets loaded')
    logger.debug('X_train: ' + str(x_train.shape) + ', Y_train: ' + str(y_train.shape))
    logger.debug('X_test: ' + str(x_test.shape) + ', Y_test: ' + str(y_test.shape))

    if option == 1:

        # Train the neural network (MLP)
        logger.debug('\n----- Neural network -----')
        neural_network(x_train, y_train, x_test, y_test)

    elif option == 2:

        # Load and complile trained neural network (MLP)
        logger.debug('\n----- Neural network -----')
        load_trained_neural_network(x_train, y_train, x_test, y_test)

    else:

        logger.debug('Invalid entry!')

    logger.debug('\nDone!')
    logger.debug('Execution time: %.4f seconds' % (time.time() - start_time))
    logger.debug('Execution date: ' + str(dt.now()))
    logger.debug('-----------------------------------------')

if __name__ == '__main__':
    main()
