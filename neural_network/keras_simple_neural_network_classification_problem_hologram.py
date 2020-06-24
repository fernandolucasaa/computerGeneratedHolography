import time
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.utils import to_categorical

def load_data():
    '''
    Load the datasets in .npy format.
    '''
    x_train = np.load('X_train.npy')
    y_train = np.load('Y_train.npy')

    x_test = np.load('X_test.npy')
    y_test = np.load('Y_test.npy')

    return x_train, y_train, x_test, y_test

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

    return

def show_results(model, X_train, Y_train, X_test, Y_test):
    '''
    Display the model results.
    '''
    # Evaluate the model
    train_acc, test_acc = evaluate_model(model, X_train, Y_train, X_test, Y_test)

    print('\n----- Accuracy -----')
    print('Train accuracy: %.2f%%, Test accuracy: %.2f%%' % (train_acc*100, test_acc*100))

    # Sumarize model
    print('\n----- Model summary -----')
    model.summary()

    return

def show_training_results(model, X_train, Y_train, X_test, Y_test, history):
    '''
    Display the training results.
    '''
    # Display the model results
    show_results(model, X_train, Y_train, X_test, Y_test)

    # Plot the history
    plot_history(history)

    return

def save_model(model, file_name_model, file_name_weights):
    '''
    Save the weights and the model separately. Note that it must re-compile
    the model when loaded.
    '''
    # Serialize model to JSON
    model_json = model.to_json()
    with open(file_name_model, "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(file_name_weights)

    return

def train_neural_network():
    '''
    Train a Multilayer Perceptron to solve a classification problem, the number of
    ponctual sources in the holograms.
    '''
    # Load .npy files
    X_train, Y_train, X_test, Y_test = load_data()

    # Convert target classes to categorical ones
    nb_class = 5
    Y_train, Y_test = categorical_target(nb_class, Y_train, Y_test)

    # Create the model
    model = Sequential()  # build a model layer by layer

    # Add model layers

    # First layer (hidden layer)
    nb_nodes1 = 1000
    input_dim1 = X_train.shape[1] # 40000 
    activation1 = 'relu' # Rectified linear unit (ReLU)

    model.add(Dense(nb_nodes1, input_dim=input_dim1, activation=activation1))

    # Second layer (hidden layer)
    nb_nodes2 = 400
    activation2 = 'relu' # Rectified linear unit (ReLU)

    model.add(Dense(nb_nodes2, activation=activation2))

    # Third layer (output layer)
    activation3 = 'softmax'

    model.add(Dense(nb_class, activation=activation3))

    # Compile the model
    loss_fct = 'categorical_crossentropy' # Used for classification problem
    opt_fct = 'adam' # Popular version of Gradient Descent
    metrics_fct = ['accuracy'] # Classification problem

    model.compile(loss=loss_fct, optimizer=opt_fct, metrics=metrics_fct)

    # Train the model
    nb_epochs = 50
    nb_batchs = 1000

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=nb_epochs, batch_size=nb_batchs, verbose=1)

    # Display training results
    show_training_results(model, X_train, Y_train, X_test, Y_test, history)

    # Save the model
    file_model = 'classification_problem/model.json'
    file_weights = 'classification_problem/model.h5'

    save_model(model, file_model, file_weights)
    print('\nModel structure and weights saved!')

    return

def load_model(file_name_model, file_name_weights):
    '''
    Load the keras model
    '''
    # Load json file
    json_file = open(file_name_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Create model
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights(file_name_weights)

    return loaded_model

def load_trained_neural_network():
    '''
    Load and re-compile the trained neural network
    '''

    # Load the model trained
    name_model = 'classification_problem/model.json'
    name_weigths = 'classification_problem/model.h5'

    loaded_model = load_model(name_model, name_weigths)

    # Re-compile model
    loss_fct = 'categorical_crossentropy' # Used for classification problem
    opt_fct = 'adam' # Popular version of Gradient Descent
    metrics_fct = ['accuracy'] # Classification problem

    loaded_model.compile(loss=loss_fct, optimizer=opt_fct, metrics=metrics_fct)

    # Load .npy files
    X_train, Y_train, X_test, Y_test = load_data()

    # Convert target classes to categorical ones
    nb_class = 5
    Y_train, Y_test = categorical_target(nb_class, Y_train, Y_test)

    # Display model results
    show_results(loaded_model, X_train, Y_train, X_test, Y_test)

    return

if __name__ == '__main__':

    # Compute execution time
    start_time = time.time()

    # Choose an option
    option = int(input('Do you want to train the neural network [1] or load the last trained model [2]?\n'))

    if option == 1:

        # Train the neural network (MLP)
        train_neural_network()

    elif option == 2:

        # Load and complile trained neural network (MLP)
        load_trained_neural_network()

    else:

        print('Invalid entry!')

    print('\nDone!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))
    