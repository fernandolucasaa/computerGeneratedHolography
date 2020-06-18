import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def main():
    '''
    bla bla bla
    '''

    # Load .npy files
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')
    '''
    # load the dataset
    dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    print(X.shape)
    print(y.shape)
    # Create the model
    model = Sequential() # build a model layer by layer

    # Add model layers

    # First layer (input layer)
    nb_nodes1 = 12
    input_dim1 = 8 
    activation1 = 'relu' # Rectified linear unit (ReLU)

    model.add(Dense(nb_nodes1, input_dim=input_dim1, activation=activation1))

    # Second layer (hidden layer)
    nb_nodes2 = 8
    activation2 = 'relu' # Rectified linear unit (ReLU)

    model.add(Dense(nb_nodes2, activation=activation2))

    # Third layer (output layer)
    nb_nodes3 = 1
    activation3 = 'sigmoid' # Sigmoid, ensures the newtowork output between 0 and 1

    model.add(Dense(nb_nodes3, activation=activation3))

    # loss_fct = 'categorical_crossentropy' # Used for classification problem
    loss_fct = 'binary_crossentropy'
    opt_fct = 'adam' # Popular version of Gradient Descent
    metrics_fct = ['accuracy'] # Classification problem

    model.compile(loss=loss_fct, optimizer=opt_fct, metrics=metrics_fct)

    nb_epochs = 150
    nb_batchs = 10

    history = model.fit(X, y, epochs=nb_epochs, batch_size=nb_batchs, verbose=0)

    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    # # Summarize model.
    # model.summary()

    # plt.plot(history.history['accuracy'], label = 'train')
    # plt.xlabel('Number of epochs')
    # plt.ylabel('Accuracy')
    # plt.show()
    '''

if __name__ == '__main__':

     # Compute execution time
    start_time = time.time()

    main()

    print('Done!')
    print('Execution time: %.4f seconds' % (time.time() - start_time))