import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM
import csv
import numpy as np
import keras
import keras.utils as keras_utils
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def getData(filename):
    with open(filename, 'r') as f:
        # read csv file
        rows = csv.reader(f)
        # create a list of lists
        ls = list(rows)
        # convert strings to integers
        ls = [[int(i) for i in row] for row in ls]

    return ls

def get_datasets():
    x_train = getData("binMNIST_data/bindigit_trn.csv")
    x_test = getData("binMNIST_data/bindigit_tst.csv")
    y_train = getData("binMNIST_data/targetdigit_trn.csv")
    y_test = getData("binMNIST_data/targetdigit_tst.csv")

    return np.asarray(x_train), np.asarray(x_test), \
           keras_utils.to_categorical(y_train, 10), keras_utils.to_categorical(y_test, 10)

def generate_weight_matrix(number_of_inputs, number_of_nodes):

    weight_matrix = np.ndarray(shape=(number_of_inputs, number_of_nodes))

    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):

            weight_matrix[i,j] = np.random.normal(0,0.01)

    return weight_matrix

def vis(x_test, pred_imgs, n):
    plt.figure(2, figsize=(n, 5))
    i = 0
    count = 0
    for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(np.reshape(x_test[i], (28, 28)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(np.reshape(pred_imgs[i], (28, 28)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            i += 1
    plt.suptitle('Original (top) vs Reconstructed (bottom) images')
    plt.show()

def weight_vis(final_weights):
    plt.figure(3, figsize=(10, 6))
    # puth the weights into a list for convenience
    weights_ls = final_weights.tolist()
    for i in range(len(weights_ls)):
            ax = plt.subplot(10, 10, i + 1)
            plt.imshow(np.reshape(weights_ls[i], (28, 28)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.suptitle('Bottom-up final weights for each hidden unit')

def run(first_hidden, second_hidden, learning_rate, epochs, batch_size):

    x_train, x_test, y_train, y_test = get_datasets()

    # ------------------FIRST HIDDEN-------------------

    weight_matrix = generate_weight_matrix(x_train.shape[1], hidden)
    visible_biases= np.zeros(shape=(784))
    hidden_biases= np.zeros(shape=(first_hidden))

    bbrbm = BBRBM(
        n_visible=784,
        n_hidden=first_hidden,
        learning_rate=learning_rate,
        momentum=0,
        # momentum=0.95,
        use_tqdm=True
    )

    bbrbm.set_weights(weight_matrix, visible_biases, hidden_biases)
    errs = bbrbm.fit(x_train, n_epoches=epochs, batch_size=batch_size)

    first_weight_matrix = np.asarray(bbrbm.get_weight_matrix())
    first_visible_biases= bbrbm.get_visible_biases()
    first_hidden_biases= bbrbm.get_hidden_biases()

    first_output = np.dot(x_test, weight_matrix)

    for i in range(first_output.shape[0]):

        first_output[i,:]+=first_hidden_biases

        for j in range(first_output.shape[1]):

            # sigmoid(relu(x))
            first_output[i,j] = sigmoid(np.max(0, first_output[i,j]))


    # ------------------SECOND HIDDEN-------------------

    bbrbm_second = BBRBM(
        n_visible=first_hidden,
        n_hidden=784,
        learning_rate=learning_rate,
        momentum=0,
        # momentum=0.95,
        use_tqdm=True
    )



    trained_outputs = np.ndarray(shape=(x_train.shape[0], x_train.shape[1]))
    for i in range(trained_outputs.shape[0]):

        trained_outputs[i,:] = bbrbm.reconstruct(x_train[i].reshape(1, -1))

    sgd = optimizers.SGD(lr=0.3, decay=0, momentum=0, nesterov=True)
    model = Sequential()
    model.add(Dense(10, input_dim=784, activation="softmax", kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros'))
    model.compile(
        optimizer=sgd,
        loss='mean_squared_error',
        metrics=['accuracy'],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None)

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    model.fit(trained_outputs,
        np.asarray(y_train),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        shuffle = True,
        callbacks= [earlyStopping],
        validation_data=(x_test, y_test)
    )

    score = model.evaluate(x_test, np.asarray(y_test), verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # vis(x_train, trained_outputs, 20)


    trained_outputs = np.dot(x_train, weight_matrix)
    for i in range(trained_outputs.shape[0]):

        trained_outputs[i,:]+=hidden_biases


    debug = 0

    # test = np.dot(x_test, weight_matrix)
    #
    # preds = []
    # for i in range(test.shape[0]):
    #
    #     test[i,:]+=hidden_biases

    # output_weight_matrix = generate_output_weight_matrix(test.shape[1])
    # sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=True)
    # model = Model(inputs=test, outputs=y_test)

    model.compile(
        optimizer=sgd,
        loss='mean_squared_error',
        metrics=['accuracy'],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None)

    debug = 0

    # plt.figure(2, figsize=(20, 5))
    # i = 0
    # count = 0
    # for i in [0,1,2,3,5,6,7,8,15,18]:
    #     # display original
    #     ax = plt.subplot(2, 10, count + 1)
    #     image = x_test[i]
    #     # plt.imshow(np.reshape(x_test[i,:], (28, 28)))
    #     # plt.gray()
    #     plt.imshow(image.reshape((28, 28)), cmap=plt.cm.gray)
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     rec = bbrbm.reconstruct(image.reshape(1, -1))
    #     ax = plt.subplot(2, 10, count + 1 + 10)
    #     # plt.imshow(np.reshape(bbrbm.reconstruct(x_test[i,:]), (28, 28)))
    #     # plt.gray()
    #     plt.imshow(rec.reshape((28, 28)), cmap=plt.cm.gray)
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     count+=1
    #
    # plt.suptitle('Original (top) vs Reconstructed (bottom) images')
    # plt.savefig("Single_layer_RBM.png")
    # plt.show()
    # plt.clf()

run(hidden=100,second_hidden=50, learning_rate=0.08, epochs=200, batch_size=128)