import matplotlib.pyplot as plt
from tfrbm import BBRBM
import csv
import numpy as np
import keras
import keras.utils as keras_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import math

# ----------------------- ACTIVATION FUNCTIONS ------------------------

def sigmoid(x):
    val = 1 / (1 + math.exp(-x))

    if val > 0.5:
        return 1
    else:
        return 0

def relu(x):

    if x<0.0:
        return 0.0
    else:
        return x

# ------------------------ DATA COLLECTION -----------------------------

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


# ----------------------- WEIGHT GENERATOR -------------------------------

def generate_weight_matrix(number_of_inputs, number_of_nodes):

    weight_matrix = np.ndarray(shape=(number_of_inputs, number_of_nodes))

    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):

            weight_matrix[i,j] = np.random.normal(0,0.01)

    return weight_matrix

# ------------------------ IMAGE VISUALIZATION -------------------------------

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

def dbn(first_layer_hidden_nodes,second_layer_hidden_nodes, learning_rate ,epoches, third_layer_hidden_nodes=0, batch_size=128):

    # ---------------------- PRE-TRAINING --------------------

    # gets datasets: x_{train,test} are np arrays
    # y_{train,test} are categorical dstributions
    # over a digit
    x_train, x_test, y_train, y_test = get_datasets()


    # ---------------------- FIRST LAYER --------------------

    # Generate weights ~ N(mean =0, stdev =0.01) & zero biases
    weight_matrix = generate_weight_matrix(x_train.shape[1], first_layer_hidden_nodes)
    visible_biases= np.zeros(shape=(x_train.shape[1]))
    hidden_biases= np.zeros(shape=(first_layer_hidden_nodes))

    # initialize Restricted Boltzman Machine
    first_bbrbm = BBRBM(
        n_visible=x_train.shape[1],
        n_hidden=first_layer_hidden_nodes,
        learning_rate=learning_rate,
        momentum=0,
        # momentum=0.95,
        use_tqdm=True
    )

    first_bbrbm.set_weights(weight_matrix, visible_biases, hidden_biases)

    # fit model
    first_bbrbm.fit(data_x=x_train,validation_data_x=x_test, n_epoches=epoches, batch_size=batch_size)

    rbm_first_layer_weight_matrix = first_bbrbm.get_weight_matrix()
    rbm_first_layer_hidden_layer_biases = first_bbrbm.get_hidden_biases()

    first_output = np.dot(x_train, rbm_first_layer_weight_matrix)

    for i in range(first_output.shape[0]):

        first_output[i, :] += rbm_first_layer_hidden_layer_biases

        for j in range(first_output.shape[1]):
            # sigmoid(relu(x))
            first_output[i, j] = sigmoid(relu(first_output[i, j]))

    first_test_output = np.dot(x_test, rbm_first_layer_weight_matrix)

    for i in range(first_test_output.shape[0]):

        first_test_output[i, :] += rbm_first_layer_hidden_layer_biases

        for j in range(first_test_output.shape[1]):
            # sigmoid(relu(x))
            first_test_output[i, j] = sigmoid(relu(first_test_output[i, j]))
            
    # ---------------------- SECOND LAYER --------------------

    # Generate weights ~ N(mean =0, stdev =0.01) & zero biases
    weight_matrix = generate_weight_matrix(first_layer_hidden_nodes, second_layer_hidden_nodes)
    visible_biases= np.zeros(shape=(first_layer_hidden_nodes))
    hidden_biases= np.zeros(shape=(second_layer_hidden_nodes))

    # initialize Restricted Boltzman Machine
    second_bbrbm = BBRBM(
        n_visible=first_layer_hidden_nodes,
        n_hidden=second_layer_hidden_nodes,
        learning_rate=learning_rate,
        momentum=0,
        # momentum=0.95,
        use_tqdm=True
    )

    second_bbrbm.set_weights(weight_matrix, visible_biases, hidden_biases)

    # fit model
    second_bbrbm.fit(data_x=first_output,validation_data_x=first_test_output, n_epoches=epoches, batch_size=batch_size)

    rbm_second_layer_weight_matrix = second_bbrbm.get_weight_matrix()
    rbm_second_layer_hidden_layer_biases = second_bbrbm.get_hidden_biases()

    second_output = np.dot(first_output, rbm_second_layer_weight_matrix)

    for i in range(second_output.shape[0]):

        second_output[i, :] += rbm_second_layer_hidden_layer_biases

        for j in range(second_output.shape[1]):
            # sigmoid(relu(x))
            second_output[i, j] = sigmoid(relu(second_output[i, j]))

    second_test_output = np.dot(first_test_output, rbm_second_layer_weight_matrix)

    for i in range(second_test_output.shape[0]):

        second_test_output[i, :] += rbm_second_layer_hidden_layer_biases

        for j in range(second_test_output.shape[1]):
            # sigmoid(relu(x))
            second_test_output[i, j] = sigmoid(relu(second_test_output[i, j]))

    # ---------------------- THIRD LAYER --------------------

    if third_layer_hidden_nodes>0:

        # Generate weights ~ N(mean =0, stdev =0.01) & zero biases
        weight_matrix = generate_weight_matrix(second_layer_hidden_nodes, third_layer_hidden_nodes)
        visible_biases = np.zeros(shape=(second_layer_hidden_nodes))
        hidden_biases = np.zeros(shape=(third_layer_hidden_nodes))

        # initialize Restricted Boltzman Machine
        third_bbrbm = BBRBM(
            n_visible=second_layer_hidden_nodes,
            n_hidden=third_layer_hidden_nodes,
            learning_rate=learning_rate,
            momentum=0,
            # momentum=0.95,
            use_tqdm=True
        )

        third_bbrbm.set_weights(weight_matrix, visible_biases, hidden_biases)

        # fit model
        third_bbrbm.fit(data_x=second_output, validation_data_x=second_test_output, n_epoches=epoches,
                         batch_size=batch_size)

        rbm_third_layer_weight_matrix = third_bbrbm.get_weight_matrix()
        rbm_third_layer_hidden_layer_biases = third_bbrbm.get_hidden_biases()

        third_output = np.dot(second_output, rbm_third_layer_weight_matrix)

        for i in range(third_output.shape[0]):

            third_output[i, :] += rbm_third_layer_hidden_layer_biases

            for j in range(third_output.shape[1]):
                # sigmoid(relu(x))
                third_output[i, j] = sigmoid(relu(third_output[i, j]))

        third_test_output = np.dot(second_test_output, rbm_third_layer_weight_matrix)

        for i in range(third_test_output.shape[0]):

            third_test_output[i, :] += rbm_third_layer_hidden_layer_biases

            for j in range(third_test_output.shape[1]):
                # sigmoid(relu(x))
                third_test_output[i, j] = sigmoid(relu(third_test_output[i, j]))

        # ------------------------ CLASSIFICATION ---------------------

        sgd = optimizers.SGD(lr=0.3, decay=0, momentum=0, nesterov=True)
        model = Sequential()
        model.add(Dense(10,
                        input_dim=third_output.shape[1],
                        activation="softmax",
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                        bias_initializer='zeros'))
        model.compile(
            optimizer=sgd,
            loss='mean_squared_error',
            metrics=['accuracy'],
            loss_weights=None,
            sample_weight_mode=None,
            weighted_metrics=None,
            target_tensors=None)

        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                                                      mode='auto')

        model.fit(third_output,
                  np.asarray(y_train),
                  batch_size=batch_size,
                  epochs=epoches,
                  verbose=1,
                  shuffle=True,
                  callbacks= [earlyStopping],
                  validation_data=(third_test_output, np.asarray(y_test))
                  )

        score = model.evaluate(third_test_output, np.asarray(y_test), verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return  score[1]

    else:

        # ------------------------ CLASSIFICATION ---------------------

        sgd = optimizers.SGD(lr=0.3, decay=0, momentum=0, nesterov=True)
        model = Sequential()
        model.add(Dense(10,
                        input_dim=second_output.shape[1],
                        activation="softmax",
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                        bias_initializer='zeros'))
        model.compile(
            optimizer=sgd,
            loss='mean_squared_error',
            metrics=['accuracy'],
            loss_weights=None,
            sample_weight_mode=None,
            weighted_metrics=None,
            target_tensors=None)

        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

        model.fit(second_output,
                  np.asarray(y_train),
                  batch_size=batch_size,
                  epochs=epoches,
                  verbose=1,
                  shuffle=True,
                  callbacks= [earlyStopping],
                  validation_data=(second_test_output, np.asarray(y_test))
                  )

        score = model.evaluate(second_test_output, np.asarray(y_test), verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return score[1]

def tune_parameters():
    res=dict()

    for third_hidden in [40,50,60,80]:
        res[str(dbn(first_layer_hidden_nodes=150,second_layer_hidden_nodes=100, learning_rate= 0.08, epoches=100,\
                    third_layer_hidden_nodes=third_hidden))] =  third_hidden

    maximum =0
    for key in res:
        if float(key)>maximum:
            maximum = float(key)
    print(maximum)
    print(str(res[key]))

tune_parameters()