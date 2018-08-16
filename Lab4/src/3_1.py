import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.neural_network import BernoulliRBM as bRBM

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

    return x_train, x_test, y_train, y_test

def generate_weight_matrix(number_of_nodes):

    weight_matrix = np.ndarray(shape=(728, number_of_nodes))

    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):

            weight_matrix[i,j] = np.random.normal(0,0.01)

    return weight_matrix

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def main(number_of_nodes):

    x_train, x_test, y_train, y_test = get_datasets()

    x_train = np.asarray(x_train)

    weight_matrix = generate_weight_matrix(number_of_nodes)
    visible_biases= np.zeros(shape=(1,784))
    hidden_biases= np.zeros(shape=(1,number_of_nodes))

    bernouli = bRBM(n_components=number_of_nodes,
                    learning_rate= 0.05,
                    batch_size=1,
                    n_iter=20,
                    )

    bernouli.intercept_hidden_ = hidden_biases
    bernouli.intercept_visible_=visible_biases
    bernouli.components_= weight_matrix
    bernouli.fit(x_train)

    errors = []


    for epoch in range(20):

        acc = 0
        for image in range(x_train.shape[0]):

            res = bernouli.gibbs(visible_biases)
            print(np.sum(np.abs(x_train[image] - res)))
            acc+= np.sum(np.abs(x_train[image] - res))/float(x_train.shape[1])

        errors.append(acc/float(x_train.shape[0]))

    epochs = range(20)

    plt.plot(epochs, errors)
    plt.savefig("totalError_vs_epochs.png")
    plt.show()

if __name__ == '__main__':
    main(10)