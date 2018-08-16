import numpy as np
from matplotlib import pyplot as plt

def create_weights():

    return np.random.rand(10,2)

def read_cities():

    cities = np.ndarray(shape=(10,2))
    stringed_cities = open('datasets/cities.dat','r').readlines()

    current_line=0

    for line in stringed_cities:

        temp1= line.split(',')
        temp2= temp1[1].split(';')

        cities[current_line, 0] = float(temp1[0])
        cities[current_line, 1] = float(temp2[0])

        current_line+=1

    return cities

def similarity(x,w):

    return np.dot( (x-w).T, x-w )

def run(eta=0.2):

    cities = read_cities()
    # x=cities[:,0]
    # y=cities[:,1]
    # plt.scatter(x,y)
    # plt.show()
    weights = create_weights()

    neihgbourhood_size = 1

    for epoch in range(1, 51):

        for city in range(cities.shape[0]):

            city_row = cities[city, :]

            minimum_similarity = similarity(city_row, weights[0, :])
            winner_node = 0

            for weight_index in range(1, weights.shape[0]):

                temp_similarity = similarity(city_row, weights[weight_index, :])

                if temp_similarity < minimum_similarity:
                    minimum_similarity = temp_similarity
                    winner_node = weight_index

        if neihgbourhood_size > 0:

            # limis of the weights to be updated
            if winner_node - neihgbourhood_size < 0:
                left = 10 + winner_node - neihgbourhood_size
            else:
                left = winner_node - neihgbourhood_size

            if winner_node + neihgbourhood_size > 9:
                right = winner_node + neihgbourhood_size - 10
            else:
                right = winner_node + neihgbourhood_size

            if left > right:
                for weight_index in range(left, 10):
                    weights[weight_index, :] = weights[weight_index, :] + eta * (city_row - weights[weight_index, :])
                for weight_index in range(0, right + 1):
                    weights[weight_index, :] = weights[weight_index, :] + eta * (city_row - weights[weight_index, :])
            else:
                for weight_index in range(left, right + 1):
                    weights[weight_index, :] = weights[weight_index, :] + eta * (city_row - weights[weight_index, :])
        else:
            weights[winner_node, :] = weights[weight_index, :] + eta * (city_row - weights[weight_index, :])

        if epoch > 8:
            neihgbourhood_size = 1
        if epoch > 15:
            neihgbourhood_size = 0

    pos = []

    for city in range(cities.shape[0]):

        city_row = cities[city, :]

        minimum_similarity = similarity(city_row, weights[0, :])
        winner_node = 0

        for weight_index in range(1, weights.shape[0]):

            temp_similarity = similarity(city_row, weights[weight_index, :])

            if temp_similarity < minimum_similarity:
                minimum_similarity = temp_similarity
                winner_node = weight_index

        pos.append(winner_node)

    city_order = np.argsort(pos)
    # city_order = np.asarray(pos)

    x=np.asarray([cities[city_order[0],0],cities[city_order[1],0]])
    y=np.asarray([cities[city_order[0],1],cities[city_order[1],1]])
    plt.plot(x,y,'b',label='1')
    x = np.asarray([cities[city_order[1], 0], cities[city_order[2], 0]])
    y = np.asarray([cities[city_order[1], 1], cities[city_order[2], 1]])
    plt.plot(x, y, 'r', label='2')
    x = np.asarray([cities[city_order[2], 0], cities[city_order[3], 0]])
    y = np.asarray([cities[city_order[2], 1], cities[city_order[3], 1]])
    plt.plot(x, y, 'fuchsia', label='3')
    x = np.asarray([cities[city_order[3], 0], cities[city_order[4], 0]])
    y = np.asarray([cities[city_order[3], 1], cities[city_order[4], 1]])
    plt.plot(x, y, 'aquamarine', label='4')
    x = np.asarray([cities[city_order[4], 0], cities[city_order[5], 0]])
    y = np.asarray([cities[city_order[4], 1], cities[city_order[5], 1]])
    plt.plot(x, y, 'brown', label='5')
    x = np.asarray([cities[city_order[5], 0], cities[city_order[6], 0]])
    y = np.asarray([cities[city_order[5], 1], cities[city_order[6], 1]])
    plt.plot(x, y, 'chocolate', label='6')
    x = np.asarray([cities[city_order[6], 0], cities[city_order[7], 0]])
    y = np.asarray([cities[city_order[6], 1], cities[city_order[7], 1]])
    plt.plot(x, y, 'green', label='7')
    x = np.asarray([cities[city_order[7], 0], cities[city_order[8], 0]])
    y = np.asarray([cities[city_order[7], 1], cities[city_order[8], 1]])
    plt.plot(x, y, 'darkgreen', label='8')
    x = np.asarray([cities[city_order[8], 0], cities[city_order[9], 0]])
    y = np.asarray([cities[city_order[8], 1], cities[city_order[9], 1]])
    plt.plot(x, y, 'gold', label='9')
    x = np.asarray([cities[city_order[9], 0], cities[city_order[0], 0]])
    y = np.asarray([cities[city_order[9], 1], cities[city_order[0], 1]])
    plt.plot(x, y, 'indigo', label='10')
    plt.legend(loc='upper right')

    plt.title("Traveling salesman route")
    plt.xlim(0,1.5)
    plt.savefig('tsp1.png')

    plt.show()



    debug = 0

run()