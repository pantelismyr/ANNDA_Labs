import numpy as np

def create_props():

    props = np.ndarray(shape=(32,84))
    stringed_data = open('datasets/animals.dat','r').readline()

    current_line=0

    for batch in range(0,len(stringed_data),168):

        acc=[]
        temp_list = stringed_data[current_line*168:(current_line+1)*168-1]

        for elem in temp_list:

            if elem!=',':
                acc.append(elem)

        props[current_line,:] = np.asarray(acc)
        current_line+=1

    return props

def create_weights():

    return np.random.rand(100,84)

# since we are not concerned about the true value
# of the similairy distance, but only for the winner
# we do not compute the square root, only the inner part of it

def similarity(x,w):

    return np.dot( (x-w).T, x-w )

def alternative_similarity(x,w):

    return np.abs(np.sum(x-w))

def print_animal_names(indices):

    names = open('datasets/animalnames.txt').readlines()
    res = open('result_names.txt','w')

    for name in indices:

        # print(names[name])
        res.writelines(names[name])

    res.close()


def run(eta=0.2):

    props = create_props()
    weights = create_weights()

    neihgbourhood_size=50

    for epoch in range(1,21):

        for animal in range(props.shape[0]):

            props_row=props[animal,:]

            minimum_similarity = similarity(props_row, weights[0,:])
            winner_node = 0

            for weight_index in range(1,weights.shape[0]):

                temp_similarity = similarity(props_row, weights[weight_index,:])

                if temp_similarity<minimum_similarity:
                    minimum_similarity = temp_similarity
                    winner_node = weight_index

            # limis of the weights to be updated
            minimum = max(0,winner_node - int(neihgbourhood_size))
            # minimum = max(0,winner_node - int(neihgbourhood_size/2))
            maximum = min(weights.shape[0]-1, winner_node + int(neihgbourhood_size) )
            # maximum = min(weights.shape[0]-1, winner_node + int(neihgbourhood_size/2) )

            for weight_index in range(minimum, maximum +1):

                weights[weight_index,:] = weights[weight_index,:] + eta*(props_row- weights[weight_index,:])

        neihgbourhood_size = int(neihgbourhood_size-2.5*epoch)

    pos=[]

    for animal in range(props.shape[0]):

        props_row = props[animal, :]

        minimum_similarity = similarity(props_row, weights[0, :])
        winner_node = 0

        for weight_index in range(1, weights.shape[0]):

            temp_similarity = similarity(props_row, weights[weight_index, :])

            if temp_similarity < minimum_similarity:
                minimum_similarity = temp_similarity
                winner_node = weight_index

        pos.append(winner_node)

    animal_order = np.argsort(pos)
    print_animal_names(animal_order)

run()