import numpy as np
from matplotlib import  pyplot as plt

import random
import decimal

def choice():

    prob = random.uniform(0,1)
    if prob<0.5:
        return -1
    else:
        return 1

def read_votes():

    votes = np.ndarray(shape=(349, 31))
    stringed_data = open('datasets/votes.dat', 'r').readline()

    current_line = 0

    acc = []
    elem_index = 0
    while elem_index <len(stringed_data)-1:

        if (elem_index==len(stringed_data)-2):
            acc.append(stringed_data[elem_index])
            votes[current_line, :] = np.asarray(acc)
            break
            debug=0

        else:
            # skip all , characters
            if (stringed_data[elem_index] !=','):

                if (stringed_data[elem_index+1] =='.'):
                    acc.append(0.5)
                    elem_index+=3
                else:
                    acc.append(float(stringed_data[elem_index]))
                    elem_index+=1
            else:
                elem_index+=1

        if len(acc)==31:
            votes[current_line, :] = np.asarray(acc)
            current_line += 1
            acc=[]

    return votes

def read_party():

    file = open('datasets/mpparty.dat', 'r')
    file.readline()
    file.readline()
    file.readline()

    acc =[]
    while(len(acc)<349):

        acc.append(int(file.readline()))

    return acc

def read_sex():

    file = open('datasets/mpsex.dat', 'r')
    file.readline()
    file.readline()

    acc =[]
    while(len(acc)<349):

        acc.append(int(file.readline()))

    return acc

def read_ditrict():

    file = open('datasets/mpdistrict.dat', 'r')

    acc =[]
    while(len(acc)<349):

        acc.append(int(file.readline()))

    return acc

def create_weights():

    weights = []
    for i in range(10):
        temp_row = []
        for j in range(10):

            random_weight_vector = np.random.rand(1,31)
            temp_row.append( random_weight_vector)

        weights.append(temp_row)

    return weights


def similarity(x,w):

    return np.dot( (x-w[0]).T, x-w[0] )

def create_visual_representation(votes, weights, data, color, names, title):

    for member_index in range(votes.shape[0]):

        member_vote = votes[member_index, :]
        minimum_similarity = similarity(member_vote, weights[0][0])

        winner_x = 0
        winner_y = 0

        for weight_index in range(1, len(weights[0])):

            temp_similarity = similarity(member_vote, weights[0][weight_index])

            if temp_similarity < minimum_similarity:
                minimum_similarity = temp_similarity
                winner_y = weight_index

        for x_index in range(1, len(weights)):

            for y_index in range(len(weights[x_index])):

                temp_similarity = similarity(member_vote, weights[x_index][y_index])

                if temp_similarity < minimum_similarity:
                    minimum_similarity = temp_similarity
                    winner_x = x_index
                    winner_y = y_index

        step_x = 1+float(decimal.Decimal(random.randrange(1, 50))/100) * choice()
        step_y = 1+float(decimal.Decimal(random.randrange(1, 50))/100) * choice()
        try:
            plt.scatter(winner_x+step_x, winner_y+step_y, c=color[int(data[member_index])])
        except AttributeError:
            print((member_index))
    plt.xlim(0, 20)
    recs=[]

    # credits to https://stackoverflow.com/questions/26558816/matplotlib-scatter-plot-with-legend/26559256
    # for this legend scheme

    import matplotlib.patches as mpatches
    for i in range(0, len(names)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=color[i]))
    # for districts only
    # plt.legend(recs, names, loc=4, ncol=3)
    plt.legend(recs, names, loc=4)

    plt.savefig(title)
    plt.show()
    plt.clf()

def run(eta=0.2):

    # fixed values for neighbourhood sizes
    # depending on epoch

    # comment or uncomment depending
    # on 50 or 100 epochs
    neighbourhood_sizes = [5,4,3,2,1]
    # neighbourhood_sizes = [5,5,4,4,3,3,2,2,1,1]

    # read input and create grid
    votes = read_votes()
    weights = create_weights()

    political_parties = read_party()
    sex=read_sex()
    districts = read_ditrict()

    for epoch in range(50):
    # for epoch in range(100):

        neighbourhood_size = neighbourhood_sizes[int(epoch/10)]

        for member_index in range(votes.shape[0]):

            member_vote = votes[member_index,:]
            minimum_similarity = similarity(member_vote, weights[0][0])

            winner_x = 0
            winner_y = 0

            for weight_index in range(1, len(weights[0])):

                temp_similarity = similarity(member_vote, weights[0][weight_index])

                if temp_similarity < minimum_similarity:
                    minimum_similarity = temp_similarity
                    winner_y = weight_index

            for x_index in range(1,len(weights)):

                for y_index in range (len(weights[x_index])):

                    temp_similarity = similarity(member_vote, weights[x_index][y_index])

                    if temp_similarity < minimum_similarity:
                        minimum_similarity = temp_similarity
                        winner_x = x_index
                        winner_y = y_index

                # limis of the weights to be updated
                minimum_x = max(0, winner_x- neighbourhood_size)
                maximum_x = min(len(weights[0]) - 1, winner_x + neighbourhood_size)

                minimum_y = max(0, winner_y- neighbourhood_size)
                maximum_y = min(len(weights[0]) - 1, winner_y + neighbourhood_size)

                # update neighbourhood

                for x_index in range(minimum_x, maximum_x+1):
                    weights[x_index][winner_y] +=  eta * (member_vote- weights[x_index][winner_y])

                for y_index in range(minimum_y, maximum_y + 1):
                    weights[winner_x][y_index] += eta * (member_vote - weights[winner_x][y_index])


    # set color scheme for visual representation of parties
    color_scheme =['black', 'lightblue', 'blue', 'red', 'tomato', 'green',  'darkblue', 'darkgreen' ]
    names = ['No party', 'Moderate', 'Liberals', 'Swedish Socialist Party', 'Left Party', 'Green Party', 'Christian Democrats', 'Centre Party' ]

    # set color scheme for visual representation of sex
    sex_color = ['blue', 'red']
    sex_name = ['Male', 'Female']

    district_colors = ['aqua', 'aquamarine', 'black', 'blue', 'chartreuse', 'chocolate', 'darkgreen', 'gold', 'indigo', 'khaki', 'lavender','lime', 'magenta', 'goldenrod', 'navy', 'orange', 'orchid', 'plum', 'purple', 'red', 'silver', 'tan', 'teal', 'tomato', 'violet', 'yellow','yellowgreen', 'turquoise', 'salmon', 'grey', 'aquamarine']

    # district_colors = ['gold' for i in range(29)]
    district_names= [str(i) for i in range(1,30)]

    # comment or uncomment depending parties/sex/districts

    create_visual_representation(votes, weights,data=political_parties, color=color_scheme, names=names, title="parties.png")
    # create_visual_representation(votes, weights,data=sex, color=sex_color, names=sex_name, title="sex.png")
    # create_visual_representation(votes, weights,data=read_ditrict(), color=district_colors, names=district_names, title="districts.png")

    # create_district_visual_representation(votes, weights, data=districts, title="ditricts.png")

run()