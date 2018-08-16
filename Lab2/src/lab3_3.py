import numpy as np
from matplotlib import pyplot as plt
from k_means import kmeans

def gaussian_RBF(nominator, sigma):

    return np.e**(-nominator**2/(2*sigma**2))

def k_means_RBFS(training_samples, N):

    rbfs = kmeans(training_samples, N)
    sigma = ( rbfs.max() - rbfs.min() )/ np.sqrt(2*N)

    return rbfs, sigma

def create_training_sets():

    # column vector of input samples
    training_samples = np.transpose(np.arange(0,2*np.pi+0.1, 0.1))
    # training set true values
    training_sin = np.asarray([np.sin(2*x) for x in training_samples])
    noisy_training_sin = np.asarray([np.sin(2*x) for x in training_samples]) + np.random.normal(0,0.1,training_samples.shape[0])

    return training_samples, training_sin, noisy_training_sin

def create_validation_set():

    # column vector of input samples
    validation_samples = np.transpose(np.arange(0.05,2*np.pi+0.1, 0.1))
    # training set true values
    validation_sin = np.asarray([np.sin(2*x) for x in validation_samples])
    noisy_validation_sin = np.asarray([np.sin(2*x) for x in validation_samples]) + np.random.normal(0,0.1,validation_samples.shape[0])

    return validation_samples, validation_sin, noisy_validation_sin

def create_ballistic_tests():
    
    training_input = np.ndarray(shape=(100,2))
    training_output = np.ndarray(shape=(100,2))
    
    training_file = open('datasets/ballist.dat', 'r')
    
    current_line=0
    for line in training_file.readlines():

        stringed_line = line.split()
        floated = [float(x) for x in stringed_line]
        training_input[current_line,0] = floated[0]
        training_input[current_line,1] = floated[1]
        training_output[current_line,0] = floated[2]
        training_output[current_line,1] = floated[3]
        
        current_line+=1

    training_file.close()

    test_input = np.ndarray(shape=(100, 2))
    test_output = np.ndarray(shape=(100, 2))

    test_file = open('datasets/balltest.dat', 'r')

    current_line = 0
    for line in test_file.readlines():
        stringed_line = line.split()
        floated = [float(x) for x in stringed_line]
        test_input[current_line, 0] = floated[0]
        test_input[current_line, 1] = floated[1]
        test_output[current_line, 0] = floated[2]
        test_output[current_line, 1] = floated[3]

        current_line += 1

    test_file.close()

    return training_input, training_output, test_input, test_output


def mean_euclidean_distance( x, y):

    acc = 0

    for index in range(x.shape[0]):

        acc += np.sqrt(np.square(x[index,0]-y[index,0]) + np.square(x[index,1]-y[index,1]))

    return acc/x.shape[0]

def create_2d_rbfs(training_input, n):

    training_x = training_input[:,0]
    training_y = training_input[:,1]

    x_rbfs, sigma_x = k_means_RBFS(training_x , n)
    y_rbfs , sigma_y = k_means_RBFS(training_y, n)

    rbfs = np.ndarray(shape=(n,2))

    rbfs[:,0] = x_rbfs
    rbfs[:,1] = y_rbfs

    sigma = max( np.max(x_rbfs) - np.min(x_rbfs), np.max(y_rbfs) - np.min(y_rbfs))
    return rbfs, sigma

def create_Phi_matrix_from_2d_data(training_samples, rbfs, sigma):

    phi = np.ndarray(shape=(training_samples.shape[0], rbfs.shape[0]))

    for i in range(0,training_samples.shape[0]):

        for j in range(0,rbfs.shape[0]):

            if i==0 and j==1:
                temp = np.sum(gaussian_RBF(rbfs[j,:]-training_samples[i],sigma ))
                debug = 0
            phi[i,j] = np.sum(gaussian_RBF(rbfs[j,:]-training_samples[i],sigma ))

    return np.asarray(phi)

def random_RBFs(training_samples, N):

    rbfs = np.random.permutation(training_samples)[:N]
    sigma = ( rbfs.max() - rbfs.min() )/ np.sqrt(2*N)

    return rbfs, sigma

def euclidean_distance(x,y):

    return np.square(x[0]-y[0]) + np.square(x[1]-y[1])


def create_Phi_matrix(training_samples, rbfs, sigma):

    phi = np.ndarray(shape=(training_samples.shape[0], rbfs.shape[0]))

    for i in range(0,training_samples.shape[0]):

        column = np.asarray([gaussian_RBF(rbf-training_samples[i],sigma ) for rbf in rbfs])
        phi[i,:] = column

    return np.asarray(phi)


def run_synthetic_experiments(eta=0.2):

    training_samples, training_sin, noisy_training_sin = create_training_sets()
    validation_samples, validation_sin, noisy_validation_sin = create_validation_set()

    rbfs, sigma = k_means_RBFS(training_samples, 10)
    phi = create_Phi_matrix(training_samples, rbfs, sigma)
    weights = np.dot(np.linalg.pinv(phi), training_sin.T)

    training_prediction_error= []
    validation_prediction_error=[]
    for epoch in range(100):

        for input in training_samples:

            minimum_distance = np.abs(input - rbfs[0])
            winner = 0
            second_minimum_distance = np.abs(input - rbfs[1])
            second_winner = 1

            for rbf in range(2,rbfs.shape[0]):

                temp_distance = np.abs(input - rbfs[rbf])

                if temp_distance< minimum_distance:
                    winner = rbf
                    minimum_distance = temp_distance
                elif temp_distance< second_minimum_distance:
                    second_minimum_distance = temp_distance
                    second_winner = rbf

            rbfs[winner]+=eta*(input-rbfs[winner])
            rbfs[second_winner]+=eta*(input-rbfs[second_winner])

        phi = create_Phi_matrix(training_samples, rbfs, sigma)
        phi_test = create_Phi_matrix(validation_samples, rbfs, sigma)

        training_predictions = np.dot(phi, weights.T)
        validation_predictions = np.dot(phi_test,weights.T)
        training_prediction_error.append( np.abs(np.sum(noisy_training_sin- training_predictions))/training_sin.shape[0] )
        validation_prediction_error.append( np.abs(np.sum(noisy_validation_sin- validation_predictions))/training_sin.shape[0] )

        weights += eta*(np.sum(training_sin-training_predictions)/training_sin.shape[0])

    epochs = range(1,101)

    plt.plot(epochs, training_prediction_error,'b')
    plt.plot(epochs, validation_prediction_error,'r')
    # plt.savefig('Training_noise_free_prediction_error.png')
    plt.savefig('Training_noise_prediction_error_2.png')
    plt.clf()

    # plt.plot(epochs, validation_prediction_error)
    # plt.savefig('Validation_noise_free_prediction_error.png')
    # plt.savefig('Validation_noise_prediction_error.png')
    # plt.clf()
    # plt.show()




    debug = 0

def run_ballistic_experiment(eta=0.5):

    training_input, training_output, test_input, test_output = create_ballistic_tests()

    training_errors = []
    test_errors = []

    rbfs, sigma = create_2d_rbfs(training_input, n=2)

    training_prediction = np.ndarray(shape=(100, 2))
    test_prediction = np.ndarray(shape=(100, 2))

    # old_rbfs = np.copy(rbfs)
    # plt.scatter(old_rbfs[:,0], rbfs[:,1],color = ['blue' for line in range(rbfs.shape[0])])
    # plt.savefig('rbf_before.png')
    # plt.show()

    for epoch in range(100):

        old_rbfs = np.copy(rbfs)

        for input in range(training_input.shape[0]):

            winner = 0
            minimum_distance = euclidean_distance(training_input[input,:], rbfs[0,:])

            for rbf_index in range(1, rbfs.shape[0]):

                temp_rbf = rbfs[rbf_index,:]
                temp_distance = euclidean_distance(training_input[input,:],temp_rbf)

                if temp_distance< minimum_distance:
                    minimum_distance = temp_distance
                    winner = rbf_index

            # print(winner)

            rbfs[winner,:] += eta*(training_input[input,:]-rbfs[winner,:])

        for line in range(rbfs.shape[0]):

            print( np.sum(np.abs(rbfs[line,:] - old_rbfs[line,])))

        phi = create_Phi_matrix_from_2d_data(training_input, rbfs, sigma)

        phi_test = create_Phi_matrix_from_2d_data(test_input, rbfs, sigma)

        w1 = np.dot(np.linalg.pinv(phi), training_input[:,0].T)
        w2 = np.dot(np.linalg.pinv(phi), training_input[:,1].T)

        training_prediction[:,0] = np.dot(phi,w1.T)
        training_prediction[:,1] = np.dot(phi,w2.T)

        test_prediction[:, 0] = np.dot(phi_test, w1.T)
        test_prediction[:, 1] = np.dot(phi_test, w2.T)
        
        training_errors.append(mean_euclidean_distance(training_prediction, training_output))        
        test_errors.append(mean_euclidean_distance(test_prediction, test_output))        

    # plt.scatter(rbfs[:,0], rbfs[:,1],color = ['red' for line in range(rbfs.shape[0])])
    # plt.savefig('rbf_after.png')
    # plt.show()

    # plt.scatter(old_rbfs[:,0], rbfs[:,1],color = ['blue' for line in range(rbfs.shape[0])], label='Before training')
    # plt.scatter(rbfs[:,0], rbfs[:,1],color = ['red' for line in range(rbfs.shape[0])], label='After training')
    # plt.savefig('rbf_combined.png')
    # plt.show()

    epochs = range(1,101)

    plt.plot(epochs, training_errors,'b', label='Training set error')
    plt.plot(epochs, test_errors,'r', label='Test set error')
    plt.xlim(0,200)
    plt.legend(loc='upper right')
    plt.show()
run_synthetic_experiments()
# run_ballistic_experiment()