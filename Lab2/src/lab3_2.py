import numpy as np
import matplotlib.pyplot as plt
from k_means import kmeans

# Gaussian transfer function
# m is the position of unit i
# sigma is its variance

def gaussian_RBF(nominator, sigma):

    return np.e**(-nominator**2/(2*sigma**2))

def create_Phi_matrix(training_samples, rbfs, sigma):

    phi = np.ndarray(shape=(training_samples.shape[0], rbfs.shape[0]))

    for i in range(0,training_samples.shape[0]):

        column = np.asarray([gaussian_RBF(rbf-training_samples[i],sigma ) for rbf in rbfs])
        phi[i,:] = column

    return np.asarray(phi)

# creates same-distanced rbf's
def create_RBFs(training_samples, N):

    length = training_samples.shape[0]
    step = int(length / N)

    rbfs = []

    for index in range(int(step/2) , training_samples.shape[0], step):

        rbfs.append(training_samples[index])

    rbfs = np.asarray(rbfs)
    sigma = 0.001*( rbfs.max() - rbfs.min() )/ np.sqrt(2*N)

    return np.asarray(rbfs), sigma

def random_RBFs(training_samples, N):

    rbfs = np.random.permutation(training_samples)[:N]
    sigma = ( rbfs.max() - rbfs.min() )/ np.sqrt(2*N)

    return rbfs, sigma

def k_means_RBFS(training_samples, N):

    rbfs = kmeans(training_samples, N)
    sigma = ( rbfs.max() - rbfs.min() )/ np.sqrt(2*N)

    return rbfs, sigma

def create_training_set():

    # column vector of input samples
    training_samples = np.transpose(np.arange(0,2*np.pi+0.1, 0.1))
    # training set true values
    training_sin = np.asarray([np.sin(2*x) for x in training_samples]) + np.random.normal(0,0.1,training_samples.shape[0])

    return training_samples, training_sin

def create_validation_set():

    # column vector of input samples
    validation_samples = np.transpose(np.arange(0.05,2*np.pi+0.1, 0.1))
    # training set true values
    validation_sin = np.asarray([np.sin(2*x) for x in validation_samples]) + np.random.normal(0,0.1,validation_samples.shape[0])

    return validation_samples, validation_sin


def batch_train_and_test_sin_function(eta=0.5):

    training_samples, training_sin = create_training_set()

    plt.plot(training_samples, training_sin)
    plt.title("sin(2x) function with zero-mean noise added")
    plt.savefig("sin(2x)_training_noise.png")
    # plt.show()
    plt.clf()

    found1=False
    found2=False
    found3=False

    validation_samples, validation_sin = create_validation_set()

    minimum_residual_error = 2

    for n in range(4,40):

        if found1:
            break

        rbfs, sigma = k_means_RBFS(training_samples, N=n)
        # rbfs, sigma = create_RBFs(training_samples, N=n)
        # rbfs, sigma = random_RBFs(training_samples, N=n)
        phi = create_Phi_matrix(training_samples, rbfs, sigma)

        weights = np.dot(np.linalg.pinv(phi), training_sin.T)

        for epoch in range(50):

            predictions = np.dot(phi, weights.T)
            step = np.sum(training_sin-predictions)/training_sin.shape[0]
            total_residual_error = np.sum(np.abs(training_sin-predictions))/training_sin.shape[0]
            minimum_residual_error = min(minimum_residual_error, total_residual_error)

            if total_residual_error < 0.001 and not(found1):
                print("Total residual error fell under 0.001 for "+str(n)+"hidden layers:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found1 = True
                plt.plot(training_samples, predictions, label="Function approximation")
                plt.plot(training_samples, training_sin, label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function with noise added for target error<0.001")
                plt.savefig("Noise_k-means_sin(2x)_error_0_001.png")
                # plt.show()
                plt.clf()

            if total_residual_error < 0.01 and not(found2):
                print("Total residual error fell under 0.01 for "+str(n)+"hidden layers:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found2 = True
                plt.plot(training_samples, predictions, label="Function approximation")
                plt.plot(training_samples, training_sin, label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function with noise added for target error<0.01")
                plt.savefig("Noise_k-means_sin(2x)_error_0_01.png")
                # plt.show()
                plt.clf()

            if total_residual_error < 0.1 and not(found3):
                print("Total residual error fell under 0.1 for "+str(n)+"hidden layers:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found3 = True
                plt.plot(training_samples, predictions, label="Function approximation")
                plt.plot(training_samples, training_sin, label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function with noise added for target error<0.1")
                plt.savefig("Noise_k-means_sin(2x)_error_0_1.png")
                plt.show()
                plt.clf()

            # print(total_residual_error)

            # rbfs,sigma = create_RBFs(training_samples, n)
            # rbfs, sigma = random_RBFs(training_samples, N=n)
            # rbfs, sigma = k_means_RBFS(training_samples, N=n)
            # phi = create_Phi_matrix(training_samples, rbfs, sigma)

            weights += eta*step

    print("minimum residual error is :"+str(minimum_residual_error))

def create_column_phi_matrix(x, rbfs, sigma):

    return np.asarray([gaussian_RBF(x-rbf, sigma) for rbf in rbfs]).T

def sequential_train_and_test_sin_function():

    inputs, targets = create_training_set()

    found1=False
    found2=False
    found3=False


    for hidden_nodes in range(4,30):

        if found1:
            break

        rbfs, sigma = k_means_RBFS(inputs, N=hidden_nodes)
        full_phi_matrix = create_Phi_matrix(inputs, rbfs, sigma)
        weights = np.dot(np.linalg.pinv(full_phi_matrix), targets.T)

        eta =0

        while eta+0.1<1.1:

            eta+=0.1

            for epoch in range(1000):

                predictions=[]
                residual_error=[]

                for index in range(inputs.shape[0]):

                    phi = create_column_phi_matrix(inputs[index],rbfs,sigma)
                    prediction = np.dot(phi,weights)
                    predictions.append(prediction)

                    deviation = targets[index] - prediction
                    residual_error.append(np.abs(deviation))
                    instantaneous_error = 0.5*deviation**2

                    add = eta*instantaneous_error
                    weights += add*phi

                residual_error = np.asarray(residual_error)
                predictions = np.asarray(predictions)
                absolute_residual_error= np.sum(residual_error)/residual_error.shape[0]

                if absolute_residual_error < 0.001 and not(found1):
                    print("Total residual error fell under 0.001 for "+str(hidden_nodes)+"hidden layers and "+str(eta) +" learning rate value:"+str(absolute_residual_error)+" after "+str(epoch)+" epochs")
                    found1 = True
                    plt.plot(inputs, predictions, label="Function approximation")
                    plt.plot(inputs, targets, label="True function")
                    plt.legend(loc='upper right')
                    plt.ylim(-1.5,1.5)
                    plt.title("Approximating sin(2x) function with noise added for target error with sequential training<0.001")
                    plt.savefig("Sequential_Noise_k-means_sin(2x)_error_0_001.png")
                    plt.show()
                    plt.clf()

                if absolute_residual_error < 0.01 and not(found2):
                    print("Total residual error fell under 0.01 for "+str(hidden_nodes)+"hidden layers and "+str(eta) +" learning rate value:"+str(absolute_residual_error)+" after "+str(epoch)+" epochs")
                    found2 = True
                    plt.plot(inputs, predictions, label="Function approximation")
                    plt.plot(inputs, targets, label="True function")
                    plt.legend(loc='upper right')
                    plt.ylim(-1.5,1.5)
                    plt.title("Approximating sin(2x) function with noise added for target error<0.01 with sequential training")
                    plt.savefig("Sequential_Noise_k-means_sin(2x)_error_0_01.png")
                    plt.show()
                    plt.clf()

                if absolute_residual_error < 0.1:
                    print("Total residual error fell under 0.1 for "+str(hidden_nodes)+"hidden layers and "+str(eta) +" learning rate value:"+str(absolute_residual_error)+" after "+str(epoch)+" epochs")
                    found3 = True
                    plt.plot(inputs, predictions, label="Function approximation")
                    plt.plot(inputs, targets, label="True function")
                    plt.legend(loc='upper right')
                    plt.ylim(-1.5,1.5)
                    plt.title("Approximating sin(2x) function with noise added for target error<0.1 with sequential training")
                    plt.savefig("Sequential_Noise_k-means_sin(2x)_error_0_1.png")
                    plt.show()
                    plt.clf()

batch_train_and_test_sin_function()
# sequential_train_and_test_sin_function()