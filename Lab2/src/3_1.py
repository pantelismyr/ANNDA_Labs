import numpy as np
from matplotlib import pyplot as plt
from k_means import kmeans

def square_function(x):

    if x<np.pi:
    # if x<2*np.pi:
        return 1
    else:
        return -1

# Gaussian transfer function
# m is the position of unit i
# sigma is its variance

def gaussian_RBF(nominator, sigma):

    return np.e**(-nominator**2/(2*sigma**2))
    # return np.e**(-nominator/(2*sigma**2))


def create_Phi_matrix(training_samples, rbfs, sigma):

    phi = np.ndarray(shape=(training_samples.shape[0], rbfs.shape[0]))

    for i in range(0, training_samples.shape[0]):

        column = np.asarray([gaussian_RBF(rbf-training_samples[i],sigma ) for rbf in rbfs])
        # column = np.asarray([gaussian_RBF(np.sqrt((training_samples[i] - rbf)**2), sigma ) for rbf in rbfs])

        phi[i,:] = column

    return np.asarray(phi)

# creates same-distanced rbf's
def create_RBFs(training_samples, N):

    length = training_samples.shape[0]
    step = int(length / N)

    rbfs = []
    variance= int((step/2)+3) * 0.1
    # variance= int((step/2)) * 0.1

    for index in range(int(step/2) , training_samples.shape[0], step):

        rbfs.append(training_samples[index])

    return np.asarray(rbfs), variance

def random_RBFs(training_samples, N):

    #rbfs = np.random.permutation(training_samples)[:N]
    rbfs = np.random.choice(training_samples, size=N, replace=False)
    sigma = (np.amax(rbfs) - np.amin(rbfs)) / np.sqrt(2*N)


    return rbfs, sigma

def k_means_RBFS(training_samples, N):

    rbfs = kmeans(training_samples, N)
    # rbfs = kmeans.fit(training_samples)
    sigma = ( rbfs.max() - rbfs.min() )/ np.sqrt(2*N)


    return rbfs, sigma


def train_and_test_sin_function():

    # column vector of input samples
    # training_samples = np.transpose(np.arange(0,2*np.pi, 0.1))
    training_samples = np.transpose(np.linspace(0, 2*np.pi, num=20*np.pi))


    # training set true values
    training_sin = np.asarray([np.sin(2*x) for x in training_samples])


    plt.plot(training_samples, training_sin)
    # plt.savefig("sin(2x)_training.png")
    plt.show()

    found1=False
    found2=False
    found3=False

    # validation_samples = np.transpose(np.arange(0.05, 2 * np.pi, 0.1))
    validation_samples = np.transpose(np.linspace(0.05, 2*np.pi, num=20*np.pi))


    # validation set true values
    validation_sin = np.asarray([np.sin(2 * x) for x in validation_samples])


    for n in range(4, 25):

        if found1:
            break
        rbfs, sigma = random_RBFs(training_samples, N=n)
        # rbfs, sigma = k_means_RBFS(training_samples, N=n)

        phi = create_Phi_matrix(training_samples, rbfs, sigma)

        # weights = np.dot(np.linalg.pinv(phi), training_sin.T)
        weights = np.dot(np.linalg.pinv(phi), training_sin)

        debug=0

        for epoch in range(100):

            new_phi = create_Phi_matrix(validation_samples, rbfs, sigma)
            # predictions = np.dot(new_phi, weights.T)
            predictions = np.dot(phi, weights)
            # total_residual_error = np.sum(np.abs(validation_sin-predictions))/validation_sin.shape[0]
            total_residual_error = np.sum(np.abs(training_sin-predictions))/training_sin.shape[0]

            if total_residual_error < 0.001 and not(found1):
                print("Total residual error fell under 0.001 for "+str(n)+" hidden units:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found1 = True
                plt.plot(validation_samples, predictions,'bs', label="Predicted points")
                plt.plot(validation_samples, validation_sin ,'r--', label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function for target error<0.001")
                #plt.savefig("k-means sin(2x)_error_0_001.png")
                plt.show()
                plt.clf()
                break

            if total_residual_error < 0.01 and not(found2):
                print("Total residual error fell under 0.01 for "+str(n)+" hidden units:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found2 = True
                plt.plot(validation_samples, predictions,'bs', label="Predicted points")
                plt.plot(validation_samples, validation_sin ,'r--', label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function for target error<0.01")
                # plt.savefig("k-means sin(2x)_error_0_01.png")
                plt.show()
                plt.clf()
                break

            if total_residual_error < 0.1 and not(found3):
                print("Total residual error fell under 0.1 for "+str(n)+" hidden units:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found3 = True
                plt.plot(validation_samples, predictions, label="Function approximation")
                plt.plot(validation_samples, validation_sin , label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function for target error<0.1")
                # plt.savefig("k-means sin(2x)_error_0_1.png")
                plt.show()
                plt.clf()
                break

            # print(total_residual_error)

            rbfs, sigma = random_RBFs(training_samples, N=n)
            #rbfs, sigma = k_means_RBFS(training_samples, N=n)
            phi = create_Phi_matrix(training_samples, rbfs, sigma)

            # weights = np.dot(np.linalg.pinv(phi), training_sin.T)
            weights = np.dot(np.linalg.pinv(phi), training_sin)

'''square 2x function approximation'''

def train_and_test_square_function():

    # column vector of input samples
    #training_samples = np.transpose(np.arange(0,2*np.pi, 0.1))
    training_samples = np.transpose(np.linspace(0, 2*np.pi, num=20*np.pi))


    # training set true values
    training_squares = np.asarray([square_function(2*x) for x in training_samples])

    # found0=False
    found1=False
    found2=False
    found3=False

    #validation_samples = np.transpose(np.arange(0.05, 2 * np.pi, 0.1))
    validation_samples = np.transpose(np.linspace(0.05, 2*np.pi, num=20*np.pi))


    # validation set true values
    validation_squares = np.asarray([square_function(2*x) for x in validation_samples])

    for n in range(2, 100):

        if found3:
            break

        # rbfs, sigma = random_RBFs(training_samples, N=n)
        rbfs, sigma = k_means_RBFS(training_samples, N=n)
        phi = create_Phi_matrix(training_samples, rbfs, sigma)

        weights = np.dot(np.linalg.pinv(phi), training_squares)

        for epoch in range(100):

            # new_phi = create_Phi_matrix(validation_samples, rbfs, sigma)
            predictions = np.dot(phi, weights.T)
            # thresholded output
            # predictions = np.asarray([np.sign(x) for x in predictions])
            total_residual_error = np.sum(np.abs(training_squares-predictions))/training_squares.shape[0]

            # if total_residual_error==0 and not(found0):
            #     print("Total residual error is zero by using" + str(n) + "hidden layers")
            #     found0 = True

            if total_residual_error < 0.001 and not(found1):
                print("Total residual error fell under 0.001 for "+str(n)+" hidden units:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found1 = True
                # plt.plot(validation_samples, predictions,'bs', label="Predicted points")
                plt.plot(validation_samples, validation_squares ,'r--', label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating square(2x) function for target error<0.001")
                # plt.savefig("k-means_square(2x)_error_0_001.png")
                plt.clf()
                plt.show()
                break

            if total_residual_error < 0.01 and not(found2):
                print("Total residual error fell under 0.01 for "+str(n)+" hidden units:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found2 = True
                plt.plot(validation_samples, predictions,'bs', label="Predicted points")
                plt.plot(validation_samples, validation_squares ,'r--', label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating square(2x) function for target error<0.01")
                # plt.savefig("square(2x)_error_0_01.png")
                plt.show()
                plt.clf()
                break

            if total_residual_error < 0.1 and not(found3):
                print("Total residual error fell under 0.1 for "+str(n)+" hidden units:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found3 = True
                plt.plot(validation_samples, predictions, label="Function approximation")
                plt.plot(validation_samples, validation_squares , label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating square(2x) function for target error<0.1")
                # plt.savefig("k-means square(2x)_error_0_1.png")
                plt.show()
                plt.clf()
                break

            # print(total_residual_error)

            # rbfs, sigma = random_RBFs(training_samples, N=n)
            rbfs, sigma = k_means_RBFS(training_samples, N=n)
            phi = create_Phi_matrix(training_samples, rbfs, sigma)

            weights = np.dot(np.linalg.pinv(phi), training_squares)

# train_and_test_sin_function()
train_and_test_square_function()