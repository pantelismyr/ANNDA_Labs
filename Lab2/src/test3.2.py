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

def random_RBFs(training_samples, N):

    rbfs = np.random.permutation(training_samples)[:N]
    sigma = ( rbfs.max() - rbfs.min() )/ np.sqrt(2*N)


    return rbfs, sigma

def k_means_RBFS(training_samples, N):

    rbfs = kmeans(training_samples, N)
    # rbfs = kmeans.fit(training_samples)
    sigma = ( rbfs.max() - rbfs.min() )/ np.sqrt(2*N)

    return rbfs, sigma

def create_training_set():

    training_samples = np.transpose(np.arange(0,2*np.pi, 0.1))
    training_sin = np.asarray([np.sin(2*x) for x in training_samples]) + np.random.normal(0,0.1,training_samples.shape[0])

    return training_samples, training_sin

def batch_train_and_test_sin_function():

    # column vector of input samples
    training_samples = np.transpose(np.arange(0,2*np.pi+0.1, 0.1))
    # training_samples+= np.random.normal(0,0.1,training_samples.shape[0])

    # training set true values
    #noise = np.random.normal(0,0.1,training_samples.shape[0])
    #training_sin = np.asarray([np.sin(2*x) for x in training_samples]) + np.random.normal(0,0.1,training_samples.shape[0])
    training_sin = np.asarray([np.sin(2*x) for x in training_samples]) + 0.1*np.random.randn(training_samples.shape[0],)
    debug=0

    plt.plot(training_samples, training_sin)
    plt.title("sin(2x) function with zero-mean noise added")
    #plt.savefig("sin(2x)_training_noise.png")
    plt.clf()
    plt.show()

    found1=False
    found2=False
    found3=False

    validation_samples = np.transpose(np.arange(0.05, 2*np.pi+0.1, 0.1))

    minimum_residual_error = 2

    # validation set true values
    validation_sin = np.asarray([np.sin(2 * x) for x in validation_samples]) + np.random.normal(0,0.01,validation_samples.shape[0])

    for n in range(4,21):

        if found1:
            break

        rbfs, sigma = k_means_RBFS(training_samples, N=n)
        #rbfs, sigma = random_RBFs(training_samples, N=n)
        phi = create_Phi_matrix(training_samples, rbfs, sigma)

        weights = np.dot(np.linalg.pinv(phi), training_sin.T)

        for epoch in range(100):

            predictions = np.dot(phi, weights.T)
            total_residual_error = np.sum(np.abs(training_sin-predictions))/len(training_sin)
            minimum_residual_error = min(minimum_residual_error, total_residual_error)
            debug=0

            if total_residual_error < 0.001 and not(found1):
                print("Total residual error fell under 0.001 for "+str(n)+"hidden layers:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found1 = True
                plt.plot(training_samples, predictions, label="Function approximation")
                plt.plot(training_samples, training_sin, label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function with noise added for target error<0.001")
                #plt.savefig("Noise_k-means_sin(2x)_error_0_001.png")
                plt.show()
                plt.clf()
                break

            if total_residual_error < 0.01 and not(found2):
                print("Total residual error fell under 0.01 for "+str(n)+"hidden layers:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found2 = True
                plt.plot(training_samples, predictions, label="Function approximation")
                plt.plot(training_samples, training_sin, label="True function")

                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function with noise added for target error<0.01")
                #plt.savefig("Noise_k-means_sin(2x)_error_0_01.png")
                plt.show()
                plt.clf()
                break

            if total_residual_error < 0.1 and not(found3):
                print("Total residual error fell under 0.1 for "+str(n)+"hidden layers:"+str(total_residual_error)+" after "+str(epoch)+" epochs")
                found3 = True
                plt.plot(training_samples, predictions, label="Function approximation")
                plt.plot(training_samples, training_sin, label="True function")
                plt.legend(loc='upper right')
                plt.ylim(-1.5,1.5)
                plt.title("Approximating sin(2x) function with noise added for target error<0.1")
                #plt.savefig("Noise_k-means_sin(2x)_error_0_1.png")
                plt.show()
                plt.clf()
                break

            # print(total_residual_error)

            #rbfs, sigma = random_RBFs(training_samples, N=n)
            rbfs, sigma = k_means_RBFS(training_samples, N=n)
            phi = create_Phi_matrix(training_samples, rbfs, sigma)

            weights = np.dot(np.linalg.pinv(phi), training_sin.T)

    print("minimum residual error is :"+str(minimum_residual_error))

def create_column_phi_matrix(x, rbfs, sigma):

    return np.asarray([gaussian_RBF(x-rbf, sigma) for rbf in rbfs]).T



def sequential_train_and_test_sin_function():

    inputs, sin = create_training_set()

    found1=False
    found2=False
    found3=False

    for hidden_nodes in range(4,21):

        if found1:
            break

        rbfs, sigma = k_means_RBFS(inputs, N=hidden_nodes)

        for epoch in range(10):

            for index in range(inputs.shape[0]):

                phi = create_column_phi_matrix(inputs[i],rbfs,sigma)





batch_train_and_test_sin_function()