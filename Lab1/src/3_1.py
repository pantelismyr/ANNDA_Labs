import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Params_:
    '''
    Get parameters
    Dsize: # of points/class (int)
    eta: step length (float)
    n_epochs: # of epochs (int)
    mu: means of the Dataset (list: 1 x #Classes) 
    cov: covariance matrix (array: #Classes x #Classes) Note: we kept it same for both classes
    '''
    def __init__(self, Dsize, eta, n_epochs, mu, cov):
        self.Dsize = Dsize
        self.eta = eta
        self.n_epochs = n_epochs
        self.mu = mu
        self.cov = cov


def dataGen(muA, muB, cov, inputbias, Dsize, vis, mode):
    '''
    Draw two sets of points in 2D from multivariate normal distribution
    Params:
    muA,muA: means of 2 classes
    cov: covariance matrix
    inputbias: bias, the extra row of ones in each class
    Dsize: # of points in each class
    vis: visualize the data, value = 'On' | 'Off'
    mode: Non(linear)-separable data, value = 'linearsep' | 'Nonlinearsep'
    '''
    if mode == 'linearsep':
        # class A
        classAx, classAy = np.random.multivariate_normal(muA[0], cov, Dsize).T
        classA = np.array([classAx, classAy, inputbias])

        # class B
        classBx, classBy = np.random.multivariate_normal(muB, cov, Dsize).T
        classB = np.array([classBx, classBy, -inputbias])

    elif mode == 'Nonlinearsep':
        # class A
        classAx_, classAy_ = np.random.multivariate_normal(muA[0], cov, int(Dsize/2)).T
        classAxx, classAyy = np.random.multivariate_normal(muA[1], cov, int(Dsize/2)).T
        classAx = np.concatenate([classAx_, classAxx])
        classAy = np.concatenate([classAy_, classAyy])
        classA = np.array([classAx, classAy, inputbias])
        
        # class B
        classBx, classBy = np.random.multivariate_normal(muB, cov, Dsize).T
        classB = np.array([classBx, classBy, -inputbias])
        
    # concatenate the classes A and B
    data = np.concatenate([classA, classB], axis=1)
    # shuffle the samples
    np.random.shuffle(data.T)

    # Store the training data in variables "patterns" and "targets"
    patterns = data[:2,:]
    targets = data[2]

    if vis == 'On':
        # visualize the sets
        plt.plot(patterns[0],patterns[1],'k.',markersize=20)
        plt.plot(classAx,classAy,'*r',markersize=5)
        plt.plot(classBx,classBy,'+g',markersize=5)
        plt.title("Linearly-separable Dataset")
        plt.show()

    return patterns, targets, classA, classB


def weightsInit(N,M):
    """
    params:
    N: # of columns of the input patterns
    M: # of the outputs (# of rows of the output)
    """
    return np.random.rand(M, N)



def perceptron(X, W, T, eta, epochs, classA, classB, vis):
    error_ls = []
    miss_ls = []
    for epoch in range(epochs):
        print('epoch =', epoch + 1)
        #--- calculate the prediction
        pred = np.dot(W, X)
        #--- set the values to be either -1 or 1
        pred = np.where(pred[0] >= 0., 1, -1)
        #--- compute the error
        error = pred - T
        #--- update
        update = -eta * np.dot(error, X.T)
        W += update
    
        if vis == 'On':
            # animation
            visAnim(W, X, classA, classB)

        #--- find the missclassified points and the error rate
        missclassified = np.count_nonzero(error)
        miss_ls.append(missclassified)
        error_ls.append(missclassified/X.shape[1])
    
    plt.show(block=False)
    return update, error_ls, miss_ls


def deltaRule(X, W, T, eta, epochs, classA, classB, vis):
    error_ls = []
    miss_ls = []
    for epoch in range(epochs):
        print('epoch =', epoch + 1)
        # --- calculate the prediction
        pred = np.dot(W, X)
        delta_error = pred - T
        # --- set the values to be either -1 or 1
        pred = np.where(pred[0] >= 0., 1, -1)
        # --- compute the error
        error = pred - T
        # --- update
        update = -eta * np.dot(delta_error, X.T)
        W += update

        if vis == 'On':
            # animation
            visAnim(W, X, classA, classB)

        # --- find the missclassified points and the error rate
        missclassified = np.count_nonzero(error)
        miss_ls.append(missclassified)
        error_ls.append(missclassified / X.shape[1])

    plt.show(block=False)
    return update, error_ls, miss_ls


def visAnim(W, X, classA, classB):
    # Visualize data separation (animation)
    linelenghth = np.sqrt(np.dot(W[0,:],W[0,:].T))*0.2
    plt.plot(X[0],X[1],'k.',markersize=20)
    plt.plot(classA[0],classA[1],'*r',markersize=5)
    plt.plot(classB[0],classB[1],'+g',markersize=5)
    # plt.plot(np.array([-W[0, 1], W[0, 1]-1/np.sqrt(W[0][1]**2 + W[0][1]**2)]) / linelenghth, np.array([W[0, 0], -W[0, 0]-1/np.sqrt(W[0][0]**2 + W[0][0]**2)]) / linelenghth)
    plt.plot(np.array([-W[0,1], W[0,1]])/linelenghth,np.array([W[0,0], -W[0,0]])/linelenghth)
    plt.title("Anime of 2 classes")
    plt.autoscale(enable=False)
    plt.pause(0.001)

def main():
    #--- get parameters
    params = Params_(Dsize = 100, eta = 0.0005 , n_epochs = 20, mu = [[[-4, -2],[4,2]], [2, 1]], cov = np.array([[1,0], [0,1]]))
    #--- extra bias term
    inputbias = np.ones([params.Dsize])
    '''
    Note: choose One of the following lines to uncomment for the paras {NOT(linearly-separable) data}
    '''
    #--- Linearly-separable data
    X, T, classA, classB = dataGen(params.mu[0], params.mu[1], params.cov, inputbias, params.Dsize, vis = 'Off', mode = 'linearsep' )    
    #--- NOT linearly-separable data
    # X, T, classA, classB = dataGen(params.mu[0], params.mu[1], params.cov, inputbias, params.Dsize, vis = 'Off', mode = 'Nonlinearsep' )   
    #--- get the Weight matrix   ---- Note: need modification for every different case
    numOutputs = 1
    W = weightsInit(len(X), numOutputs)
    #--- Delta rule
    update, error_ls, miss_ls = deltaRule(X, W, T, params.eta, params.n_epochs, classA, classB, vis = 'On')
    #--- Results from Delta Rule
    
    #--- get the index with the lower error rate
    best_error_idx = np.argmin(error_ls)
    #--- get the # of missclassified points of the best index
    missclf = miss_ls[best_error_idx]
    #--- print results to stdout
    print('-' * 15)
    print("Error rate = " '%.2f' % error_ls[best_error_idx] + '%')
    print("Missclassified Points: " + str(missclf) + " out of " + str(X.shape[1]))
    print('-' * 15)
    #--- Visualize error 
    ax = plt.figure().gca()
    ax.plot(error_ls)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()


if __name__ == "__main__":
    main()