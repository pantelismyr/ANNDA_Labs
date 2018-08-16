import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Params_:
    '''
    Get parameters
    Dsize: # of points per class (int)
    eta: step length (float)
    nodes: # of nodes 
    outputNodes: # of nodes of the output
    alpha: how much the old weight update vector contributes to the new update (momentum)
    n_epochs: # of epochs (int)
    mu: means of the Dataset (list: 1 x #Classes) 
    cov: covariance matrix (array: #Classes x #Classes) Note: we kept it same for both classes
    '''
    def __init__(self, Dsize, eta, nodes, outputNodes, alpha, n_epochs, mu, cov):
        self.Dsize = Dsize
        self.eta = eta
        self.nodes = nodes
        self.outputNodes = outputNodes
        self.a = alpha
        self.n_epochs = n_epochs
        self.mu = mu
        self.cov = cov


def dataGen(muA, muB, cov, inputbias, Dsize, vis, mode):
    '''
    Draw two sets of points in 2D from multivariate normal distribution
    Params:
    muA,muB: means of 2 classes
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
        # np.random.seed(0)
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

def weightsInit(Ν, Μ):
    """
    params:
    N: # of columns of the input patterns
    M: # of the outputs (# of rows of the output)
    """
    # np.random.seed(5)
    return np.random.rand(Ν, Μ)

def transF(x):
    return (2 / (1 + np.exp(-x))) - 1

def transFprime(x):
    return ((1 + transF(x)) * (1 - transF(x))) / 2.


def sequential_multi_layer_training(X, X_test,W, V, T, T_test, eta=0.005, bias =1, epochs=100, momentum = 0.9):

    for epoch in range(epochs):

        error = []
        train_error = []
        test_error = []

        if (W.shape[0]==1):

            h = np.ndarray(shape=(W.shape[1]))
            h_star = np.ndarray(shape=(W.shape[1]))
            delta_h = np.ndarray(shape=(X.shape[0]))

            o = np.ndarray(shape=(X.shape[1]))
            o_star = np.ndarray(shape=(X.shape[1]))
            delta_o = np.ndarray(shape=(X.shape[1]))

            for input_point in range(X.shape[1]):

                for neuron in range(W.shape[1]):
                    sum_h_star= 0

                    for dimensionality in range(X.shape[0]):

                        sum_h_star+=W[0,dimensionality]*X[dimensionality,input_point]

                    h_star[neuron]= sum_h_star + bias
                    h[neuron]=transF(h_star[neuron])

                sum_o_star=0
                for output_neuron in range(V.shape[0]):
                    sum_o_star+=V[output_neuron]*h[output_neuron]

                o_star[input_point] = sum_o_star + bias

                if transF(o_star[input_point])>0:
                    o[input_point] = 1
                else:
                    o[input_point] = -1

                if o[input_point] != T[input_point]:
                    delta_o[input_point] = (o[input_point]-T[input_point])*transFprime(o[input_point])
                    error.append(1)
                else:
                    delta_o[input_point]=0
                    error.append(0)

                for output_neuron in range(V.shape[0]):
                    V[output_neuron] = momentum*V[output_neuron]-(1-momentum) *eta*delta_o[input_point]*h[output_neuron]

                for neuron in range(W.shape[1]):
                    delta_h[neuron] = delta_o[input_point]*transFprime(h_star[neuron])
                    W[neuron] = momentum*W[neuron] - (1-momentum)*eta*X[neuron,input_point]*delta_h[neuron]

            train_error.append((np.sum((o - T) ** 2)) / 2.)

            bias = np.asarray([np.ones([X_test.shape[1]])])
            hin = np.dot(W.T, X_test)
            hout = transF(hin)
            hout = np.concatenate([hout, bias])
            oin = np.dot(V.T, hout)
            out = transF(oin)
            test_error = out - T_test

    return W, T, train_error, test_error



def fwdPass(X, W, V):
    bias = np.asarray([np.ones([X.shape[1]])])
    #--- input signal
    hin = np.dot(W.T, X)
    #---  output signal
    hout = transF(hin)
    #--- add bias to the last row of the output signal
    hout = np.concatenate([hout, bias])
    #--- next layer
    oin = np.dot(V.T, hout)
    #--- output pattern
    out = transF(oin)
    
    return out, oin, hin, hout

def bwdPass(X, T, W, V, nodes):
    out, oin, hin, hout = fwdPass(X, W, V)
    #--- error signals
    delta = (out - T) * transFprime(oin)
    #--- next layer
    delta_h = np.dot(V, delta)[0:nodes, :] * transFprime(hin)

    return delta, delta_h, hout, out

def backProp(X, X_test, T, T_test, W, V, a, eta, nodes, epochs, test):
    dW, dV = 0., 0.
    W_ls, V_ls, train_error, test_error = [], [], [], []
    
    for epoch in range(epochs):
        print('epoch = ', epoch + 1)
        delta, delta_h, hout, out = bwdPass(X, T, W, V, nodes)

        #--- Weight update
        dW = (dW * a) - np.dot(delta_h, X.T) * (1 - a)
        dV = (dV * a) - np.dot(delta, hout.T) * (1 - a)
        W = W + dW.T * eta
        V = V + dV.T * eta
        W_ls.append(W)
        V_ls.append(V)
        
        #--- find the training error
        pred = np.where(out >= 0., 1, -1)
        train_error.append((np.sum((pred - T)**2))/2.)

        if test == 'On':
            #--- feed forward
            out_test, _, _, _ = fwdPass(X_test, W_ls[epoch], V_ls[epoch])
            #--- compute missclassified points and error ratio
            pred = np.where(out_test >= 0., 1, -1)
            test_error.append((np.sum((pred - T_test)**2))/2.)
    
    #--- find the best (minimum error) weights 
    min_error_idx = np.argmin(train_error)
    W_best = W_ls[min_error_idx]
    V_best = V_ls[min_error_idx]
    
    return W_ls, V_ls, W_best, V_best, train_error, test_error

def evaluate(X, T, W_best, V_best): 
    #--- feed forward with the best weights
    out_best, _, _, _ = fwdPass(X, W_best, V_best)
    #--- compute missclassified points and error ratio
    pred = np.where(out_best >= 0., 1, -1)
    missclf = np.count_nonzero(pred - T)
    errorRate = missclf / X.shape[1]
    #--- print the results
    print("Error rate = " '%.2f' % errorRate + '%')
    print("Missclassified Points: " + str(missclf) + " out of " + str(X.shape[1]))


def grid(X, classA, classB, W, V, title):
    '''
    Create mesh grid
    '''
    gridXrange = np.arange(-8, 10, 0.01)
    gridYrange = np.arange(-8, 10, 0.01)
    grid = np.asarray(np.meshgrid(gridXrange, gridYrange))
    grid = np.array([grid[0].flatten(), grid[1].flatten(), np.ones(len(grid[0].flatten()))])

    #--- Forward pass
    bias = np.array([np.ones(len(grid[0].flatten()))])
    l1 = transF(np.dot(W.T, grid))
    l1 = np.concatenate([l1, bias])
    l2 = np.dot(V.T, l1)
    l2 = np.where(transF(l2) >= 0., 1, -1)
    
    #--- Plot
    plt.pcolormesh(gridXrange, gridYrange, l2.reshape((len(gridXrange), len(gridYrange))), cmap = 'gist_ncar', alpha = 0.3)
    plt.contour(gridXrange, gridYrange, l2.reshape((len(gridXrange), len(gridYrange))), levels = [0])
    plt.plot(X[0],X[1],'k.',markersize = 20)
    plt.plot(classA[0],classA[1],'*r',markersize=10)
    plt.plot(classB[0],classB[1],'+g',markersize=10)
    plt.title(title)
    plt.show()


  

def main():
    ''' 2 layer Perceptron '''
    #--- get parameters                 @---> Note: Control the parameters here
    params = Params_(Dsize = 100, eta = 0.005, nodes = 8, outputNodes = 1, alpha = 0.9, n_epochs = 100,
                     mu = [[[-3, -3],[-3,-3]], [0, 1]], cov = np.array([[2,1], [1,2]]))
    # --- extra bias term
    intargets = np.ones([params.Dsize])
    #--- Get data.      @---> Note: change mode to 'Nonlinearsep' or 'linearsep' for (Non)Linearly-separable data
    X, T, classA, classB = dataGen(params.mu[0], params.mu[1], params.cov, intargets, params.Dsize, vis = 'Off', mode = 'Nonlinearsep' )  
    
    #--- Generate the TEST set
    test_size = 50
    intargets_test = np.ones([test_size ])
    X_test, T_test, classA_test, classB_test = dataGen(params.mu[0], params.mu[1], params.cov, intargets_test, test_size, vis = 'Off', mode = 'Nonlinearsep' )   
    
    #--- get the Weight matrices
    W = weightsInit(len(X) + 1, params.nodes)
    V = weightsInit(params.nodes + 1, params.outputNodes)

    # test = sequential_multi_layer_training2(X,W,X,T)
    
    # --- create the bias vectort (Used 2 different initialization due to the different sizes of X and X_test)
    bias = np.asarray([np.ones([X.shape[1]])])
    bias_test = np.asarray([np.ones([X_test.shape[1]])])
    
    #--- add bias to the last row of the patterns (input)
    X = np.concatenate([X, bias])
    X_test = np.concatenate([X_test, bias_test])

    #--- Backpropagation
    W_ls, V_ls, W_best, V_best, train_error, test_error = backProp(X, X_test, T, T_test, W, V, params.a, params.eta, params.nodes, params.n_epochs, test = 'On')
    
    #--- Print the results
    print('-' * 15 + ' Training:')
    evaluate(X, T, np.copy(W_best), np.copy(V_best))
    print()
    print('-' * 15 + ' Testing:')
    evaluate(X_test, T_test, np.copy(W_best), np.copy(V_best))
    print('-' * 15)

    plt.plot(train_error, 'b', label = 'Training')
    plt.plot(test_error, 'r', label = 'Testing')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(('Training error', 'Testing error'),
            loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title('Training VS Testing error')
    plt.show()   

    #--- Vis CLF for the test set
    grid(X_test, classA_test, classB_test, W_best, V_best, title = 'Test set')

if __name__ == "__main__":
    main()