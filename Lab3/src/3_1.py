import numpy as np
import matplotlib.pyplot as plt


def getData():
    #--- memory patterns
    x1 = [-1, -1, 1, -1, 1, -1, -1, 1] 
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1] 
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
    #--- inputs
    X = np.vstack([x1, x2, x3])

    #--- distorted patterns
    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]
    Xd = np.vstack([x1d, x2d, x3d])

    #--- more than half wrong
    x1h = [-1, 1, 1, 1, 1, -1, -1, 1]
    x2h = [1, -1, 1, 1, -1, 1, -1, -1]
    x3h = [1, 1, -1, -1, 1, 1, -1, 1]
    Xh = np.vstack([x1h, x2h, x3h])

    return X, Xd, Xh
    
def weightMat(X):
    #--- Weight matrix dimension
    P = len(X[0])
    #--- initialise the weight matrix
    W = np.zeros((P, P))
    #--- iterate over each input x and calculate the weight matrix
    for x in X:
        w = np.outer(x, x.T)
        #--- fill the diagonal with zeros since it contains NO information
        # np.fill_diagonal(w, 0)
        W += w
    return W

def update(W, original, activation, num_iter, update):
    
    if update == 'Synchronous':    
        patterns = activation
        activation_ls = np.asarray(patterns)
        
        print('Original Data')
        print(original)
        print()
        print('Noisy Data')
        print(activation)
        print()
        
        for n in range(num_iter):
            tmp = np.copy(activation_ls)
            for pat in range(len(patterns)):
                activation = np.where(np.sum(W * tmp[pat], axis=1) > 0, 1, -1)
                activation_ls[pat] = activation
        
        
        print('Prediction')
        print(activation_ls)

        if np.array_equal(original, activation_ls):
            print()
            print('Stable fixed point ---> Reached')
            print('Attractors ---> x1, x2, x3')
        else:
            print()
            print(np.equal(original, activation_ls))

    elif update == 'Asynchronous':
        pass
    
def main():
    #--- inputs
    inputs, inputs_dis, inputs_half = getData()
    #--- calculate the weight matrix using the original inputs
    W = weightMat(np.copy(inputs))
    #--- update step using noisy inputs
    num_iter = 100
    update(W, np.copy(inputs), np.copy(inputs_half), num_iter, update = 'Synchronous')
    
            

        
        
    
    
    

if __name__ == "__main__":
    main()