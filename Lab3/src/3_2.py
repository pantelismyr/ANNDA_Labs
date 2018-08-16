import numpy as np 
import matplotlib.pyplot as plt
import random
from colorama import Fore
from colorama import Style

def getData(filename):
    '''
    Read data from the file 
    and returns a list which contain 11 pictures (11 lists)
    '''
    with open(filename, 'r') as f:
        lines = f.read().split(',')
        lines = list(map(int, lines))
    
    P = []
    for i in range(0,len(lines),1024):
        P.append(lines[i : i + 1024])

    return np.asarray(P)

def vis(original, pict):
    # reshape these 1024-dim patterns as a 32 × 32 image.
    titles = ['Original', 'Predicted']
    original = np.reshape(original, (32,32))  
    pict = np.reshape(pict, (32,32))  
    im = [original, pict]
    fig1 = plt.figure(figsize = (20, 5))
    for idx in range(len(im)):
        ax = fig1.add_subplot(1, 2, idx+1) 
        ax.set_title(titles[idx])
        ax.imshow(im[idx].T)
        ax.axis('off')
    plt.pause(0.01)
    input("Press [enter] to continue.")

def weightMat(X):
    #--- Weight matrix dimension
    N = X[0].size
    #--- initialise the weight matrix
    W = np.zeros((N, N))
    #--- iterate over each input x and calculate the weight matrix
    for x in X:
        w = np.outer(x, x.T)
        #--- fill the diagonal with zeros since it contains NO information
        # np.fill_diagonal(w, 0)
        W += w
    return W / N

def randWeight(X):
    N = X[0].size
    # np.random.seed(5)
    return np.random.randn(N,N)
    # return np.random.normal(0, 0.001, (N,N))
    

def energy(s, W):
    return -np.dot(np.dot(s.T, W), s)

def update(W, original, activation, num_iter, mode):
    patterns = np.copy(activation)
    upd = 1
    E_ls = []
    if mode == 'Synchronous':    
        #--- visualize the images before the update
        vis(original, patterns)
        for n in range(num_iter):
            patterns = np.where(np.sum(W * patterns, axis=1) > 0, 1, -1)
            #--- compute the Energy
            E = energy(patterns, W)
            
            if np.array_equal(original, patterns):
                print('-'*30)
                print('Num of updates: ', n + 1)
                print('Energy: ', E)
                print('-'*30)
                print(Fore.GREEN)
                print(' Stable fixed point ---->>> Reached')
                print(Style.RESET_ALL)
                print()
                vis(original, patterns)
                break
            elif n % 10 == 0:
                print('updates: %d' %(n+1), 'Energy = ', E)
                vis(original, patterns)
            
    elif mode == 'Asynchronous':
        neurons = len(activation)
        uniqIdx = set()
        count = 0

        vis(original, patterns)
        try:
            while True:
                #--- get a random index
                idx = random.randrange(neurons)
                if idx not in uniqIdx:
                    uniqIdx.add(idx)
                    #--- update in the random index
                    patterns[idx] = np.where(np.dot(W[idx, :], patterns.T) > 0, 1, -1)
                    #--- compute the Energy
                    E = energy(patterns, W)
                    E_ls.append(E)
                    
                    if np.array_equal(original, patterns) and len(uniqIdx) == neurons:
                        print('-'*30)
                        print('Num of updates: ', upd)
                        print('Energy = ', E)
                        print('-'*30)
                        print(Fore.GREEN)
                        print(' Stable fixed point ---->>> Reached')
                        print(Style.RESET_ALL)
                        print()
                        vis(original, patterns)
                        break
                        
                    elif len(uniqIdx) == neurons:
                        upd += 1
                        # wrong_val = np.count_nonzero(original - patterns)
                        print('-'*30)
                        print('updates: %d ' %(upd-1))
                        print('Energy = ', E)
                        print('-'*30)
                        vis(original, patterns)
                        uniqIdx.clear()
        except KeyboardInterrupt:
        
            #--- plot the Energy evolution
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            plt.plot(E_ls)
            plt.xlim([0, len(E_ls)])
            plt.xlabel("Iterations")
            plt.ylabel("Energy")
            plt.show()
                        
        
def main():
    #--- Remember data is a list of lists
    data = getData("pict.dat.txt")
    #--- Used first three pict to learn
    learn = data[:3]
    #--- calculate the weight matrix using the original inputs
    # W = weightMat(np.copy(learn))
    #--- calculate the weight matrix using normally distributed random numbers
    W_rand = randWeight(np.copy(learn))
    #--- make the weight matrix symmetric
    W_rand = 0.5 * (W_rand + W_rand.T)
    #--- update step 
    num_iter = 100000
    #--- Remember: change W / W_rand
    update(W_rand, np.copy(data[0]), np.copy(data[9]), num_iter, mode = 'Asynchronous')
   
    

if __name__ == "__main__":
    main()



    