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

def createNoisyData(pic, numDisUnits):
    noisyPic = np.copy(pic)
    np.random.seed(100)
    #--- generate a number of random indices into a list
    idx = random.sample(range(0, len(pic)), numDisUnits)
    for i in idx:
        noisyPic[i] *= -1
    return noisyPic


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

def computeRho(X):
    #--- Weight matrix dimension
    N = X[0].size
    #--- Num of patterns
    P = len(X)
    #--- compute the average activity
    for x in X:    
        activities = np.sum(x)
        activities += activities
    return activities / (N*P)
    

def weightNew(X, rho):
    #--- Weight matrix dimension
    N = X[0].size
    #--- initialise the weight matrix
    W = np.zeros((N, N))
    for x in X:
        w = np.outer((x - rho), (x.T - rho))
        #--- fill the diagonal with zeros since it contains NO information
        # np.fill_diagonal(w, 0)
        W += w
    return W
    
    
    

def energy(s, W):
    return -np.dot(np.dot(s.T, W), s)

def update(W, original, activation, theta, mode):
    patterns = np.copy(activation)
    upd = 1
    E_ls = []
    if mode == 'synchronous':    
        #--- visualize the images before the update
        vis(original, patterns)
        while True:
            patterns = np.where(np.dot(W, patterns) > 0, 1, 0)
            # patterns = 0.5 + 0.5 * np.where((np.sum((W * patterns.T) - theta, axis=1)) > 0, 1, 0)            
            
            #--- compute the Energy
            E = energy(patterns, W)
            if np.array_equal(original, patterns):
                print('-'*30)
                print('Num of updates: ', upd)
                print('Energy: ', E)
                print('-'*30)
                print(Fore.GREEN)
                print(' Stable fixed point ---->>> Reached')
                print(Style.RESET_ALL)
                print()
                vis(original, patterns)
                break
            elif upd % 10 == 0:
                print('-'*30)
                print('updates: %d' %upd)
                print('Energy = ', E)
                print('-'*30)
                vis(original, patterns)
            upd += 1
            
    elif mode == 'Asynchronous':
        neurons = len(activation)
        uniqIdx = set()
        count = 0

        vis(original, patterns)
        while True:
            #--- get a random index
            idx = random.randrange(neurons)
            if idx not in uniqIdx:
                uniqIdx.add(idx)
                #--- update in the random index
                patterns[idx] = 0.5 + (0.5 * np.where((np.sum(W[idx, :] * patterns - theta)) > 0, 1, 0))
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
                    print('-'*30)
                    print('updates: %d ' %(upd-1))
                    print('Energy = ', E)
                    print('-'*30)
                    vis(original, patterns)
                    uniqIdx.clear()
        
        #--- plot the Energy evolution
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        plt.plot(E_ls)
        plt.xlabel("Iterations")
        plt.ylabel("Energy")
        plt.show()
                        
        
def main():
    #--- Remember data is a list of lists
    data = getData("pict.dat")
    data = np.where(data>0, 1, 0)
    #--- Used first three pict to learn
    learn = data[:2]
    #--- Get noisy Data
    noisyPic = createNoisyData(np.copy(data[0]), numDisUnits=500)       
    #--- compute rho
    rho = computeRho(np.copy(learn))
    #--- calculate the weight matrix using the original inputs #--- For sparser patterns use rho = 0.05 or 0.01
    W = weightNew(np.copy(learn), rho)
    # np.set_printoptions(threshold=np.nan)
    # print(np.average(W))
    # return
    # #--- update step 
    # #--- Remember: change W / W_rand
    update(W, np.copy(data[0]), np.copy(noisyPic), theta = 0.0001, mode = 'Asynchronous')
   
    

if __name__ == "__main__":
    main()



    