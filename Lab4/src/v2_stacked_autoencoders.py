import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, EarlyStopping
from keras import optimizers, losses, initializers, metrics
from colorama import Fore, Style
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
import matplotlib.patches as mpatches


def getData(filename):
    with open(filename, 'r') as f:
        # read csv file
        rows = csv.reader(f)
        # create a list of lists
        data = list(rows)
        # convert strings to integers
        data = [[int(i) for i in row] for row in data]
    return data

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        plt.plot(self.x, self.losses, 'b', label="loss")
        plt.plot(self.x, self.val_losses, 'r', label="val_loss")
        plt.legend(['loss', 'val_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss evolution')
        
        
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))

def vis(x_test, pred_imgs, n):
    plt.figure(2, figsize=(n, 5))
    i = 0
    count = 0
    for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(np.reshape(x_test[i], (28, 28)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(np.reshape(pred_imgs[i], (28, 28)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            i += 1
    plt.suptitle('Original (top) vs Reconstructed (bottom) images')

def weight_vis(final_weights):
    plt.figure(3, figsize=(10, 6))
    # puth the weights into a list for convenience
    weights_ls = final_weights.tolist()
    for i in range(len(weights_ls)):
            ax = plt.subplot(15, 15, i + 1)
            plt.imshow(np.reshape(weights_ls[i], (28, 28)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.suptitle('Bottom-up  final weights for each hidden unit')

def plotCLF(pred, y):
    colors = {0: 'b', 1: 'g', 2: 'r', 3:'c', 4:'m',
          5:'y', 6:'k', 7:'orange', 8:'darkgreen', 9:'maroon'}

    markers = {0: 'o', 1: '+', 2: 'v', 3:'<', 4:'>',
          5:'^', 6:'s', 7:'p', 8:'*', 9:'x'}

    plt.figure(5, figsize=(10, 10))
    patches = []
    for i in range(300):
        point = pred[i]
        target = y[i][0]
        plt.plot(point[0], point[1], color= colors[target], marker=markers[target], markersize=8)
        
def getAutoencoder(img_dim, encd_lay_ls, decd_lay_ls, num_labels, clf):
    # input (784d) placeholder
    input_img = Input(shape=(img_dim, ))
    # initialize weights
    # weight_init = initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None) # default 'glorot_uniform'(Xavier uniform initializer). Really good results
    weight_init = initializers.glorot_uniform()
    encoded = input_img 
    for i in range(len(encd_lay_ls)):
        encoded = Dense(encd_lay_ls[i], activation='sigmoid', use_bias=True, kernel_initializer=weight_init, 
                            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded) 
    
    # Note: We integrated the code phase into encode for convinience - Next commented lines left for future use
    # code = Dense(code_sz, activation='sigmoid', use_bias=True, kernel_initializer=weight_init, 
    #                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
    #                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)
    
    decoded = encoded
    for i in range(len(decd_lay_ls)):
        if i == len(encd_lay_ls) - 1:
            decoded = Dense(decd_lay_ls[i], activation='sigmoid', use_bias=True, kernel_initializer=weight_init, 
                            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(decoded)
        else:
            decoded = Dense(decd_lay_ls[i], activation='sigmoid', use_bias=True, kernel_initializer=weight_init, 
                            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(decoded)

    # Input to its reconstruction
    autoencoder = Model(input_img, decoded)
    
    #--- Classification
    clf = Model(input_img, encoded)
    # output layer
    out = Dense(num_labels, activation='softmax')(encoded)
    reduced = Model(input_img, out)
   
    # Uncomment next line to save a fig with the topology of the net
    # plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return autoencoder, reduced
        

def train_test(x_train, x_test, y_train, y_test, loss_f, encd_lay_ls, decd_lay_ls, opt, epochs, batch_size, earlyStop, num_labels, clf):
    
    '''--------------- Pre - training ---------------'''

    # get the models
    autoencoder, reduced = getAutoencoder(len(x_train[0]), encd_lay_ls, decd_lay_ls, num_labels, clf)
    # set optimizer and loss function and compile
    autoencoder.compile(optimizer=opt, loss=loss_f, metrics=[metrics.binary_accuracy])

    # edit callbacks
    plot_losses = PlotLosses()
    history = LossHistory()

    if earlyStop is 'On':
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00009, patience=5, verbose=0, mode='min')
        autoencoder.fit(x_train, x_train, batch_size, epochs, shuffle = True, 
                        validation_data=(x_test, x_test), callbacks=[plot_losses, earlyStopping, history])
    else:
        autoencoder.fit(x_train, x_train, batch_size, epochs, shuffle = True,
                         validation_data=(x_test, x_test), callbacks=[plot_losses, history])

    # plot reconstruction
    pred = autoencoder.predict(x_test)
    final_weights = autoencoder.layers[-1].get_weights()[0]
    vis(x_test, pred, n=20)
    weight_vis(final_weights)

    '''--------------- Classification ---------------'''

    train_y_cat = np_utils.to_categorical(y_train, num_labels)
    test_y_cat = np_utils.to_categorical(y_test, num_labels)

    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.)
    sgd = optimizers.SGD(lr=0.9, momentum=0., decay=0.) 
    opt = sgd

    reduced.compile(optimizer=opt, loss=loss_f, metrics=[metrics.binary_accuracy])
    score = reduced.evaluate(x_test, test_y_cat, verbose=1)
    print(Fore.GREEN)
    print('Test score before fine turning:', score[0])
    print('Test accuracy before fine turning:', score[1])
    print(Style.RESET_ALL)

    # earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00009, patience=3, verbose=0, mode='min')
    reduced.fit(x_train, train_y_cat, batch_size, epochs, shuffle = True, 
                        validation_data=(x_test, test_y_cat), callbacks=[plot_losses, EarlyStopping(min_delta=0.0001, patience=5), history])

    #--- score after fine tunig:
    score = reduced.evaluate(x_test, test_y_cat, verbose=1)
    print(Fore.GREEN)
    print('Test score after fine turning:', score[0])
    print('Test accuracy after fine turning:', score[1])
    print(Style.RESET_ALL)
    # get the optimal validation loss
    optimal_val_loss = history.losses[-1]
    # get the hidden to output weights
    final_weights = autoencoder.layers[-1].get_weights()[0]
    # test
    pred = autoencoder.predict(x_test)
    plotCLF(pred, y_test)
    
    return pred, final_weights, optimal_val_loss



def main():
    """
    Important Note: 
                    for the initialization, regularization and activation function tuning params,
                    go to 'getAutoencoder' function.
    """
    #--- get Data
    print('Loading Data...')
    x_train = getData("binMNIST_data/bindigit_trn.csv")
    y_train = getData("binMNIST_data/targetdigit_trn.csv")
    x_test = getData("binMNIST_data/bindigit_tst.csv")
    y_test = getData("binMNIST_data/targetdigit_tst.csv")
    print('Data loaded!')

    ''' --------------- set parameters --------------- '''
    img_dim = len(x_train[0])
    encd_lay_ls = [150, 75, 50]
    decd_lay_ls = [75, 150, img_dim]
    epochs = 10000
    batch_size = 128
    earlyStop = 'On'
    num_labels = 10
    clf = 'On'
    eta_adam = 0.001
    eta_SGD = 3
    ''' ---------------------------------------------- '''

    #--- set optimizer and loss function
    ''' optimizers 
    control gradient clipping: clipnorm = 1., clipnorm = 0.5
    '''
    '''---> SGD: 
    # more epochs to converge
    # bigger learning rate
    '''
    sgd = optimizers.SGD(lr=eta_SGD, momentum=0., decay=0.) 
    '''---> Adam: '''
    adam = optimizers.Adam(lr=eta_adam, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.)
    loss_f = losses.mean_squared_error
    #--- select optimizer
    opt = sgd


    ''' Train and Test '''
    # '1' for training and '2' for searching
    mode = input('For training press 1, For param searching press 2: ')
    if mode == '1':
        #--- training and testing
        pred, final_weights, optimal_val_loss = train_test(x_train, x_test, y_train, y_test, loss_f, encd_lay_ls, decd_lay_ls,
                                                                   opt, epochs, batch_size, earlyStop, num_labels, clf)
    elif mode == '2':
        #--- search for the optimal number of nodes
        unit_ls = [50, 75, 100, 150]
        val_losses_ls = []
        for units in unit_ls:
            pred, final_weights, optimal_val_loss = train_test(x_train, x_test, y_train, y_test, loss_f, encd_lay_ls, decd_lay_ls, 
                                                                       opt, epochs, batch_size, earlyStop, num_labels, clf)
            val_losses_ls.append(optimal_val_loss)
    
        print(val_losses_ls)
        best_model_idx = np.argmin(val_losses_ls)
        best_model_loss = val_losses_ls[best_model_idx]
        best_model_units = unit_ls[best_model_idx]
        print(Fore.GREEN)
        print('Best model :')
        print(Style.RESET_ALL)
        print('Nodes =', best_model_units)
        print('Loss =', "%.4f" % best_model_loss)
    else:
        print("Wrong input!")
        raise SystemExit
    

    ''' visualize results '''
    plt.show()
    

if __name__ == '__main__':
    main()