import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import losses
from keras import initializers
from keras import metrics
from colorama import Fore
from colorama import Style
from keras.utils.vis_utils import plot_model

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
        
        self.fig = plt.figure(1)
        
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
            ax = plt.subplot(10, 10, i + 1)
            plt.imshow(np.reshape(weights_ls[i], (28, 28)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.suptitle('Bottom-up  final weights for each hidden unit')
        
def getAutoencoder(img_dim, encd_lay_ls, decd_lay_ls):
    # input (784d) placeholder
    input_img = Input(shape=(img_dim, ))
    # initialize weights
    weight_init = initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None) # default 'glorot_uniform'(Xavier uniform initializer). Really good results
    
    encoded = input_img 
    for i in range(len(encd_lay_ls)):
        if i == len(encd_lay_ls) - 1:
            encoded = Dense(encd_lay_ls[i], activation='relu', use_bias=True, kernel_initializer=weight_init, 
                            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)                    
            decoded = Dense(decd_lay_ls[i], activation='sigmoid', use_bias=True, kernel_initializer=weight_init, 
                                bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                                activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)
        else:
            encoded = Dense(encd_lay_ls[i], activation='relu', use_bias=True, kernel_initializer=weight_init, 
                            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)                    
            decoded = Dense(decd_lay_ls[i], activation='relu', use_bias=True, kernel_initializer=weight_init, 
                                bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                                activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(encoded)
            
    # Input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # Input to encoded rep
    encoder = Model(input_img, encoded)
    # encoded input placeholder 
    encoded_input = Input(shape=(encd_lay_ls[-1], ))
    # get the last layer
    decoder_layer = autoencoder.layers[-1]
    # Output (decoder)
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    # plot of the topology of the net
    # plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return autoencoder, encoder, decoder   
        

def train_test(x_train, x_test, loss_f, encd_lay_ls, decd_lay_ls, opt, epochs, batch_size, earlyStop):
    # get the models
    autoencoder, encoder, decoder = getAutoencoder(len(x_train[0]), encd_lay_ls, decd_lay_ls,)
    # set optimizer and loss function and compile
    autoencoder.compile(optimizer=opt, loss=loss_f, metrics=[metrics.binary_accuracy])
    
    #--- train
    # for Tensorboard
    # tbCallBack = TensorBoard(log_dir="/tmp/autoencoder",
    #                          histogram_freq=0,
    #                          write_graph=True, 
    #                          write_images=True)
    plot_losses = PlotLosses()
    history = LossHistory()
    if earlyStop is 'On':
        earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        autoencoder.fit(x_train, x_train, batch_size, epochs, shuffle = True, 
                        validation_data=(x_test, x_test), callbacks=[plot_losses, earlyStopping, history])
    else:
        autoencoder.fit(x_train, x_train, batch_size, epochs, shuffle = True,
                         validation_data=(x_test, x_test), callbacks=[plot_losses, history])
    
    # get the optimal validation loss
    optimal_val_loss = history.losses[-1]
    # get the hidden to output weights
    final_weights = autoencoder.layers[-1].get_weights()[0]
    # test
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    
    return decoded_imgs, final_weights, optimal_val_loss



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
    #--- set optimizer and loss function
    ''' optimizers 
    control gradient clipping: clipnorm = 1., clipnorm = 0.5
    '''
    '''---> SGD: 
    # more epochs to converge
    # bigger learning rate
    '''
    # sgd = optimizers.SGD(lr=0.3, momentum=0., decay=0.) 
    '''---> Adam: '''
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.)
    loss_f = losses.mean_squared_error

    ''' set num of layers and hidden nodes '''
    img_dim = len(x_train[0])
    encd_lay_ls = [250,100, 50]
    decd_lay_ls = [100,250, img_dim]
    opt = adam
    epochs = 500
    batch_size = 250
    earlyStop='Off'

    ''' Train and Test '''
    # '1' for training and '2' for searching
    mode = input('For training press 1, For param searching press 2: ')
    if mode == '1':
        #--- training and testing
        decoded_imgs, final_weights, optimal_val_loss = train_test(x_train, x_test, loss_f, encd_lay_ls, decd_lay_ls,
                                                                   opt, epochs, batch_size, earlyStop)
    elif mode == '2':
        #--- search for the optimal number of nodes
        unit_ls = [50, 75, 100, 150]
        val_losses_ls = []
        for units in unit_ls:
            decoded_imgs, final_weights, optimal_val_loss = train_test(x_train, x_test, loss_f, encd_lay_ls, decd_lay_ls, 
                                                                       opt, epochs, batch_size, earlyStop)
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
    vis(x_test, decoded_imgs, n=20)
    weight_vis(final_weights)
    plt.show()
    

    

if __name__ == '__main__':
    main()