import tensorflow as tf
from sklearn.model_selection import train_test_split
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import time 


def create_training_set():
    # column vector of input samples
    # training_samples = np.transpose(np.arange(0, 2 * np.pi + 0.1, 0.1))
    training_samples = np.transpose(np.linspace(0, 2*np.pi, num=20*np.pi))

    # training set true values
    training_sin = np.asarray([np.sin(2 * x) for x in training_samples]) + (0.1**2)*np.random.randn(training_samples.shape[0],)+0

    return training_samples, training_sin


def create_validation_set():
    # column vector of input samples
    # validation_samples = np.transpose(np.arange(0.05, 2 * np.pi + 0.1, 0.1))
    validation_samples = np.transpose(np.linspace(0.05, 2*np.pi, num=20*np.pi))

    # training set true values
    validation_sin = np.asarray([np.sin(2 * x) for x in validation_samples]) + (0.1**2)*np.random.randn(validation_samples.shape[0],)

    return validation_samples, validation_sin


class MLPRegr():
    '''
    MLP for time series prediction
    '''
    def __init__(self, X, Y, X_test, Y_test):
        '''
        X: array shape[# of samples, # of features]
        Y: array shape[# of samples, # of responses]      
        '''
        # self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, Y, shuffle = False, test_size = 0.3)
        # self.X_test = X_test
        # self.Y_test = Y_test
        self.X_train = X
        self.y_train = Y
        self.X_val = X_test
        self.y_val = Y_test

        
    def net(self, layers, eta, epochs, earlyStop, regularization):
        '''
        layers:    list which contains the # of nodes in each layer. Note the length of the list is the # of layers
        eta:       learning rate
        epochs:    # of epochs
        '''
        W, b = [], []
        x = tf.placeholder(tf.float32, [None, 1], name = 'x')  
        y = tf.placeholder(tf.float32, [None, 1], name = 'y') 
        
        #--- define layers list which contains the # of nodes in each layer. Note the length of the list is the # of layers
        layers.append(self.y_train.shape[1])
    
        #--- define weights: 1st layer to hidden
        W.append(tf.Variable(tf.random_normal([1, layers[0]], stddev = 0.01)))
        b.append(tf.Variable(tf.random_normal([layers[0]], stddev = 0.01)))
        
        #--- compute output: 1st layer to hidden
        out = tf.nn.relu(tf.add(tf.matmul(x, W[0]), b[0]))
        for l in range(1, len(layers)):
            #--- define weights: hidden layer to hidden or output
            W.append(tf.Variable(tf.random_normal([layers[l - 1], layers[l]], stddev = 0.01)))
            b.append(tf.Variable(tf.random_normal([layers[l]], stddev = 0.01)))
            #--- compute output
            out = tf.add(tf.matmul(out, W[l]), b[l])
            
        
        #--- loss function (MSE)
        mse = tf.losses.mean_squared_error(labels = y, predictions = out)  
        
        ''' Regularization '''
        if regularization == 'L2 Reg':
            reg = 0
            beta = 0.01
            for l in range(0, len(layers)):
                reg += tf.nn.l2_loss(W[l])
            loss = tf.reduce_mean(mse + beta * reg)
        else:
            #--- Think to add different regularization techniques as well
            loss = mse

        
        # loss = 0.5 * tf.reduce_sum(tf.square(loss))
        #--- backprop
        for i in range(len(W)):
            backprop = [
                tf.assign(W[i], W[i] - eta * loss),
                tf.assign(b[i], b[i] - eta * loss)
            ]


        #--- optimizer (Adam)
        optimizer = tf.train.AdamOptimizer(learning_rate = eta).minimize(loss)
        
        # initiate the variables defined above (tf.Variable(...))
        init_var = tf.global_variables_initializer()
        
        #--- start session
        with tf.Session() as sess:
            # initialize variables
            sess.run(init_var)
            cool = 20
            cool_count = 0
            minDiff = 0.01
            train_error, val_error = [], []
            
            for epoch in range(epochs):
                time_start = time.clock()
                avg_val_errr = 0
                
                sess.run(backprop, feed_dict = {x: self.X_train, y: self.y_train})
                _, cost = sess.run([optimizer, loss], feed_dict = {x: self.X_train, y: self.y_train})
                train_error.append(cost) 
                val_error.append(sess.run(loss, feed_dict = {x: self.X_val, y: self.y_val}))
                
                

                # if epoch % 1 == 0:
                print('Epoch:', (epoch +1), 'train_error =', '{:.3f}'.format(train_error[epoch]))
                print('         validation_error =', '{:.3f}'.format(val_error[epoch]))
                print('-'*20)
            
                ''' Early stop '''
                if earlyStop == 'On':    
                    if epoch > 0 and (val_error[epoch - 1] - val_error[epoch]) > minDiff:
                        cool_count = 0
                    else:
                        cool_count += 1
                    if cool_count > cool and train_error[epoch] < 0.01:
                        print("It's time to stop!"); print()
                        break

            time_elapsed = (time.clock() - time_start)
            
            print('Computational cost (time) = ', time_elapsed)
            # print('Test_error =', '{:.3f}'.format(sqrt(sess.run(loss, feed_dict = {x: self.X_test, y: self.Y_test}))))
            pred = sess.run(out, feed_dict = {x:self.X_val})
            plt.plot(pred, 'b', label = 'Prediction')
            plt.plot(self.y_val, 'r', label = 'Original')
            plt.legend(('Prediction', 'Original'), loc='upper right')
            plt.show()
            
            #--- Visualize results
            plt.plot(train_error, 'g', label = 'Training')
            plt.plot(val_error, 'b', label = 'Validation')
            plt.legend(('Training error', 'Validation error', 'Test error'), loc='upper right')
            plt.xlabel("Epochs")
            plt.title('Training VS Validation error')
            plt.show()


def main():
    '''
    change std for the Gaussian Noise and set noise flag to On/Off
    '''
    # mu = 0
    # std = 0.18
    # input_points, output, x_test, y_test = getData(mu, std, noise = 'Off')
    

    x_train, y_train = create_training_set()
    x_valid, y_valid = create_validation_set()

    x_tr = x_train.reshape(len(x_train),1)
    y_tr = y_train.reshape(len(y_train),1)
    x_val = x_valid.reshape(len(x_valid),1)
    y_val = y_valid.reshape(len(y_valid),1)
    
    mlp = MLPRegr(x_tr, y_tr, x_val, y_val)
    mlp.net(layers = [3, 3], eta = 0.005, epochs = 300, earlyStop = "On", regularization = 'None')
    
    
    
    
    

if __name__ == "__main__":
    main()