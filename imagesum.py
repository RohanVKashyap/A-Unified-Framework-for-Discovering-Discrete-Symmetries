from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#import matplotlib
import tensorflow as tf
#import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Conv1D, Activation, Lambda, Permute, Conv2D, Flatten
from tensorflow.keras.models import Model, load_model
from keras.constraints import maxnorm, nonneg
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
#from tensorflow.keras.utils.vis_utils import model_to_dot
from tqdm import tqdm,trange
import gc
from itertools import product,combinations 

import pandas as pd
import sys
from keras.constraints import maxnorm, nonneg
import time

from math import comb
from sklearn.linear_model import Ridge
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error

def main():
    def my_regularizer(x):
       return 1e-3 * tf.reduce_sum(tf.square(x))

    def save_pd(history, fileName):
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 

        # save to json:  
        hist_json_file = fileName + '_history.json' 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

        # save to csv: 
        hist_csv_file = fileName + '_history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

    features = np.load("./feature.npy")
    all_labels = np.load("./all_labels.npy")
    print("shape info:",[features.shape, features.dtype, all_labels.shape])

    image_height = 28
    image_width = 28

    patience = 10
    def run_model(model, prefix, basepath, idx, train_data, val_data, test_data, max_epochs=200):
        checkpointFile = prefix + str(idx) + '_model_weights.h5'
        checkpointFile = os.path.join(basepath, checkpointFile)
        tensorBoardLogDir = os.path.join(basepath, prefix + str(idx) + '_logs')
        csvLogs = os.path.join(basepath, prefix + str(idx) + '_training.log')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=patience, min_lr=0.000001)

        #EarlyStopping(patience=10)

        my_callbacks = [
            ModelCheckpoint(filepath=checkpointFile, 
                            save_best_only=True,
                            save_weights_only=True),
            TensorBoard(log_dir=tensorBoardLogDir),
            CSVLogger(csvLogs),
            reduce_lr]


        #get_deepset_discovery_model_conv() # 

        model_train_history = model.fit(train_data[0], 
                                        train_data[1], 
                                        epochs=max_epochs, batch_size=128,
                                        shuffle=True, validation_data=val_data,
                                        callbacks=my_callbacks) 


        train_history = os.path.join(basepath, prefix + str(idx) + '_train')
        save_pd(model_train_history, train_history)

        # Testing
        model.load_weights(checkpointFile)
        model_test_history = model.evaluate(x = test_data[0], 
                                                y = test_data[1])
        print("test error:",[prefix, idx, model_test_history])
        #test_errors_5_7_9[idx, 0] = model_test_history
        return model_test_history
    
    def get_matrix(n):
        x=np.arange(n)
        M1=np.zeros((n*(n-1),n))
        M2=np.zeros((n*(n-1),n))
        
        r=np.array([(i,j) for (i,j) in product(x,repeat=2) if i!=j])
        M1_idx=r[:,0]
        M2_idx=r[:,1]
        M1[np.arange(len(M1)),M1_idx]=1
        M2[np.arange(len(M2)),M2_idx]=1
        return M1,M2
    
    def get_deepset_discovery_model(l1_lambda):
        l1_reg = tf.keras.regularizers.l1(l1_lambda)
        
        input_img = Input(shape=(10, 28*28))
        #Input(batch_size =128, shape=(10, 28*28))
        
        x = Permute((2, 1))(input_img)
    
        x = Dense(10, kernel_regularizer=l1_reg, kernel_constraint=nonneg())(x)  
        x = Permute((2, 1))(x)
        
        x = Dense(300, activation='tanh')(x)       
        x = Dense(100, activation='tanh')(x)  
        x = Dense(30, activation='tanh')(x)
        
        Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
        
        x = Adder(x)
        #print(x.shape)
        
        encoded = Dense(1)(x)
        
        summer = Model(input_img, encoded)
        #adam = Adam(learning_rate=1e-3, epsilon=1e-3)
        #summer.compile(optimizer=adam, loss='mae')
        #summer.get_layer(index=1).set_weights([images])
        return summer
    
    num_train_examples = 70000
    num_val_examples = 25000
    num_test_examples = 30000
    max_train_length = 10
    image_height = 28
    image_width = 28
    total_examples = num_train_examples + num_val_examples + num_test_examples
    d = 10
    all_indices = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    basepath = "./discover_v1_Sk10_trails_1"

    indices = all_indices[0]
    labels = all_labels[:,np.array(indices)]
    labels = np.sum(labels,1)            

    train_data = (features[0:num_train_examples], labels[0:num_train_examples])

    val_data = (features[num_train_examples:num_train_examples+num_val_examples],
                labels[num_train_examples:num_train_examples+num_val_examples])

    test_data = (features[num_train_examples+num_val_examples:],
                labels[num_train_examples+num_val_examples:])

    model_discover = get_deepset_discovery_model(0.01)
    #M1, M2 = get_matrix(d)

    sample_y = np.random.randn(1, 10, 784)
    sample_output = model_discover(sample_y)

    model_discover.layers[2].trainable = False
    #model_discover.layers[3].trainable = False
    #model_discover.layers[4].trainable = False

    adam = Adam(learning_rate=1e-3, epsilon=1e-3)
    model_discover.compile(optimizer=adam, loss='mae')

    initial_weights = model_discover.get_weights()
    model_discover.set_weights(initial_weights)

    #adam = Adam(learning_rate=1e-3, epsilon=1e-3)
    #model_discover.compile(optimizer=adam, loss='mae')

    #model_discover.layers[3].set_weights([M1.T])
    #model_discover.layers[4].set_weights([M2.T])
    print(model_discover.summary())

    def new_matrix(k,n,indices):
        x = np.zeros((n,n))
        for row,index in zip(np.arange(k),indices):
          x[row,index] = 1
        return x
    
    def generate_matrix():
        k_=int(np.random.choice(d,1,replace=False))
        train_indices=np.sort(np.random.choice(list(range(0,d)),k_,replace=False))
        #Generate matrix
        matrix=new_matrix(len(train_indices),d,train_indices)
        #print('Matrix:',matrix)
        return matrix, train_indices
    
    def generate_matrix_given_indices(train_indices):
        #Generate matrix
        matrix=new_matrix(len(train_indices),d,train_indices)
        #print('Matrix:',matrix)
        return matrix

    filepath = 'saved_model.h5'
    callback = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                  save_best_only=True,
                                                  save_weights_only=True,)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=patience, min_lr=0.000001)

    def get_loss(M):
        model_discover.set_weights(initial_weights)
        #sample_output=Model_discover(val_ds)
        bias=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        model_discover.layers[2].set_weights([M.T,bias])

        start=time.time()
        train_history = model_discover.fit(train_data[0], 
                                           train_data[1], 
                                           epochs=2, 
                                           batch_size=128,
                                           shuffle=True,
                                           validation_data=val_data,
                                           callbacks=[callback,reduce_lr]) 
        
        end=time.time()
        print('Time:',end-start)
        
        return train_history
    
    C = [list(combinations(range(d),i)) for i in range(1,d+1)]
    C = [list(item) for sublist in C for item in sublist]

    At = np.zeros((len(C), d))
    for i, idx in zip(range(len(C)), C):
        At[i][idx] = 1

    l2_norms_rows = LA.norm(At, 2, axis=1)[:, np.newaxis]
    At = At/l2_norms_rows

    At = np.hstack((At, np.ones((At.shape[0], 1))))
    print('At shape:',At.shape)

    R = 0.01
    epsilon = 0.5
    delta = 0.5
    n_features = d+1
    B = np.eye(n_features)
    B_inv = np.eye(n_features)
    f = np.zeros((n_features, 1))
    mu_hat = np.zeros((n_features, 1))
    arm_iterations = 300
    v = R * np.sqrt(24 / epsilon * n_features * np.log(1 / delta))
    contexts = At
    all_rewards = []
    all_train_indices = []
    all_mu_hat = []
    all_y_hat = []

    for i in np.arange(arm_iterations):
        print("Iteration:",i)
        #Sample mu
        mu_tilde = np.random.multivariate_normal(mu_hat.flat, v**2 * B_inv)[..., np.newaxis]
        costs = contexts.dot(mu_tilde)
        
        #Pick best arm
        choosen_arm = np.argmax(costs)
        contexts_t = contexts[choosen_arm]

        #Generate matrix
        train_indices = np.where(contexts_t[:-1] > 0)[0].tolist()
        matrix = generate_matrix_given_indices(train_indices)

        #Get reward
        train_history = get_loss(matrix)
        min_loss = np.min(train_history.history['val_loss'])
        reward = -min_loss

        #Updates
        contexts_t = np.reshape(contexts_t,(-1,1))
        B += np.dot(contexts_t, contexts_t.T)
        B_inv = np.linalg.inv(B)
        f += contexts_t * reward
        mu_hat = B_inv.dot(f)

        y_hat = np.dot(contexts_t.T,mu_hat)
        all_y_hat.append(y_hat) 
        all_mu_hat.append(mu_hat)

        all_rewards.append(reward)
        all_train_indices.append(train_indices)
        print('loss:',mean_squared_error(all_rewards,[h.flatten()[0].tolist() for h in all_y_hat]))
        print([min_loss,train_indices])
    
    np.save('d_rewards.npy',np.array(all_rewards))
    np.save('d_y_hat.npy',np.array(all_y_hat))
    np.save('d_mu_hat.npy',np.array(all_mu_hat))
    np.save('d_train_indices.npy',np.array(all_train_indices))

    l=[]
    for a in contexts:
        l.append((np.round(np.dot(np.reshape(a,(1,-1)),mu_hat)[0][0],4),np.where(a[:-1]>0)[0].tolist())) 

    sort_l = sorted(l,key=lambda x:x[0],reverse=True) 

    true_indices = all_indices[0]
    top_indices = [b for a,b in sort_l[:5]]  
    if true_indices in top_indices:
        print(True)

    print(sort_l)    

if __name__=='__main__':
    main()
