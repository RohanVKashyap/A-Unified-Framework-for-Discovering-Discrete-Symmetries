import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.constraints import maxnorm, nonneg, unit_norm
import os
import gc
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Conv1D, Activation, Lambda, Permute, Conv2D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from itertools import permutations

from math import comb
from itertools import combinations
from sklearn.linear_model import Ridge
import time
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Changes: R = 0.1 (Default value is R = 0.01)
# Normalizing
# Add additional datapoints
# 9-5-2023
# Add additional datapoints to (val_ds, val_y)
# epochs = 250, batch_size = 4
# Add additional layers in the model
# np.save(f'c_{k}_10_0.1_val_data_r_ts_rewards.npy',np.array(all_rewards))
# All arms for Sk, Zk, D2k
# linear_regression_5_10.txt

def main(args):
    Z2 = [[0, 1], [1, 0]]
    Z4 = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
    Z5 = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1], [3, 4, 0, 1, 2], [4, 0, 1, 2, 3]]
    Z8 = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 0, 1, 2, 3, 4, 5, 6], [6, 7, 0, 1, 2, 3, 4, 5], [5, 6, 7, 0, 1, 2, 3, 4],
        [4, 5, 6, 7, 0, 1, 2, 3], [3, 4, 5, 6, 7, 0, 1, 2], [2, 3, 4, 5, 6, 7, 0, 1], [1, 2, 3, 4, 5, 6, 7, 0]]
        
    perm = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
            [3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2], [0, 3, 2, 1]]

    Z5_Z10 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 0, 1, 2, 3, 5, 6, 7, 8, 9],
            [3, 4, 0, 1, 2, 5, 6, 7, 8, 9], [2, 3, 4, 0, 1, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 5, 6, 7, 8, 9]]

    Z10 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],  [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],  [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],  [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0,]]
    
    def generate_per_matrix(n):
        #P = np.zeros((n,n)).astype(np.int64)
        atom = np.arange(n)
        P = np.array([np.roll(atom, shift=i) for i in np.arange(n)])
        return P

    def poly_Zk_Zn(x, indices):
        def inv1(a, b):
            return a * b ** 2

        unstacked_variables  = tf.unstack(x, axis=1)
        unstacked_variables = tf.gather(unstacked_variables, indices)

        q1 = 0
        for i in np.arange(len(indices)-1):
            q1 += inv1(unstacked_variables[i], unstacked_variables[i+1])
        q1 += inv1(unstacked_variables[len(indices)-1], unstacked_variables[0])        
        return q1

    def apply_layers(x, layers):
        for l in layers:
            x = l(x)
        return x

    def sigmaPi(fin, m, n, p):
        fin = tf.transpose(fin, (0, 2, 1, 3))
        fin = fin[:, :, tf.newaxis]
        fin = tf.tile(fin, (1, 1, m, 1, 1))
        y = fin @ p
        y = tf.linalg.diag_part(y)
        y = tf.reduce_prod(y, axis=3)
        y = tf.reduce_sum(y, axis=2)
        return y

    def prepare_permutation_matices(perm, n, m):
        p1 = np.eye(n, dtype=np.float32)
        p = np.tile(p1[np.newaxis], (m, 1, 1))
        for i, x in enumerate(perm):
            p[i, x, :] = p1[np.arange(n)]
        return p 
               
    def get_matrix(d):
        I = np.eye(d)
        M1 = np.vstack([I]*((d-1)))
        P = np.roll(I,1,axis=-1)
        M2 = P@I
        P_ = P
        for i in range(1,d-1):
            P1 = P_@P
            M2 = np.vstack((M2,P1@I))
            P_ = P1
        return M1,M2

    class GroupInvariance(tf.keras.Model):
        def __init__(self, perm, num_features):
            super(GroupInvariance, self).__init__()
            activation=tf.keras.activations.tanh

            self.num_features = num_features
            self.n = len(perm[0])
            self.m = len(perm)
            self.p = prepare_permutation_matices(perm, self.n, self.m)

            self.features = [
                tf.keras.layers.Dense(16, activation),
                tf.keras.layers.Dense(64, activation),
                tf.keras.layers.Dense(self.n * self.num_features, tf.keras.activations.sigmoid),
                #tf.keras.layers.Dense(self.n * self.num_features, None),
            ]

            self.fc = [
                #tf.keras.layers.Dense(32, tf.keras.activations.tanh),
                tf.keras.layers.Dense(32, tf.keras.activations.relu, use_bias=False),
                tf.keras.layers.Dense(1),
            ]

        def call(self, inputs):
            x = inputs[:, :, tf.newaxis]
            x = apply_layers(x, self.features)
            x = tf.reshape(x, (-1, self.n, self.num_features, self.n))
            x = sigmaPi(x, self.m, self.n, self.p)
            x = apply_layers(x, self.fc)
            return x
        
    np.random.seed(1024)
    ts = 64 
    vs = 480
    d = 10
    divisors = [1, 2, 5, 10]
    k = args.k

    def get_data(x, train_indices):
        P = np.zeros((d, d))
        p_indices = np.roll(train_indices, 1)
        j = 0
        for i in range(d):
            if i in train_indices:
                P[i][p_indices[j]] = 1
                j += 1
            else:
                P[i][i] = 1

        c_train_ds = []
        P1 = P
        for i in range(len(train_indices)-1):
            c_train_ds.append(np.dot(P1, x))
            P1 = P@P1

        p_indices = np.array(list(permutations(train_indices)))
        random_indices = list(np.random.choice(len(p_indices), size=2*k, replace=False))
        new_indices = p_indices[random_indices]

        p_train_ds = []
        for i in range(2*k):
            y = x.copy()
            y[train_indices] = y[new_indices[i]]
            p_train_ds.append(y)

        return np.array(c_train_ds), np.array(p_train_ds)
    
    def get_data_v1(x, train_indices):
        P = np.zeros((d, d))
        p_indices = np.roll(train_indices, 1)
        j = 0
        for i in range(d):
            if i in train_indices:
                P[i][p_indices[j]] = 1
                j += 1
            else:
                P[i][i] = 1
       
        P = P.T
        c_train_ds = [np.dot(x, P)]
        for i in range(len(train_indices)-2):
            c_train_ds.append(np.dot(c_train_ds[-1], P))

        P1 = np.eye(d)
        id0 = train_indices[0]
        id1 = train_indices[1]
        P1[[id0,id1]] = P1[[id1,id0]]
        d_train_ds = [np.dot(x, P1)]
        for i in range(len(train_indices)-1):
            d_train_ds.append(np.dot(d_train_ds[-1], P))      

        p_indices = np.array(list(permutations(train_indices)))
        random_indices = list(np.random.choice(len(p_indices), size=2*k, replace=False))
        new_indices = p_indices[random_indices]

        p_train_ds = []
        for i in range(2*k):
            y = x.copy()
            y[:,train_indices] = y[:,new_indices[i]]
            p_train_ds.append(y)

        return np.concatenate(c_train_ds, axis=0), np.concatenate(d_train_ds, axis=0), np.concatenate(p_train_ds, axis=0)   

    def create_data(d, k, batch_size, true_indices):
        train_ds = np.random.rand(ts*batch_size, d)
        val_ds = np.random.rand(vs*batch_size, d)
        indices = np.array(true_indices).astype(np.int64)

        # Additional data
        c_train_ds, d_train_ds, p_train_ds = get_data_v1(train_ds[0:5], true_indices)
        train_ds = np.vstack((train_ds, c_train_ds, d_train_ds, p_train_ds))
        train_y = poly_Zk_Zn(train_ds, indices).numpy()

        c_val_ds, d_val_ds, p_val_ds = get_data_v1(val_ds[0:5], true_indices)
        val_ds = np.vstack((val_ds, c_val_ds, d_val_ds, p_val_ds))
        val_y = poly_Zk_Zn(val_ds, indices).numpy()
        
        print("Shape info:", [train_ds.shape, train_y.shape, val_ds.shape, val_y.shape])
        return train_ds, train_y, val_ds, val_y

    true_indices = np.sort(np.random.choice(list(range(0, d)), k, replace=False)).tolist()
    train_ds, train_y, val_ds, val_y = create_data(d, k, 1, true_indices)
    print('k:',k)
    print('True Indices:',true_indices)

    def my_regularizer(x):
        #x = tf.abs(x) + 1e-8
        x = x/(tf.reduce_sum(x, axis=0))
        entropy = tf.reduce_mean(tf.reduce_sum(-x*tf.math.log(x), axis=0))
        return 1e-5 * entropy

    lambda_val = 1e-02
    l2_reg = tf.keras.regularizers.l2(1e-5)
    class CyclicGroupInvarianceDiscover(tf.keras.Model):
        def __init__(self, d):
            super(CyclicGroupInvarianceDiscover, self).__init__()
            activation = tf.keras.activations.tanh
            self.d = d

            self.linear1 = Dense(d,activation=None)
            self.inputs1 = Dense(d * (d-1),use_bias=True,activation=None)
            self.inputs2 = Dense(d * (d-1),use_bias=True,activation=None)
            self.linear2 = Dense(d * (d-1),activation=None)

            self.features = [
                tf.keras.layers.Dense(16, activation, kernel_regularizer=l2_reg),
                tf.keras.layers.Dense(64, activation, kernel_regularizer=l2_reg),
                tf.keras.layers.Dense(128, activation, kernel_regularizer=l2_reg),
            ]

            self.Add = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
            self.fc = [
                tf.keras.layers.Dense(64, tf.keras.activations.relu, use_bias=False, kernel_regularizer=l2_reg),
                tf.keras.layers.Dense(64, tf.keras.activations.relu, use_bias=False, kernel_regularizer=l2_reg),
                tf.keras.layers.Dense(32, tf.keras.activations.relu, use_bias=False, kernel_regularizer=l2_reg),
                tf.keras.layers.Dense(32, tf.keras.activations.relu, use_bias=False, kernel_regularizer=l2_reg),
                tf.keras.layers.Dense(1, kernel_regularizer=l2_reg),
            ]
            
        def call(self, inputs):
            inputs = self.linear1(inputs)                   # (B, d)
            in1 = self.inputs1(inputs)                      # (B, d*(d-1))
            in1 = self.linear2(in1)[:, :, tf.newaxis]       # (B, d*(d-1), 1)
            in2 = self.inputs2(inputs)                      # (B, d*(d-1))
            in2 = self.linear2(in2)[:, :, tf.newaxis]       # (B, d*(d-1), 1)

            x = tf.concat((in1,in2), axis=-1)               # (B, d*(d-1), 2)             
            x = apply_layers(x, self.features)
            x = self.Add(x)
            x = apply_layers(x, self.fc)
            return x               

    def new_matrix(k,d,indices,subgroup_indices):
        M3 = np.zeros((d,d))
        for row,index in zip(np.arange(k),indices):
            M3[row,index] = 1     
        if k>1:
            if subgroup_indices == 0:                                
                M4 = np.eye(d*(d-1),d*(d-1))                                # L2, Sk     

            elif subgroup_indices == 1 or 2:                                # Zk, D2k
                M4 = np.zeros((d*(d-1),d*(d-1)))
                M4[0:k-1,0:k-1] = np.eye(k-1)
                key_loc_Zk = ((d-k)*d) + k - 1
                M4[k-1, key_loc_Zk] = 1
                if subgroup_indices == 1:
                    M4[k:(2*k)-1, -d+1:-d+k] = np.eye(k-1)
                    key_loc_D2k = ((k-2)*d)
                    M4[(2*k)-1, key_loc_D2k] = 1
        else:
            M4 =  np.zeros((d*(d-1),d*(d-1)))
            M4[0,0] = 1               

        return M3, M4
            
    filepath = 'saved_model.h5'
    callback = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                  save_best_only=True,
                                                  save_weights_only=True,)

    Model_discover = CyclicGroupInvarianceDiscover(d)
    M1, M2 = get_matrix(d)
    adam = Adam(learning_rate=1e-3)
    sample_output = Model_discover(val_ds)

    Model_discover.layers[0].trainable = False                    # Linear1
    Model_discover.layers[1].trainable = False
    Model_discover.layers[2].trainable = False
    Model_discover.layers[3].trainable = False                    # Linear 2

    Model_discover.compile(optimizer=adam, loss='mae')
    # initial_weights=Model_discover.get_weights()
    # Model_discover.set_weights(initial_weights)

    bias_l1 = np.zeros(d*(d-1))
    bias_l2 = np.zeros(d*(d-1))

    Model_discover.layers[1].set_weights([M1.T, bias_l1])
    Model_discover.layers[2].set_weights([M2.T, bias_l2])

    initial_weights=Model_discover.get_weights()
    Model_discover.set_weights(initial_weights)
    print(Model_discover.summary())     
    
    def generate_matrix(k_,subgroup_indices):
        #k_=int(np.random.choice(divisors,1,replace=False))
        train_indices=np.sort(np.random.choice(list(range(0,d)),k_,replace=False))
        #Generate matrix
        L1, L2 = new_matrix(len(train_indices),d,train_indices,subgroup_indices)
        #print('Matrix:',matrix)
        return L1, L2, train_indices

    def generate_matrix_given_indices(train_indices,subgroup_indices):
        #Generate matrix
        L1, L2 = new_matrix(len(train_indices),d,train_indices,subgroup_indices)
        #print('Matrix:',matrix)
        return L1, L2

    def get_loss(M5,M6):
        Model_discover.set_weights(initial_weights)
        #sample_output=Model_discover(val_ds)
        bias1 = np.zeros(d)
        bias2 = np.zeros(d*(d-1))
        Model_discover.layers[0].set_weights([M5.T, bias1])
        Model_discover.layers[3].set_weights([M6.T, bias2])
        
        start=time.time()
        train_history=Model_discover.fit(train_ds,train_y, 
                            epochs=250,
                            batch_size=16,
                            shuffle=True,
                            validation_data=(val_ds,val_y),
                            callbacks=[callback]) 
        end=time.time()
        print('Time:',end-start)
        
        return train_history

    C = [list(combinations(range(d), i)) for i in range(1, d+1)]
    C = [list(item) for sublist in C for item in sublist]

    At = np.zeros((len(C), d))
    for i, idx in zip(range(len(C)), C):
        At[i][idx] = 1

    # Normalization
    # l2_norms_rows = LA.norm(At, 2, axis=1)[:, np.newaxis]
    # At = At/l2_norms_rows
    At = np.hstack((At, np.ones((At.shape[0], 1))))

    s_arms = np.hstack((At, np.array([[1, 0, 0]] * At.shape[0])))
    d_arms = np.hstack((At, np.array([[0, 1, 0]] * At.shape[0])))
    z_arms = np.hstack((At, np.array([[0, 0, 1]] * At.shape[0])))

    At = np.vstack((s_arms, d_arms, z_arms))
    print('At shape:',At.shape)

    #R = 0.01
    R = 0.1
    epsilon = 0.5
    delta = 0.5
    n_features = d+4
    subgroup = {0:'Sk', 1:'D2k', 2:'Zk'}
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
        train_indices = np.where(contexts_t[:-4] > 0)[0].tolist()
        subgroup_indices = int(np.where(contexts_t[-3:] == 1)[0])
        L1, L2 = generate_matrix_given_indices(train_indices,subgroup_indices)

        #Get reward
        train_history = get_loss(L1,L2)
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
        print('subgroup indices:',subgroup[subgroup_indices])
        print('loss:',mean_squared_error(all_rewards,[h.flatten()[0].tolist() for h in all_y_hat]))
        print('mu_hat:',mu_hat)
        #print([min_loss,train_indices])
        print([min_loss, np.where(mu_hat[:-4] > 0)[0].tolist()])

        np.save(f'c_{k}_10_dec_regression_rewards.npy',np.array(all_rewards))
        np.save(f'c_{k}_10_dec_regression_dec_hat.npy',np.array(all_y_hat))
        np.save(f'c_{k}_10_dec_regression_mu_hat.npy',np.array(all_mu_hat))
        np.save(f'c_{k}_10_dec_regression_train_indices.npy',np.array(all_train_indices))

    l=[]
    for a in contexts:
        l.append((np.round(np.dot(np.reshape(a,(1,-1)),mu_hat)[0][0],4),np.where(a[:-4]>0)[0].tolist())) 

    sort_l = sorted(l,key=lambda x:x[0],reverse=True) 

    top_indices = [b for a,b in sort_l[:10]]  
    if true_indices in top_indices:
        print(True)

    print(sort_l) 

    del Model_discover 
    tf.keras.backend.clear_session()
    gc.collect() 

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--k',type=int)
    args, _ = parser.parse_known_args()
    main(args)
