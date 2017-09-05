# -*- coding: utf-8 -*-
"""
Very basic script for investigating anomaly detection
in the http data_set.
1. Loads data (567498 points)
2. Prints basic stats
3. Trains a VAE on normal data only
4. Saves statistics (ELBO/Reconstruction Losses) on the training set
5. Uses a validation set to select a threshold for classifying anomaly data
6. At test time performs a number of passes and performs inference

v0.0: linear encoder - RNN decoder, learning also the initialization
v1.0: final version of standard VAE that can handle ECG data
v1.1: introduction of RNN decoder
v2.0: Stable version, includes dropout, piphole connections (False)
                      10 z-samples for testing, trained initialization
                      for RNN, clipped gradients - could support BN also
                      %% Dropout slows down training
v2.1: change the data generation method to investigate the VAE vs AE bottleneck
                      example
@author: Heraclitus
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import partial
import matplotlib.gridspec as gridspec
import os
import time
import math
from sklearn.preprocessing import OneHotEncoder
from skimage.data import camera
from skimage.filters import roberts
from skimage import transform as transf
from sklearn.metrics import confusion_matrix
import itertools
import project_helper as ph
import importlib
from functools import partial
from bayes_opt import BayesianOptimization
from scipy.interpolate import spline
importlib.reload(ph)
from tensorflow.examples.tutorials.mnist import input_data

def normalize_seq(data):
    print(data.shape)
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(abs(temp)) - 0.5) * 2
    return out

def augment(data):
    print("Input size:",data.shape)
    s = data.shape[1]
    x_out= np.zeros([1,s])
    data = np.concatenate((data,data),axis=1)
    for i in range(s):
        x_temp = data[:,i:s+i]
        x_out = np.concatenate((x_out,x_temp),axis=0)
    x_out = np.delete(x_out,0,0)
    print("Output size:",x_out.shape)
    return x_out

def add_noise(batch,scale=0.005):
    s = batch.shape
    mask = np.random.normal(0,scale,s)
    return batch+mask

def random_shift(batch,scale=0.0125):
    s = batch.shape[0]
    mask = np.random.normal(0,scale,[s,1])
    return batch+mask

def random_rescale(batch,scale=0.2):
    s = batch.shape[0]
    mask = np.random.uniform(-scale,scale,s).reshape(s,1)
    #print(mask)
    return batch*(np.ones_like(mask)+mask)

def fetch_data(filename='ecg.csv',split_ratio=0.25, real_data=True):
    loaded = np.loadtxt(filename,delimiter=',')
    # load data
    x_tr = loaded[:100,1:]
    y_tr = loaded[:100,0]
    x_test = loaded[100:,1:]
    y_test = loaded[100:,0]
    # find anomalies
    anom_idx = y_tr==-1
    num_tr_anom = sum(y_tr==-1)
    # split to normal and anomalies
    test_idx = y_test==-1
    x_anom_tr = x_tr[anom_idx,:]
    x_norm_tr = x_tr[~anom_idx,:]
    x_anom_test = x_test[test_idx,:]
    x_norm_test = x_test[~test_idx,:]
    # split into training and validation set
    split_n = int(np.ceil(x_test.shape[0]*(split_ratio)))
    #split_a = int(np.ceil(num_tr_anom*(1-split_ratio)))
    x_train_norm = x_norm_tr#x_norm_tr[0:split_n,:]
    x_valid_norm = x_norm_test[split_n:,:]
    x_valid_anom = x_anom_test[0:10]
    x_test_norm = x_norm_test[split_n:,:]
    x_test_anom = x_anom_test[10:,:]
    #x_test_anom = np.concatenate([x_test_anom,x_anom_tr])
    # normalize data
    x_test_norm = normalize_seq(x_test_norm)
    x_train_norm = normalize_seq(x_train_norm)
    x_valid_norm = normalize_seq(x_valid_norm)
    x_test_anom = normalize_seq(x_test_anom)
    x_valid_anom = normalize_seq(x_valid_anom)
    # get sequence length
    s = x_test.shape[1]
    # augment y_test vector
    y_test = np.repeat(y_test.reshape([-1,1]),s,axis=0)  #XXXXXXXXXXX
    # make 0/1
    #y_test = y_test.reshape([-1,1]) #     XXXXXXXXXXXXXXXXXXXX
    y_test = (y_test+1)/2
    # one hot encode
    enc = OneHotEncoder()
    enc.fit(y_test)
    y_test = enc.transform(y_test).toarray()
    # augment data
    x_test_norm = augment(x_test_norm)     #XXXXXXXXXXXXXXXXXXXXXXXXXXX
    x_train_norm = augment(x_train_norm)
    x_valid_norm = augment(x_valid_norm)
    x_test_anom = augment(x_test_anom)     #XXXXXXXXXXXXXXXXXXXXXXXXXXX
    x_valid_anom = augment(x_valid_anom)
    print('x_test:',x_test.shape)
    print('y_test:',y_test.shape)

    return x_train_norm, x_valid_norm, x_test_norm, x_valid_anom, x_test_anom


def get_data(file_name, split_ratio=0.6, real_data=True):
    '''
    Helper function to load the dataset 
    file_name:      the file from which to bring data 
    split_point:    the point at which to perform the data split of the anomalous 
                    data between the validation and the test sets
    return:         x_train,x_valid,x_test,x_valid, anom_valid, anom_test    
    '''
    if real_data:
        mydata = np.load(file_name)
        # Split into training, validation and test sets
        # normal_data
        x_train = ph.standardize(mydata['x_train_normal'])
        x_valid = ph.standardize(mydata['x_valid_normal'])
        x_test = ph.standardize(mydata['x_test_normal'])
        # anomalies
        anom_valid, anom_test = ph.standardize(mydata['anom_valid']),ph.standardize(mydata['anom_test'])
        #x_anomalies = mydata['x_anomalies']

        # # get sizes
        # anomalies_size = x_anomalies.shape[0]
        # # split anomaly data
        # # train/test
        # split = int(anomalies_size * split_ratio)
        # anom_valid = x_anomalies[:split, ]
        # anom_test = x_anomalies[split:, ]
        return x_train, x_valid, x_test, anom_valid, anom_test
    else:
        return make_data()

def make_data(t=10000):
    '''
    Functions that makes fake training data for testing purposes
    Returns: x_train, x_valid, x_test, anom_valid, anom_test
    '''
    t_min, t_max = 0, 10
    resolution = 0.01
    n_steps = 100
    #t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
    def time_series1(t):
        freq = 5 + np.random.normal(0.0,0.05,[t.shape[0],1])
        omega = 2*np.pi * freq
        phase = 2*np.pi*(1+np.random.normal(0.0,1.0,1))
        return np.sin(omega*t)+np.random.normal(0.0,0.5,[t.shape[0],t.shape[1]])#+phase) #+ 2 * np.sin(t * 5)
    def time_series2(t):
        freq = 5 + np.random.normal(0.0,0.05,[t.shape[0],1])
        omega = 2*np.pi * freq
        phase = 2*np.pi*(1+np.random.normal(0.0,1.0,1))
        return np.sin(omega*t)+np.random.normal(0.0,1.5,[t.shape[0],t.shape[1]])#+phase) #+ 2 * np.sin(t * 5)
    def time_series3(t):
        return (np.sin(4.5*t) / 3)  * np.sin(t * 2.1)
    def next_batch(batch_size, n_steps):
        t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
        Ts = t0 + np.arange(0., n_steps) * resolution
        ys_n = time_series1(Ts)
        ys_a = time_series2(Ts)
        return ys_n[:, :].reshape(-1, n_steps), ys_a[:, :].reshape(-1, n_steps)
    x_norm, x_anom = next_batch(t,n_steps)
    a = np.random.permutation(t)
    return x_norm[a[0:int(t*0.6)],:], x_norm[a[int(t*0.6):int(t*0.8)],:], x_norm[a[int(t*0.8):t],:], \
           x_anom[a[0:int(t*0.5)],:], x_anom[a[int(t*0.5):t],:]

def weighted_loss(anomaly_threshold):
    conf_matrix = net(TRAIN_VAE = False, load_model = True, real_data=False,
                      anomaly_threshold=anomaly_threshold, verbose=False, plots=False )
    tp, fp, tn, fn = conf_matrix[0:4]
    w_loss = 80*fn+20*fp
    return w_loss

def threshold_selector(normal_elbo,anomaly_elbo,fp_ratio,fn_ratio,n_pts=10):
    """
    Perform BayesianOptimization to select the threshold value
    """
    lim = -1
    normal_elbo=np.sort(normal_elbo, axis=0)[:lim]
    anomaly_elbo = np.sort(anomaly_elbo, axis=0)[:lim]
    gp_params = {"alpha": 1e-5}
    min_t = min(normal_elbo)[0]
    max_t = np.max(anomaly_elbo)
    median_t = np.median(anomaly_elbo)
    print(median_t)
    eps = 1e-9
    def f1_weighted_cost(threshold,fp_ratio = fp_ratio, fn_ratio=fn_ratio,\
                         normal_elbo = normal_elbo, anomaly_elbo = anomaly_elbo):
        # save accuracy rates
        tn = np.sum(normal_elbo< threshold)
        fp = np.sum(normal_elbo> threshold)
        tp = np.sum(anomaly_elbo > threshold)
        fn = np.sum(anomaly_elbo < threshold)
        w_precision = tp / max((tp + fp_ratio*fp),eps)
        w_recall = tp / max((tp + fn_ratio*fn),eps)
        w_f1 = 2 * w_precision * w_recall / max((w_precision + w_recall),eps)
        return w_f1
    VAE_BO = BayesianOptimization(f1_weighted_cost,
            {'threshold': (min_t, median_t)})
    VAE_BO.explore({'threshold': [min_t, median_t, median_t]})
    VAE_BO.maximize(n_iter=10, **gp_params)
    return VAE_BO.res['max']['max_params']['threshold'], VAE_BO

def elbo_hist( normal_elbo,anomaly_elbo,anomaly_threshold, title, filename, print_dir, dataset, plots=True):
    if plots:
        plt.figure()
        lim = -30
        normal_elbo = np.sort(normal_elbo,axis=0)[:lim]
        anomaly_elbo = np.sort(anomaly_elbo,axis=0)[:lim]
        #sns.distplot(normal_elbo, kde=True, color='blue',  label='Normal')
        plt.hist(normal_elbo, bins=50, histtype='bar', normed=True,color='b', label='Normal')
        print('a_elbo min{}\t,max{}'.format(min(anomaly_elbo),max(anomaly_elbo)))
        plt.hist(anomaly_elbo, bins=50, histtype='bar', normed=True, color='r', alpha=0.5, label='Anomalous')
        #plt.xlim([0,100])
        #sns.distplot(anomaly_elbo, kde=True, color='red', label='Anomalous')
        plt.axvline(x = anomaly_threshold, linewidth=4, color ='g', label='Threshold')
        tit="Dataset: {} - {}".format(dataset,title)+"\nNormalised Histogram"
        plt.title(tit,size= 18)
        plt.xlabel("Evidence Lower Bound, $\mathcal{L}_{ELBO}$",size= 16)
        plt.ylabel("Empirical Density",size= 16)
        plt.axis([0,  max(max(anomaly_elbo),max(normal_elbo))*1.1, 0, 0.5])
        plt.legend(loc='upper left')
        plt.grid(True,which='both')
        if filename != "":
            plt.savefig(print_dir + dataset + filename)
        plt.show()

def plot_bayes_opt(bo_results, ratio, filename, print_dir, dataset, plots=True):
    if plots:
        t_ = bo_results.res['all']['params']
        t = []
        for i in range(len(t_)):
            t.append(t_[i]['threshold'])
        c = bo_results.res['all']['values']
        idx = np.argsort(t)
        x = np.array(t)[idx]
        y = np.array(c)[idx]
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = spline(x, y, x_smooth)
        plt.figure()
        #plt.plot(x_smooth,y_smooth,'-b')
        plt.plot(x, y,'b')
        plt.plot(x,y, 'or', label='Evaluation points')
        plt.axis([0,1.1*max(t),max(0.0,min(c)/2),1.1])
        plt.title("Bayesian Optimisation for Threshold Selection\nDataset: {}, ratio fn/fp={}".format(dataset,ratio),size= 18)
        plt.xlabel("Threshold",size= 16)
        plt.ylabel("Weighted F1 score",size= 16)
        plt.legend(loc='upper left')
        plt.grid(True,which='both')
        if filename != "":
            plt.savefig(print_dir + dataset + filename)
        plt.show()


def net(TRAIN_VAE = True, load_model = True, real_data=True, bayesian_opt = True,
		anomaly_threshold = 10, fp_ratio=1,fn_ratio=10, verbose=True, plots=True,
        batch_norm = False, dropout = True):

    # Set random generator seed for reproducibility
    np.random.seed(168)
    tf.set_random_seed(168)
    # Reset the default graph
    tf.reset_default_graph()

    ###############################################################################
    # ======================= file and directory names ========================== #
    ###############################################################################
    mode = "supervised"
    run_comment = "lin_VAE_rnn_ecg200_test10zsamples"#"lin_VAE_rnn_ecg200_10xRcLoss" #
    dataset = "ecg200"
    save_dir = ("./"+dataset+"/")
    save_sum_dir = save_dir+"logs/"
    save_print_dir = save_dir+"prints/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_sum_dir):
        os.makedirs(save_sum_dir)
    if not os.path.exists(save_print_dir):
        os.makedirs(save_print_dir)

    ###############################################################################
    ##########                    Set model parameters                   ##########
    ###############################################################################

    # RNN parameters
    RNN_size = 128;             n_rnn_layers = 3;
    # hidden and latent space parameters
    z_dim = 10;                h_dim = 512
    # running parameters (hyper-parameters
    epochs  = 56;              batch_size = 48;     early_stopping = 20
    learning_rate = 1e-4;       l2_reg = 1e-3;       drop_probs = 0.25
    val_epochs = 1;             anneal_rate = 1e-0;     n_test_steps = 100
    # sampling parameters & classification criterion
    num_z_samples = 10;          output_samples = 1;        criterion = 'precision';
    # load data directory
    #directory1 = 'LSTM_Data/'
    #data_filename = directory1+'ecg_data3.npz'
    data_filename = 'dataset/' + dataset + "_normal_anomaly_seperate.npz"
    # run parameters
    split_ratio = 0.5
    anom_split = str(np.round(1-split_ratio,decimals=3)).replace(".","_")
    file_name = run_comment+'_'+dataset+"_h_"+str(h_dim)+"_z_"+str(z_dim)
    load_file_name = file_name

    ###############################################################################
    # ========================= get data and variables ========================== #
    ###############################################################################

    # get data
    x_train, x_valid, x_test, anom_valid, anom_test =  fetch_data()
                                                        #get_data(data_filename,split_ratio,real_data=real_data)
    # calculate sizes
    train_size, valid_size, test_size, anom_valid_size, anom_test_size, X_dim =   \
        x_train.shape[0], x_valid.shape[0],x_test.shape[0], anom_valid.shape[0], anom_test.shape[0], x_train.shape[1]
    # other
    num_batches = train_size//batch_size;       epoch = 0;      save_epochs = 0;    b_t_ratio = 2#0train_size//batch_size
    # Training losses containers
    best_eval_loss = np.inf
    training_loss = []
    validation_loss= []
    kl_training_loss = []
    rc_training_loss =[]
    # initiallize regularizer and graph
    regularize = tf.contrib.layers.l2_regularizer(l2_reg, scope=None)
    initialize = tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32)
    graph = tf.Graph()
    # put placeholders on graph
    with graph.as_default():
        # =========================  Placeholders  ==================================
        with tf.name_scope("input"):
            X = tf.placeholder(tf.float32, shape=[None, X_dim],name="input_X")
    #        y = tf.placeholder(tf.float32, shape=[None, y_dim],name="label")
            drop_prob = tf.placeholder(tf.float32, shape=(),name='dropout_prob')
            alpha_KL = tf.placeholder_with_default(input = 1.0,shape=(),name='KL_annealing')
            rc_weight = tf.placeholder_with_default(input = 1.0,shape=(),name='reconstruction_loss_weight')
            is_train = tf.placeholder_with_default(input = False,shape=(),name='train_test_state')
            l_rate = tf.placeholder_with_default(input=learning_rate, shape=(), name='var_learning_rate')
        with tf.name_scope("latent"):
            z = tf.placeholder(tf.float32, shape=[None, z_dim],name="latent_vars")
        # introduce convenience function for batch norm
        batch_norm_layer = partial(tf.layers.batch_normalization, training=is_train, momentum=0.95)
        # =============================== Q(z|X) ====================================
        def encode(x, scope='encoder', reuse=False, drop_prob=drop_probs, is_train=is_train,
                   batch_norm=batch_norm, dropout = dropout):
            '''
            Discriminative model (decoder)
            Input:
                    x : input data
            Returns:
                    z_mu, Z_logvar : mean and standard deviation of z
                '''
            with tf.variable_scope("encoder", reuse = reuse):

                # ======  Qz(x)  ======#
                inputs = x
                h = tf.layers.dense(inputs, h_dim, activation=None,kernel_initializer=initialize,
                                    kernel_regularizer=regularize,name='e_hidden_1')
                if dropout:
                    h = tf.layers.dropout(h, training=is_train, rate=drop_probs, seed=128)
                if batch_norm:
                    h = batch_norm_layer(h)
                h = tf.nn.elu(h)
                z_mu = tf.layers.dense(h, z_dim, activation=None,kernel_initializer=initialize,
                                    kernel_regularizer=regularize,name='z_mu')
                if batch_norm:
                    z_mu = batch_norm_layer(z_mu)
                z_logvar = tf.layers.dense(h, z_dim, activation=None,kernel_initializer=initialize,
                                    kernel_regularizer=regularize,name='z_logvar')
                if batch_norm:
                    z_logvar = batch_norm_layer(z_logvar)
            return z_mu, z_logvar

        def sample_z(mu, log_var):
            eps = tf.random_normal(shape=tf.shape(mu))
            return mu + tf.exp(log_var / 2) * eps

        # =============================== P(X|z) ====================================
        def decode(z, scope = 'decoder', reuse=False, drop_prob=drop_probs, is_train=is_train,
                   batch_norm=batch_norm, dropout = dropout):
            '''
            Generative model (decoder)
            Input:      z : latent space data
            Returns:    x : generated data
            '''
            with tf.variable_scope("decoder", reuse=reuse):

                #======  Px(z)  ======#
                inputs = z # tf.concat(axis=1, values=[z])
                # calculate hidden
                h = tf.layers.dense(inputs, 2*n_rnn_layers*RNN_size, activation=None,kernel_initializer=initialize,
                                    kernel_regularizer=regularize,name='RNN_state_init_layer')
                if dropout:
                    h = tf.layers.dropout(h, training=is_train, rate=drop_probs, seed=128)
                if batch_norm:
                    h = batch_norm_layer(h)
                h = tf.nn.elu(h)
                #h = tf.unstack(h, axis=0)
                h = tf.reshape(h,[n_rnn_layers,2,-1,RNN_size])
                init_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(h[idx][0], h[idx][1])
                                    for idx in range(n_rnn_layers)])
                memory_cell = [tf.contrib.rnn.LSTMCell(RNN_size, use_peepholes= False, forget_bias=1.0, state_is_tuple=True)
                               for cell in range(n_rnn_layers)]
                memory_cell = tf.contrib.rnn.MultiRNNCell(memory_cell)
                #memory_cell = tf.layers.dropout(memory_cell, rate=drop_prob, training = is_train, seed=128)
                inputs = tf.expand_dims(inputs,-1)
                #   [10,50,1]
                rnn_outputs, states = tf.nn.dynamic_rnn(memory_cell, inputs=inputs,#dtype=tf.float32)
                                                       initial_state=init_state,dtype=tf.float32)
                #   [10,50,128]
                stacked_outputs = tf.reshape(rnn_outputs,[-1,RNN_size*z_dim])
                if batch_norm:
                    stacked_outputs = batch_norm_layer(stacked_outputs)
                #   [10,50*128]
                # calculate the mean of the output (Gausian)
                x_mu = tf.layers.dense(stacked_outputs, X_dim, activation=None,kernel_initializer=initialize,
                                    kernel_regularizer=regularize,name='x_mu')
                if batch_norm:
                    x_mu = batch_norm_layer(x_mu)
                #   [10,100]
                #x_mu = tf.reshape(x_mu,[-1,X_dim])
                #   [500,100]    v.s. [10 100]
                # x_logvar = tf.layers.dense(stacked_outputs, X_dim, activation=None,kernel_initializer=initialize,
                #                     kernel_regularizer=regularize,name='x_logvar')
                # if batch_norm:
                #     x_logvar = batch_norm_layer(x_logvar)
                #x_logvar = tf.reshape(x_logvar,[-1,X_dim])
                #assert ph.shape(x_logvar)==(50,100)
                #print(ph.shape(x_logvar))
            return x_mu #, x_logvar

        # =============================== ELBO ====================================
        def loss(X,x_sample,z_mu,z_logvar,reuse=None, n_z_samples=1,alpha_KL=alpha_KL,rc_weight=rc_weight):
            with tf.name_scope("loss"):
                # E[log P(X|z)]
                # print(ph.shape(x_sample))
                # print(ph.shape(X))
                recon_loss = 0.5 * tf.reduce_sum(tf.square(x_sample-X),axis=1) / n_z_samples
                #tf.cond(is_train,False):
                for i in range(n_z_samples-1):
                    z_sample = sample_z(z_mu, z_logvar)
                    #x_mu, x_logvar  = decode(z_sample,reuse=reuse)
                    x_mu = decode(z_sample, reuse=reuse)
                    # x_sample = sample_z(x_mu, x_logvar)
                    x_sample = x_mu
                    recon_loss += 0.5 * tf.reduce_sum(tf.square(X-x_sample),axis=1) / n_z_samples
                # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
                kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, axis=1)
                # print(tf.shape(kl_loss))
                # print(tf.shape(recon_loss))
                # Regularisation cost
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                # VAE loss
                ELBO = rc_weight*recon_loss + alpha_KL *kl_loss
                vae_loss = tf.add_n([tf.reduce_mean(ELBO)+ tf.reduce_sum(reg_variables)])
                #batch_loss = tf.reduce_mean(ELBO)
                # summary
            with tf.name_scope("Summaries"):
                #tf.summary.scalar("ELBO",ELBO)
                tf.summary.scalar("Batch_loss",tf.reduce_mean(ELBO))
                merger = tf.summary.merge_all()
            return vae_loss,ELBO,merger, tf.reduce_mean(kl_loss), tf.reduce_mean(recon_loss)

        # =============================== TRAINING ====================================
        # embed (encode)
        z_mu, z_logvar = encode(X)
        with tf.name_scope("latent"):
            z_sample = sample_z(z_mu, z_logvar)
        # generate (decode)
        # x_mu, x_logvar = decode(z_sample, reuse=None)
        x_mu = decode(z_sample, reuse=None)
        # sample x
        #x_sample = sample_z(x_mu,x_logvar)
        x_sample = x_mu
        # loss
        vae_loss, ELBO, merger,kl_loss,rec_loss= loss(X,x_sample,z_mu,z_logvar,reuse=True, n_z_samples=10,
                                                      alpha_KL=alpha_KL,rc_weight=rc_weight)
        with tf.name_scope("optimiser"):
            # updater
            train_step = tf.train.AdamOptimizer(learning_rate=l_rate)
            grads = train_step.compute_gradients(vae_loss)
            clipped_grads = [(tf.clip_by_value(grad, -2,2),var) for grad,var in grads]
            solver = train_step.apply_gradients(clipped_grads)
            bn_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # =============================== TESTING =================================== #
        with tf.name_scope("Testing"):
            z_mu_t, z_logvar_t = encode(X,reuse=True)
            z_sample_t = sample_z(z_mu_t, z_logvar_t)
            # x_mu_t, x_logvar_t = decode(z_sample_t, reuse=True)
            x_mu_t = decode(z_sample_t, reuse=True)
            # sample x
            #x_sample_t = sample_z(x_mu_t, x_logvar_t)
            x_sample_t = x_mu_t
            # loss
            _,ELBO_t,_,_,_ = loss(X,x_sample_t,z_mu_t,z_logvar_t,reuse=True,\
													n_z_samples=num_z_samples,alpha_KL=1.0)

        # =============================== Session =============================== #
        if TRAIN_VAE:
            sum_writter = tf.summary.FileWriter(save_sum_dir,graph)
            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer(),{is_train: False})
            if load_model:
                print("Loading saved model")
                if save_dir is None:
                        raise ValueError("Filename and path not supplied! Aborting...")
                else:
                    try:
                        tf.train.Saver().restore(sess,save_dir+load_file_name)
                        print("Done! Continuing training...\n")
                        loaded = np.load(save_dir+file_name+'.npz')
                        best_eval_loss = loaded['best_eval_loss']
                    except Exception:
                        print("Could not find saved file. Training from scratch")

            # ========================= START TRAINING ========================== #
            # normalize data
            train_mean, train_std, x_train = ph.normalize(x_train)
            x_valid = ph.p_normalize(x_valid,train_mean, train_std)
            print("Ttraining set length:{}".format(x_train.shape[0]))
            print(" -----Training Started-----")
            count = 0
            for epoch in range(epochs):
                epoch_loss = 0
                r_loss = 0
                kl = 0
                epoch_time = time.time()
                x_train = ph.shuffle_x_data(x_train)
                # For each epoch train with mini-batches of size (batch_size)
                for batch_num in range(train_size//batch_size):
                    # anneal KL cost (full cost after 50.000 batches)
                    kl_a = 2*(-.5+1/(1+np.exp(-count*anneal_rate)))
                    X_mb = x_train[batch_num*batch_size:(batch_num+1)*batch_size,]
                    X_mb = random_rescale(X_mb)
                    X_mb = add_noise(X_mb)
                    #X_mb = random_shift(X_mb)
                    # train
                    train_dict = {X: X_mb, is_train: True, drop_prob: drop_probs, alpha_KL: kl_a, rc_weight: b_t_ratio,
                                  l_rate: learning_rate}
                    _, loss,k_,r_ = sess.run([solver, vae_loss,kl_loss,rec_loss], feed_dict= train_dict) # DELETED ,bn_update
                    epoch_loss+=loss;  	kl+= k_;	r_loss+= r_; count+=1
                # print progress
                ph.progress(epoch,(epoch_loss/num_batches),(r_loss/num_batches),(kl/num_batches),\
                            time.time()-epoch_time)
                training_loss.append(epoch_loss/num_batches)
                rc_training_loss.append(r_loss / num_batches)
                kl_training_loss.append(kl / num_batches)
                # validate
                if epoch >0 and epoch%val_epochs ==0:
                    vloss = 0
                    valid_dict={X: x_valid,is_train:False,drop_prob:0.0, alpha_KL : 1.0}
                    vloss, vaeloss = sess.run([vae_loss,merger], feed_dict=valid_dict )
                    sum_writter.add_summary(vaeloss, epoch)
                    # print progress
                    print('Validation_Training_Epoch: {}'.format(epoch))
                    print('Loss: {}'. format(vloss))
                    validation_loss.append(vloss)
                    if vloss < best_eval_loss:
                        # update best result and save checkpoint
                        best_eval_loss = vloss
                        saver.save(sess, save_dir+ file_name)
                        save_epochs = epoch
                # early stopping condition
                if epoch - save_epochs > early_stopping//2:
                    learning_rate/=2
                if epoch - save_epochs >early_stopping :
                    print("Early stopping condition reached. No progress for {} epochs".format(early_stopping))
                    break
            # write summary to npz file
            description = "Dataset: "+dataset+", model: "+run_comment+", h: "+str(h_dim)\
                        + ", z: "+str(z_dim)+", learning_rate: "+str(learning_rate)+ ", L2: "+str(l2_reg)\
                        + ", batch_size: " + str(batch_size)+", split: "+str(split_ratio)\
                        + ", epochs"+str(save_epochs)
            np.savez(save_dir+file_name,\
                     training_loss=training_loss,validation_loss=validation_loss,\
                     best_eval_loss = best_eval_loss, description=description)
            ph.save2txt(save_print_dir,file_name, dataset,  run_comment, h_dim, z_dim, num_z_samples, learning_rate, batch_size, drop_probs,
                        l2_reg,save_epochs, early_stopping,X_dim,train_size,valid_size,test_size,0,anom_valid_size,anom_test_size,save_dir)
            # print training curves
            plt.figure()
            tl=np.array(rc_training_loss) + np.array(kl_training_loss)
            plt.plot(tl, 'b', label='training loss')
            plt.plot(rc_training_loss, 'm', label='reconstruction loss')
            plt.plot(validation_loss, 'r', label='validation loss')
            plt.plot(kl_training_loss, 'g', label='KL loss')
            plt.title('Training Curves\nDataset:{}, Method:{}'.format(dataset,mode))
            plt.xlabel('Training epoch')
            plt.ylabel('Loss')
            plt.legend(loc="upper right")
            plt.show()
            plt.figure()
            plt.plot(tl, 'b', label='training loss')
            plt.plot(validation_loss, 'r', label='validation loss')
            plt.title('Training Curves\nDataset:{}, Method:{}'.format(dataset,mode))
            plt.xlabel('Training epoch')
            plt.ylabel('Loss')
            plt.legend(loc="upper right")
            plt.show()
        else:
            # load saved model
            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            print("Loading saved model")
            if save_dir is None:
                    raise ValueError("Filename and path not supplied! Aborting...")
            else:
                try:
                    saver.restore(sess,save_dir+load_file_name)
                    print("Done! \n")
                except Exception:
                    print("Could not find saved file.")
        # normalize data
        if not TRAIN_VAE:
            train_mean, train_std, x_train = ph.normalize(x_train)
            x_valid = ph.p_normalize(x_valid,train_mean, train_std)
            anom_valid = ph.p_normalize(anom_valid,train_mean,train_std)
        # break the validation set evaluation into 'n_test_steps' steps to avoid memory overflow
        normal_valid_size = x_valid.shape[0]
        normal_elbo = np.zeros([normal_valid_size, 1])
        anomaly_elbo = np.zeros([anom_valid_size, 1])
        # evaluate ELBO on the normal validation-set
        for j in range(n_test_steps - 1):
            start = j * (normal_valid_size // n_test_steps);
            stop = (j + 1) * (normal_valid_size // n_test_steps)
            normal_valid_dict = {X: x_valid[start:stop], is_train: False, drop_prob: 0.0}
            x_elbo_v = sess.run([ELBO_t], feed_dict=normal_valid_dict)
            normal_elbo[start:stop, 0] = x_elbo_v[0]
        # compute the last slice separately since it might have more points
        normal_valid_dict = {X: x_valid[stop:], is_train: False, drop_prob: 0.0}
        x_elbo_v = sess.run([ELBO_t], feed_dict=normal_valid_dict)
        normal_elbo[stop:, 0] = x_elbo_v[0]
        normal_elbo = np.clip(normal_elbo, None, 1e4)
        # evaluate ELBO on the anomaly valildation-set
        for j in range(n_test_steps - 1):
            start = j * (anom_valid_size // n_test_steps);
            stop = (j + 1) * (anom_valid_size // n_test_steps)
            anomalous_valid_dict = {X: anom_valid.reshape([-1, X_dim])[start:stop], is_train: False, drop_prob: 0.0}
            a_elbo_v = sess.run([ELBO_t], feed_dict=anomalous_valid_dict)
            anomaly_elbo[start:stop, 0] = a_elbo_v[0]
        # compute the last slice separately since it might have more points
        anomalous_valid_dict = {X: anom_valid.reshape([-1, X_dim])[stop:], is_train: False, drop_prob: 0.0}
        a_elbo_v = sess.run([ELBO_t], feed_dict=anomalous_valid_dict)
        anomaly_elbo[stop:, 0] = a_elbo_v[0]
        anomaly_elbo = np.clip(anomaly_elbo, None, 1e4)

        ph.plot_hist(normal_elbo, title='ELBO distribution\n validation set (normal data)', print_dir= save_print_dir,
                  f_name="valid_normal", figsize=(8,5), dataset=dataset, plots=plots)
        ph.plot_hist(anomaly_elbo,title='ELBO distribution\n validation set (anomaly data)', print_dir= save_print_dir,\
                  f_name="valid_anomaly", figsize=(8,5), dataset=dataset, plots=plots)
        # print stats
        ph.report_stats("\nValidation Statistics - Normal Data\n",normal_elbo, verbose=verbose)
        ph.report_stats("\nValidation Statistics - Anomaly Data\n",anomaly_elbo, verbose=verbose)

        ###########################################################################################
        ######                          THRESHOLD SELECTION                                 #######
        ###########################################################################################
        x_mean_elbo_valid = np.mean(normal_elbo)
        a_mean_elbo_valid = np.mean(anomaly_elbo)
        # set threshold value
        valid_elbo_dict = {'n_mean':np.mean(normal_elbo), 'n_std':np.std(normal_elbo),
                     'a_mean':np.mean(anomaly_elbo), 'a_std':np.std(anomaly_elbo) }
        if a_mean_elbo_valid > x_mean_elbo_valid:
            #anomaly_threshold = 25
            anomaly_threshold = ph.select_threshold(valid_elbo_dict, criterion)
        else:
            print('Training error! Anomaly loss smaller than normal loss! Aborting...')
            anomaly_threshold = ph.select_threshold(valid_elbo_dict, criterion)
        # If Bayesian optimisation is selected then send the data to the scoring function and call
        # the bayesian opt routine -> and collect the results
        if bayesian_opt:
            # anomaly_threshold = 150
            anomaly_threshold, bo_results = threshold_selector(normal_elbo, anomaly_elbo, \
                                                               fp_ratio=fp_ratio, fn_ratio=fn_ratio)
            plot_bayes_opt(bo_results, fn_ratio, 'figure 3', save_print_dir, dataset, plots=plots)
        elbo_hist(normal_elbo, anomaly_elbo, anomaly_threshold, 'Minas_Validation Set', \
                      'figure 4', save_print_dir, dataset, plots=plots)

        #=================== Evaluation on Test Set ==========================#
        # normalize data
        x_test = ph.p_normalize(x_test,train_mean,train_std)
        anom_test = ph.p_normalize(anom_test,train_mean,train_std) #SSSSSOOOOOSSSSS
        print(x_test.shape)
        print(anom_test.shape)
        if verbose:
            print("#=========================Test Set=============================#")
        print('Anomaly threshold: {}', anomaly_threshold)
        start_time = time.time()
        t_normal_elbo = np.zeros([test_size, 1])
        t_anomaly_elbo = np.zeros([anom_test_size, 1])
        x_samples_normal = np.zeros([test_size,X_dim])
        x_samples_anomaly = np.zeros([anom_test_size,X_dim])
        # evaluate ELBO on the normal validation set
        for j in range(n_test_steps - 1):
            start = j * (test_size // n_test_steps);
            stop = (j + 1) * (test_size // n_test_steps)
            normal_test_dict = {X: x_test[start:stop], is_train: False, drop_prob: 0.0}
            x_elbo_t = sess.run([ELBO_t], feed_dict=normal_test_dict)
            t_normal_elbo[start:stop, 0] = x_elbo_t[0]
            x_samples_normal[start:stop, :] = sess.run([x_sample_t], feed_dict=normal_test_dict)[0]
        # compute the last slice separately since it might have more points
        normal_test_dict = {X: x_test[stop:], is_train: False, drop_prob: 0.0} #.reshape([-1,X_dim])
        x_elbo_t = sess.run([ELBO_t], feed_dict=normal_test_dict)
        t_normal_elbo[stop:, 0] = x_elbo_t[0]
        t_normal_elbo = np.clip( t_normal_elbo,None,1e4)
        x_samples_normal[stop:, :] = sess.run([x_sample_t], feed_dict=normal_test_dict)[0]
        start = stop = 0
        # evaluate ELBO on the anomaly test-set
        for j in range(n_test_steps - 1):
            start = j * (anom_test_size // n_test_steps);
            stop = (j + 1) * (anom_test_size // n_test_steps)
            anomalous_test_dict = {X: anom_test[start:stop].reshape([-1,X_dim]), is_train: False, drop_prob: 0.0}
            a_elbo_t = sess.run([ELBO_t], feed_dict=anomalous_test_dict)
            t_anomaly_elbo[start:stop, 0] = a_elbo_t[0]
            x_samples_anomaly[start:stop, :] = sess.run([x_sample_t], feed_dict=anomalous_test_dict)[0]
        # compute the last slice separately since it might have more points
        anomalous_test_dict = {X: anom_test[stop:].reshape([-1,X_dim]), is_train: False, drop_prob: 0.0}
        a_elbo_t = sess.run([ELBO_t], feed_dict=anomalous_test_dict)
        t_anomaly_elbo[stop:, 0] = a_elbo_t[0]
        t_anomaly_elbo = np.clip(t_anomaly_elbo,None,1e4)
        x_samples_anomaly[stop:, :] = sess.run([x_sample_t], feed_dict=anomalous_test_dict)[0]
        # save accuracy rates
        tn = np.sum(t_normal_elbo < anomaly_threshold)
        fp = np.sum(t_normal_elbo > anomaly_threshold)
        tp = np.sum(t_anomaly_elbo > anomaly_threshold)
        fn = np.sum(t_anomaly_elbo < anomaly_threshold)
        y_pred_n = t_normal_elbo > anomaly_threshold
        y_pred_a = t_anomaly_elbo > anomaly_threshold
        end_time = time.time()
        y_pred = np.concatenate((y_pred_n, y_pred_a), axis=0)
        y_true = np.concatenate((np.zeros([test_size]), np.ones([anom_test_size])), axis=0)
        # calculate the AUC-score
        t_elbo = np.concatenate((t_normal_elbo, t_anomaly_elbo), axis=0)
        # calculate and report total stats
        # scores
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        f01 = (1 + (1 / fn_ratio) ** 2) * precision * recall / ((1 / fn_ratio ** 2) * precision + recall)
        auc, thresh = ph.auc_plot(t_elbo, y_true, anomaly_threshold, f01)
        print("AUC:", auc)
        ph.report_stats("\nTest Statistics - Normal Data", t_normal_elbo, verbose=verbose)
        ph.report_stats("\nTest Statistics - Anomaly Data", t_anomaly_elbo, verbose=verbose)
        ph.scores_x(tp,fp,fn,tn, verbose=verbose)

        elbo_hist(t_normal_elbo, t_anomaly_elbo, anomaly_threshold, 'Test Set','Minas_figure 5',
                  save_print_dir, dataset, plots=plots)

        ph.plot_hist(t_normal_elbo, title='ELBO distribution\n test set (normal data)', print_dir=save_print_dir, \
                     f_name="test_normal", figsize=(8, 5), dataset=dataset, plots=plots)
        ph.plot_hist(t_anomaly_elbo, title='ELBO distribution\n test set (anomaly data)', print_dir=save_print_dir, \
                     f_name="test_anomaly", figsize=(8, 5), dataset=dataset, plots=plots)
        # Compute confusion matrix
        cnf_matrix = np.array([[int(tp),int(fn)],
                               [int(fp),int(tn)]])
        np.set_printoptions(precision=2)
        class_names = np.array(['Anomaly','Normal'],dtype='<U10')
        # Plot non-normalized confusion matrix
        if plots:

            ph.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                    title=" Confusion matrix\n"+"Dataset: "+dataset+" - "+mode, plots=plots)
            plt.savefig(save_print_dir + dataset + "_threshold_" + str(round(anomaly_threshold, 2)) + '_conf_mat.png')
            plt.show()
        if verbose:
            print('total inference time for {} data points:{:6.3}s '.format(y_true.shape[0],\
                                                                    (end_time-start_time)))
    test_elbo_dict = {'n_mean':t_normal_elbo.mean(), 'n_std':t_normal_elbo.std(),
                       'a_mean':t_anomaly_elbo.mean(), 'a_std':t_anomaly_elbo.std()}
    # save all elbo results to a file for post-processing
    np.savez(save_dir + file_name + 'res', descr=run_comment,
             val_norm_elbo=normal_elbo,val_anom_elbo=anomaly_elbo,x_val=x_valid,
             tst_norm_elbo=t_normal_elbo,tst_anom_elbo=t_anomaly_elbo,x_tst=x_test)
    ph.saveres2txt(save_print_dir, file_name, dataset,round(anomaly_threshold,2),
                   tp,fp,tn,fn,f1, auc, f01,precision,recall,valid_elbo_dict,test_elbo_dict,learning_rate)
    # return statements

    return [tp, fp, tn, fn],x_samples_normal,x_samples_anomaly, x_test, anom_test, \
           normal_elbo,anomaly_elbo,t_normal_elbo,t_anomaly_elbo


if __name__ == "__main__":
    conf_matrix,xn,xa,x_test,anom_test,normal_elbo_valid, anomaly_elbo_valid,normal_elbo_test,anomaly_elbo_test = \
        net(TRAIN_VAE =False, load_model = True, real_data= True, bayesian_opt=False, anomaly_threshold=150,
            fp_ratio=1,fn_ratio=1,verbose=True, plots=True,batch_norm=False, dropout=True)
    # one can use xn (prediction for x_test_normal) and xa (prediction for x_test_anomalous) for visualisation

    bayesian_opt=False; dataset = 'ECG200'
    fp_ratio = 1; fn_ratio = 10

    fig, ax = plt.subplots(10)#xn.shape[0])
    ax[0].set_title('Normal_Predictions')
    for i in range(10):
        ax[i].plot(xn[i, :], label=str(i))
    plt.show()
    ax[0].set_title('Normal_True')
    for i in range(10):
        ax[i].plot(x_test[i, :], label=str(i))
    plt.legend(('Predicted', 'True'), loc='best')
    plt.show()
    fig, ax = plt.subplots(10)#xn.shape[0])
    ax[0].set_title('Anomaly__Predictions')
    for i in range(10):
        ax[i].plot(xa[i, :], label=str(i))
    plt.show()
    ax[0].set_title('Anomaly__True')
    for i in range(10):
        ax[i].plot(anom_test[i, :], label=str(i))
    plt.legend(('Predicted', 'True'), loc='best')
    plt.show()
    anomaly_threshold = 65
    if bayesian_opt:
        # anomaly_threshold = 150
        anomaly_threshold, bo_results = threshold_selector(normal_elbo_valid, anomaly_elbo_valid, \
                                                           fp_ratio=fp_ratio, fn_ratio=fn_ratio,n_pts=15)
        plot_bayes_opt(bo_results, fn_ratio, 'figure 3', '', dataset, plots=True)
    elbo_hist(normal_elbo_valid, anomaly_elbo_valid, anomaly_threshold, 'Validation Set', \
              'figure 4', '', dataset, plots=True)
    elbo_hist(normal_elbo_test, anomaly_elbo_test, anomaly_threshold, 'Test Set', \
              'figure 5', '', dataset, plots=True)
    tp, fp, tn, fn, precision, recall, f1, f01, auc, y_pred, y_true, cnf = ph.score_stats(normal_elbo_test,
                                                                                       anomaly_elbo_test, 22,
                                                                                       fn_ratio=10)


    def post_process_ecg():


        return

