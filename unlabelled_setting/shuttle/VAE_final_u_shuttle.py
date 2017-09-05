# -*- coding: utf-8 -*-
"""
Very basic script for investigating anomaly detection
in the http data_set.
1. Loads data (567498 points)
2. Prints basic stats
3. Trains a VAE on mixed data
4. Saves statistics (ELBO/Reconstruction Losses) on the training set
5. Uses a validation set to select a threshold for classifying anomaly data
6. At test time performs a number of passes and performs inference

v0.0: linear encoder - RNN decoder, learning also the initialization
@author: Heraclitus
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import math
from sklearn.metrics import confusion_matrix
import itertools
import project_helper as ph
import importlib
importlib.reload(ph)
from bayes_opt import BayesianOptimization
from functools import partial
from scipy.interpolate import spline
from tensorflow.examples.tutorials.mnist import input_data

def get_data(file_name, split_ratio=0.8, real_data=True):
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
        x_train = mydata['x_train']
        x_valid = mydata['x_valid']
        x_test = mydata['x_test']
        # get labels
        y_train = mydata['y_train']
        y_valid = mydata['y_valid']
        y_test = mydata['y_test']
        # train/test
        # split = int(anomalies_size//split_ratio)
        ##  y = [1 0] -> normal
        ##  y = [0 1] -> anomaly
        # get sizes
        train_size = x_train.shape[0]
        validation_size = x_valid.shape[0]
        test_size = x_test.shape[0]
        print("Training size:",train_size,"validation size:")
        # data dimensionality
        X_dim = x_train.shape[1]
        y_dim = y_train.shape[1]
        return x_train, x_valid, x_test, y_train, y_valid, y_test
    else:
        return make_data()

def make_data(t=1000):
    '''
    Functions that makes fake training data for testing purposes
    Returns: x_train, x_valid, x_test, anom_valid, anom_test
    '''
    t_min, t_max = 0, 100
    resolution = 0.1
    n_steps = 50
    t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
    def time_series1(t):
        return np.sin(2*t) / 3 #+ 2 * np.sin(t * 5)
    def time_series2(t):
        return (np.sin(4.5*t) / 3) *2 * np.sin(t * 2.1)
    def next_batch(batch_size, n_steps):
        t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
        Ts = t0 + np.arange(0., n_steps + 1) * resolution
        ys_n = time_series1(Ts)
        ys_a = time_series2(Ts)
        return ys_n[:, :].reshape(-1, n_steps), ys_a[:, :].reshape(-1, n_steps)
    x_norm, x_anom = next_batch(100,50)
    a = np.random.permutation(100)
    return x_norm[a[0:60],:], x_norm[a[60:80],:], x_norm[a[80:100],:], x_anom[a[0:50],:], x_anom[a[50:100],:]

def weighted_loss(anomaly_threshold):
    conf_matrix = net(TRAIN_VAE = False, load_model = True, real_data=False, anomaly_threshold=anomaly_threshold)
    tp, fp, tn, fn = conf_matrix[0:4]
    w_loss = 80*fn+20*fp
    return w_loss

def threshold_selector(normal_elbo,anomaly_elbo,fp_ratio,fn_ratio):
    """
    Perform BayesianOptimization to select the threshold value
    """
    gp_params = {"alpha": 1e-5}
    min_t = min(normal_elbo.min(axis=1))
    max_t = np.mean(anomaly_elbo.mean(axis=1))
    eps = 1e-9
    def f1_weighted_cost(threshold,fp_ratio = fp_ratio, fn_ratio=fn_ratio,\
                         normal_elbo = normal_elbo, anomaly_elbo = anomaly_elbo):
        nsamples = normal_elbo.shape[1]
        tn,fn,fp,tp = [np.zeros([nsamples]) for i in range(4)]
        for i in range(nsamples):
            # save accuracy rates
            tn[i] = np.sum(normal_elbo[:,i]< threshold)
            fp[i] = np.sum(normal_elbo[:,i]> threshold)
            tp[i] = np.sum(anomaly_elbo[:,i] > threshold)
            fn[i] = np.sum(anomaly_elbo[:,i] < threshold)
        w_precision = tp.mean() / max((tp.mean() + fp_ratio*fp.mean()),eps)
        w_recall = tp.mean() / max((tp.mean() + fn_ratio*fn.mean()),eps)
        w_f1 = 2 * w_precision * w_recall / max((w_precision + w_recall),eps)
        return w_f1
    VAE_BO = BayesianOptimization(f1_weighted_cost,
            {'threshold': (min_t, max_t)})
    VAE_BO.explore({'threshold': [min_t, (min_t+max_t)/2, max_t]})
    VAE_BO.maximize(n_iter=10, **gp_params)
    return VAE_BO.res['max']['max_params']['threshold'], VAE_BO

def unsupervised_threshold(elbos, anom_rate):
    """
    Helper function for calculating the anomaly threshold for the unsupervised setting
    input:      calculated elbos, anom_rate
    output:     elbo_threshold
    """
    sorted_elbo = np.sort(elbos,axis=0,kind='mergesort')
    idx = int(np.ceil(elbos.shape[0] * anom_rate))
    print('threshold:{}, rate:{}, index:{}'.format(sorted_elbo[-idx],anom_rate,-idx))
    return sorted_elbo[-idx]

def elbo_hist( normal_elbo,anomaly_elbo,anomaly_threshold, title, filename, print_dir, dataset, plots=True):
    if plots:
        plt.figure()
        #sns.distplot(normal_elbo, kde=True, color='blue',  label='Normal')
        plt.hist(normal_elbo, bins=50, histtype='bar', normed=True,color='b', label='Normal')
        plt.hist(anomaly_elbo, bins=50, histtype='bar', normed=True, color='r', alpha=0.5, label='Anomalous')
        #sns.distplot(anomaly_elbo, kde=True, color='red', label='Anomalous')
        plt.axvline(x = anomaly_threshold, linewidth=4, color ='g', label='Threshold')
        tit="Dataset: {} - {}".format(dataset,title)+"\nNormalised Histogram"
        plt.title(tit)
        plt.xlabel("Evidence Lower Bound, $\mathcal{L}_{ELBO}$")
        plt.ylabel("Empirical Density")
        plt.axis([0, max(anomaly_elbo)*1.1, 0, 0.8])
        plt.legend(loc='upper left')
        plt.grid(True,which='both')
        if filename != "":
            plt.savefig(print_dir + dataset + filename)
        plt.show()

def plot_bayes_opt(bo_results, title, filename, print_dir, dataset, plots=True):
    if plots:
        t_ = bo_results.res['all']['params']
        t = []
        for i in range(len(t_)):
            t.append(t_[i]['threshold'])
        c = bo_results.res['all']['values']
        idx = np.argsort(t)
        x = np.array(t)[idx]
        y = np.array(c)[idx]
        # x_smooth = np.linspace(x.min(), x.max(), 100)
        # y_smooth = spline(x, y, x_smooth)
        plt.figure()
        # plt.plot(x_smooth,y_smooth,'-b')
        plt.plot(x, y,'b')
        plt.plot(x,y, 'or', label='Evaluation points')
        plt.axis([0,1.1*max(t),max(0.0,min(c)/2),1.1])
        plt.title("Bayesian Optimisation for Threshold Selection\nDataset: {}, ratio fn/fp={}".format(dataset,title))
        plt.xlabel("Threshold")
        plt.ylabel("Weighted F1 score")
        plt.legend(loc='upper left')
        plt.grid(True,which='both')
        if filename != "":
            plt.savefig(print_dir + dataset + filename)
        plt.show()
        if filename != "":
            plt.savefig(print_dir + dataset + filename)
        plt.show()

def net(TRAIN_VAE = True, load_model = True, real_data=True, bayesian_opt = True, 
		anomaly_threshold = 10, fp_ratio=1,fn_ratio=10, verbose=True, plots=True,
        batch_norm = False, dropout = False, anom_rate =0.01):

    tf.reset_default_graph()

    ###############################################################################
    # ======================= file and directory names ========================== #
    ###############################################################################
    mode = "unsupervised_"
    run_comment = mode+"_linVAElin_2"
    dataset = "Shuttle"
    save_dir = ("./"+mode+dataset+"/")
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
    RNN_size = 128;             n_rnn_layers = 1;
    input_keep_prob = 1.0;      output_keep_prob = 1.0;
    # hidden and latent space parameters
    z_dim = 10;                h_dim = 1024
    # running parameters (hyper-parameters
    epochs  = 101;              batch_size = 100;       early_stopping = 30
    learning_rate = 1e-5;       l2_reg = 1e-2;          drop_probs = 0.15
    valid_every = 1;            anneal_rate = 5e-1;     n_test_steps = 100
    # sampling parameters & classification criterion
    num_z_samples = 100;          output_samples = 1;        kl_a = 1
    #anom_rate = 0.01
    # load data directory
    #directory1 = 'Data/'
    data_filename = 'dataset/'+dataset+"_normal_anomaly_mixed.npz" # directory1+'xy_o_30.npz'
    # run parameters
    split_ratio = 0.5
    anom_split = str(np.round(1-split_ratio,decimals=3)).replace(".","_")
    file_name = run_comment+'_'+dataset+"_h_"+str(h_dim)+"_z_"+str(z_dim)
    load_file_name = file_name

    ###############################################################################
    # ========================= get data and variables ========================== #
    ###############################################################################

    # get data
    x_train, x_valid, x_test, y_train, y_valid, y_test =   get_data(data_filename,split_ratio,real_data=real_data)
    # calculate sizes
    train_size, valid_size, test_size, y_dim, X_dim =   \
        x_train.shape[0], x_valid.shape[0],x_test.shape[0], y_train.shape[1], x_train.shape[1]
    print("X-dim:",X_dim)
    # other
    num_batches = train_size//batch_size;       epoch = 0;      save_epochs = 0;    b_t_ratio = 2#train_size//batch_size
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
            l_rate = tf.placeholder_with_default(input = learning_rate,shape=(),name='var_learning_rate')
        with tf.name_scope("latent"):
            z = tf.placeholder(tf.float32, shape=[None, z_dim],name="latent_vars")
        # introduce convenience function for batch norm
        batch_norm_layer = partial(tf.layers.batch_normalization, training=is_train, momentum=0.95)
        # =============================== Q(z|X) ====================================
        def encode(x, scope='encoder', reuse=False, drop_prob=drop_probs, is_train=is_train,
                   batch_norm=batch_norm, dropout = dropout):
            '''
            Discriminative model (decoder)
            Input:      x : input data
            Returns:    z_mu, Z_logvar : mean and standard deviation of z
            '''
            with tf.variable_scope("encoder", reuse = reuse):
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
                inputs = z 
                # calculate hidden
                h = tf.layers.dense(inputs, h_dim, activation=None,kernel_initializer=initialize,
                                    kernel_regularizer=regularize,name='d_hidden_1')
                if dropout:
                    h = tf.layers.dropout(h, training=is_train, rate=drop_probs, seed=128)
                if batch_norm:
                    h = batch_norm_layer(h)
                h = tf.nn.elu(h)

                # calculate the mean of the output (Gausian)
                x_mu = tf.layers.dense(h, X_dim, activation=None,kernel_initializer=initialize,
                                       kernel_regularizer=regularize,name='x_mu')
                if batch_norm:
                    x_mu = batch_norm_layer(x_mu)
            return x_mu

        # =============================== ELBO ====================================
        def loss(X,x_sample,z_mu,z_logvar,reuse=None, n_z_samples=1, alpha_KL=alpha_KL,rc_weight=rc_weight):
            with tf.name_scope("loss"):
                # E[log P(X|z)]
                recon_loss = 0.5 * tf.reduce_sum(tf.square(X-x_sample),axis=1)
                # loop for number of MC samples
                for i in range(n_z_samples-1):
                    z_sample = sample_z(z_mu, z_logvar)
                    x_mu  = decode(z_sample,reuse=reuse)
                    x_sample = x_mu# sample_z(x_mu, x_logvar)
                    recon_loss += 0.5 * tf.reduce_sum(tf.square(X-x_sample),axis=1)
                # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
                kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, axis=1)
                # Regularisation cost
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                # VAE loss
                ELBO = rc_weight*(recon_loss / n_z_samples) + alpha_KL * kl_loss
                vae_loss = tf.add_n([tf.reduce_mean(ELBO)+ tf.reduce_sum(reg_variables)])
                # summary
            with tf.name_scope("Summaries"):
                tf.summary.scalar("Batch_loss",tf.reduce_mean(ELBO))
                merger = tf.summary.merge_all()
            return vae_loss,ELBO,merger, tf.reduce_mean(kl_loss), tf.reduce_mean(recon_loss)

        # =============================== TRAINING ====================================
        # embed (encode)
        z_mu, z_logvar = encode(X)
        with tf.name_scope("latent"):
            z_sample = sample_z(z_mu, z_logvar)
        # generate (decode)
        x_mu = decode(z_sample, reuse=None)
        # sample x
        x_sample = x_mu #sample_z(x_mu,x_logvar)
        # loss
        vae_loss, ELBO, merger,kl_loss,rec_loss= loss(X,x_sample,z_mu,z_logvar,reuse=True, n_z_samples=1,
                                                      alpha_KL=alpha_KL,rc_weight=rc_weight)
        with tf.name_scope("optimiser"):
            # optimiser
            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            # collect batch norm losses
            if batch_norm:
                bn_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(bn_update):
                    solver = optimizer.minimize(vae_loss)
            solver = optimizer.minimize(vae_loss)

        # =============================== TESTING =================================== #
        with tf.name_scope("Testing"):
            z_mu_t, z_logvar_t = encode(X,reuse=True)
            z_sample_t = sample_z(z_mu_t, z_logvar_t)
            x_mu_t = decode(z_sample_t, reuse=True)
            # sample x
            x_sample_t = x_mu_t#sample_z(x_mu_t, x_logvar_t)
            # loss
            _,ELBO_t,_,_,_ = loss(X,x_sample_t,z_mu_t,z_logvar_t,reuse=True,\
													n_z_samples=num_z_samples,alpha_KL=1.0,rc_weight=1.0)

        # =============================== Session =============================== #
        if TRAIN_VAE:
            sum_writter = tf.summary.FileWriter(save_sum_dir,graph)
            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
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
                x_train, y_train = ph.shuffle_data(x_train, y_train)
                # For each epoch train with mini-batches of size (batch_size)
                for batch_num in range(train_size//batch_size):
                    # anneal KL cost (full cost after 50.000 batches)
                    kl_a = 2*(-.5+1/(1+np.exp(-count*anneal_rate)))
                    #print("batch_number:{}".format(batch_num))
                    X_mb = x_train[batch_num*batch_size:(batch_num+1)*batch_size,]
                    # train
                    train_dict={X: X_mb, is_train:True, drop_prob:drop_probs, alpha_KL : kl_a,rc_weight:b_t_ratio,l_rate:learning_rate}
                    _, loss,k_,r_ = sess.run([solver, vae_loss,kl_loss,rec_loss], feed_dict= train_dict)
                    epoch_loss+=loss; 	kl+= k_;	r_loss+= r_
                    count+=1
                # print progress
                ph.progress(epoch,(epoch_loss/num_batches),(r_loss/num_batches),(kl/num_batches),\
                            time.time()-epoch_time)
                training_loss.append(epoch_loss/num_batches)
                rc_training_loss.append(r_loss / num_batches)
                kl_training_loss.append(kl / num_batches)
                # validate
                if epoch >0 and epoch%valid_every ==0:
                    vloss = 0
                    valid_dict={X: x_valid,is_train:False,drop_prob:0.0}
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
                        l2_reg,save_epochs, early_stopping,X_dim,train_size,valid_size,test_size,
                        sum(y_train[:,0]==0),sum(y_valid[:,0]==0),sum(y_test[:,0]==0),save_dir)
            # print training curves
            plt.figure()
            tl=np.array(rc_training_loss) + np.array(kl_training_loss)
            plt.plot(tl, 'b', label='training loss')
            plt.plot(rc_training_loss, 'm', label='reconstruction loss')
            plt.plot(validation_loss, 'r', label='validation loss')
            plt.plot(kl_training_loss, 'g', label='KL loss')
            plt.title('Training Curves\nDataset:{}, Method:{}'.format(dataset,mode))
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
        ##############################################################################
		#=====================   Threshold Selection Routine   ======================#
		##############################################################################
        # break the validation set evaluation into 'n_test_steps' steps to avoid memory overflow
        valid_elbo = np.zeros([valid_size, 1])
        for j in range(n_test_steps - 1):
            start = j * (valid_size // n_test_steps);
            stop = (j + 1) * (valid_size // n_test_steps)
            valid_dict = {X: x_valid[start:stop], is_train: False, drop_prob: 0.0}
            x_elbo_v = sess.run([ELBO_t], feed_dict=valid_dict)
            valid_elbo[start:stop, 0] = x_elbo_v[0]
        # compute the last slice separately since it might have more points
        valid_dict = {X: x_valid[stop:], is_train: False, drop_prob: 0.0}
        x_elbo_v = sess.run([ELBO_t], feed_dict=valid_dict)
        valid_elbo[stop:, 0] = x_elbo_v[0]
        valid_elbo = np.clip(valid_elbo,None,1e3)
        # call the detection routine and obtain the anomaly threshold (based on the anomalous rate estimate)
        anomaly_threshold = unsupervised_threshold(valid_elbo, anom_rate)
        #ph.plot_hist(valid_elbo,dataset,'Validation Set_figure 4',  plots=plots)
        #=================== Evaluation on Test Set ==========================#
        # normalize data
        x_test = ph.p_normalize(x_test,train_mean,train_std)
        if verbose:
            print("#=========================Test Set=============================#")
        tn,fn,fp,tp = [0 for i in range(4)]
        y_pred = np.zeros([test_size,2])
        t_elbo = np.zeros([test_size,1])
        start_time = time.time()
        # break the test set evaluation into 'n_test_steps' steps to avoid memory overflow
        for j in range(n_test_steps-1):
            start = j * (test_size//n_test_steps); stop = (j+1) * (test_size//n_test_steps)
            test_dict = {X: x_test[start:stop], is_train: False, drop_prob: 0.0}
            x_elbo_t = sess.run([ELBO_t], feed_dict= test_dict)
            t_elbo[start:stop,0] = x_elbo_t[0]
        # compute the last slice separately since it might have more points
        test_dict = {X: x_test[stop:], is_train: False, drop_prob: 0.0}
        x_elbo_t = sess.run([ELBO_t], feed_dict=test_dict)
        t_elbo[stop:,0] = x_elbo_t[0]
        t_elbo = np.clip(t_elbo,None,1e3)
        # save accuracy rates
        tn = sum([t_elbo[j] < anomaly_threshold and y_test[j,0]==1 for j in range(x_test.shape[0])])
        fp = sum([t_elbo[j] > anomaly_threshold and y_test[j,0]==1 for j in range(x_test.shape[0])])
        tp = sum([t_elbo[j] > anomaly_threshold and y_test[j,0]==0 for j in range(x_test.shape[0])])
        fn = sum([t_elbo[j] < anomaly_threshold and y_test[j,0]==0 for j in range(x_test.shape[0])])
        end_time = time.time()
        # calculate and report total stats
        # scores
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        f01 = (1+(1/fn_ratio)**2) * precision * recall / ((1/fn_ratio**2)*precision + recall)
        # bring back the auc score
        print("f0.1 score",f01)
        print(type(f01))
        auc,thresh=ph.auc_plot(t_elbo,y_test,anomaly_threshold,f01[0])

        # print the anomaly threshold
        print("AUC:",auc)
        print('Anomaly threshold: {}', anomaly_threshold)
        ph.report_stats("\nTest Statistics ", t_elbo, verbose=verbose)
        #ph.report_stats("\nTest Statistics - Anomaly Data", t_anomaly_elbo.mean(axis=1), verbose=verbose)
        ph.scores_x(tp,fp,fn,tn, verbose=verbose)
        #elbo_hist(t_normal_elbo.mean(axis=1), t_anomaly_elbo.mean(axis=1), anomaly_threshold, 'Test Set','figure 5',
        #          save_print_dir, dataset, plots=plots)
        #ph.plot_hist(t_elbo, save_print_dir, dataset, title='Test set', f_name="" ,figsize=(5, 5), plots=plots)
        #ph.plot_hist(t_anomaly_elbo.mean(axis=1), save_print_dir, dataset, title='Anomaly', f_name="" ,figsize=(5, 5), plots=plots)
        # Compute confusion matrix
        cnf_matrix = np.array([[int(tp),int(fn)],
                               [int(fp),int(tn)]])
        np.set_printoptions(precision=2)
        class_names = np.array(['Anomaly','Normal'],dtype='<U10')
        # Plot non-normalized confusion matrix
        if plots:
            ph.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                    title=" Confusion matrix\n"+"Dataset: "+dataset+'-'+mode, plots=plots)
            plt.savefig(save_print_dir+dataset+"_threshold_"+str(anomaly_threshold)+'_conf_mat.png')
            plt.show()
        if verbose:
            print('total inference time for {} data points (x{} samples each):{:6.3}s '.format(y_test.shape[0],\
                                                                            num_z_samples,(end_time-start_time)))
    # save all elbo results to a file for post-processing
    np.savez(save_dir + file_name + 'res', descr=run_comment,
             val_elbo=valid_elbo,y_val=y_valid,x_val=x_valid,
             tst_elbo=t_elbo,y_tst=y_test,x_tst=x_test)
    ph.saveres2txt(save_print_dir, file_name, dataset, anomaly_threshold, tp, fp, tn, fn, f1, auc, f01, precision,
                   recall)

    if bayesian_opt: return [tp, fp, tn, fn], bo_results,valid_elbo,t_elbo,anomaly_threshold,y_valid,y_test,save_dir+file_name+'res'
    else: return [tp, fp, tn, fn],{},valid_elbo,t_elbo,anomaly_threshold,y_valid,y_test,save_dir+file_name+'res'

if __name__ == "__main__":
    conf_matrix,bo_results,valid_elbo,test_elbo,elbo_threshold,y_valid,y_test,res_np_file = \
                    net(TRAIN_VAE =False, load_model =True, real_data= True, fp_ratio=1,fn_ratio=10,anom_rate = 0.077, \
                        bayesian_opt=False, anomaly_threshold=50, batch_norm=False, dropout=True,verbose=True, plots=True)
    dataset = 'shuttle'
    # produce validation plot
    ph.post_plot_unsuper2(valid_elbo,elbo_threshold)
    # post processing
    # get ratios for different values of anomaly rates
    precision, recall, f1, f01, tn, fn, tp, fp, ratios = ph.unsupervised_elbo_analysis(valid_elbo, test_elbo, y_test, fn_ratio=10)
    print(precision);  print(recall);  print(f1) ; print(f01) ;print(tp,fp);  print(tn,fn)
    # get ratios by selecting the cluster of anomalies
    # precision, recall, f1, f01, tn, fn, tp, fp = ph.select_elbo_region(valid_elbo, test_elbo, y_test, 10, 50, fn_ratio=10)
    # cnf_matrix = np.array([[int(tp), int(fn)],
    #                        [int(fp), int(tn)]])
    # np.set_printoptions(precision=2)
    # class_names = np.array(['Anomaly', 'Normal'], dtype='<U10')
    # # Plot non-normalized confusion matrix
    # ph.plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                              title=" Confusion matrix\n" + "Dataset: " + dataset + " - Selecting clustered anomalies", plots=True)
    # # plot elbos and zoom around cluster of anomalies
    # start,stop, threshold = 10,0,30
    # ph.post_plot_unsuper(valid_elbo,test_elbo,threshold ,start,stop)