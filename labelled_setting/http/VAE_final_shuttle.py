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
@author: Heraclitus
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import time
import datetime
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


def out(tmp):
    """tmp is a list of numbers"""
    outs = []
    mean = sum(tmp)/(1.0*len(tmp))
    var = sum((tmp[i] - mean)**2 for i in range(0, len(tmp)))/(1.0*len(tmp))
    std = var**0.5
    outs = [tmp[i] for i in range(0, len(tmp)) if abs(tmp[i]-mean) > 10*std]
    return outs


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
        # normal_data
        x_train = mydata['x_train_normal']
        x_valid = mydata['x_valid_normal']
        x_test = mydata['x_test_normal']
        x_anomalies = mydata['x_anomalies']
        # anomalies
        # x_anomalies = mydata['x_anomalies']
        # get sizes
        anomalies_size = x_anomalies.shape[0]
        # split anomaly data
        # train/test
        split_ratio = 0.5
        split_ratio = 1 / split_ratio
        split = int(anomalies_size // split_ratio)
        anom_valid = x_anomalies[:split, ]
        anom_test = x_anomalies[split:, ]
        return x_train, x_valid, x_test, anom_valid, anom_test
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
    VAE_BO.maximize(n_iter=20, **gp_params)
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
        plt.axis([0,  max(anomaly_elbo)*1.1, 0, 0.8])
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
    run_comment = mode+"_linVAElin_2"
    dataset = "Shuttle"
    save_dir = ("./"+mode+"_"+dataset+"/")
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
    epochs  = 251;              batch_size = 100;       early_stopping = 40
    learning_rate = 1e-5;       l2_reg = 1e-2;          drop_probs = 0.15
    val_epochs = 1;             anneal_rate = 5e-5;     n_test_steps = 100
    # sampling parameters & classification criterion
    num_z_samples = 100;          output_samples = 1;        kl_a = 1
    # load data directory
    #directory1 = 'Data/'
    data_filename = 'dataset/'+dataset+"_normal_anomaly_seperate.npz" # directory1+'xy_o_30.npz'
    # run parameters
    split_ratio = 0.5
    anom_split = str(np.round(1-split_ratio,decimals=3)).replace(".","_")
    file_name = run_comment+'_'+dataset+"_h_"+str(h_dim)+"_z_"+str(z_dim)
    load_file_name = file_name

    ###############################################################################
    # ========================= get data and variables ========================== #
    ###############################################################################

    # get data
    x_train, x_valid, x_test, anom_valid, anom_test =   get_data(data_filename,split_ratio,real_data=real_data)
    # calculate sizes
    train_size, valid_size, test_size, anom_valid_size, anom_test_size, X_dim =   \
        x_train.shape[0], x_valid.shape[0],x_test.shape[0], anom_valid.shape[0], anom_test.shape[0], x_train.shape[1]
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
                # calculate the std of the output (Gausian)
                x_logvar = tf.layers.dense(h, X_dim, activation=None,kernel_initializer=initialize,
                                           kernel_regularizer=regularize,name='x_logvar')
                if batch_norm:
                    x_logvar = batch_norm_layer(x_logvar)
            return x_mu, x_logvar

        # =============================== ELBO ====================================
        def loss(X,x_sample,z_mu,z_logvar,reuse=None, n_z_samples=1,alpha_KL=alpha_KL,rc_weight=rc_weight):
            with tf.name_scope("loss"):
                # E[log P(X|z)]
                recon_loss = tf.clip_by_value(0.5 * tf.reduce_sum(tf.square(X-x_sample),axis=1),1e-8,1e8)
                # loop for number of MC samples
                for i in range(n_z_samples-1):
                    z_sample = sample_z(z_mu, z_logvar)
                    x_mu, x_logvar  = decode(z_sample,reuse=reuse)
                    x_sample = sample_z(x_mu, x_logvar)
                    recon_loss += tf.clip_by_value(0.5 * tf.reduce_sum(tf.square(X-x_sample),axis=1),1e-8,1e8)
                # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
                kl_loss = tf.clip_by_value(0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, axis=1),1e-8,1e8)
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
        x_mu, x_logvar = decode(z_sample, reuse=None)
        # sample x
        x_sample = sample_z(x_mu,x_logvar)
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
            x_mu_t, x_logvar_t = decode(z_sample_t, reuse=True)
            # sample x
            x_sample_t = sample_z(x_mu_t, x_logvar_t)
            # loss
            _,ELBO_t,_,_,_ = loss(X,x_sample_t,z_mu_t,z_logvar_t,reuse=True,\
													n_z_samples=num_z_samples,alpha_KL=1.0)

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
                x_train = ph.shuffle_x_data(x_train)
                # For each epoch train with mini-batches of size (batch_size)
                for batch_num in range(train_size//batch_size):
                    # anneal KL cost (full cost after 50.000 batches)
                    kl_a = 2*(-.5+1/(1+np.exp(-count*anneal_rate)))
                    #print("batch_number:{}".format(batch_num))
                    X_mb = x_train[batch_num*batch_size:(batch_num+1)*batch_size,]
                    # train
                    train_dict={X: X_mb,is_train:True,drop_prob:drop_probs, alpha_KL : kl_a,rc_weight:b_t_ratio,l_rate:learning_rate}
                    _, loss,k_,r_ = sess.run([solver, vae_loss,kl_loss,rec_loss], feed_dict= train_dict)
                    epoch_loss+=loss; 	kl+= k_;	r_loss+= r_; count+=1
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
            np.savez(save_print_dir+file_name,\
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
        ##############################################################################
		#=====================   Threshold Selection Routine   ======================#
		##############################################################################

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
        normal_elbo = np.clip(normal_elbo,None,1e3)
        # evaluate ELBO on the anomaly valildation-set
        for j in range(n_test_steps - 1):
            start = j * (anom_valid_size // n_test_steps);
            stop = (j + 1) * (anom_valid_size // n_test_steps)
            anomalous_valid_dict = {X: anom_valid.reshape([-1,X_dim])[start:stop], is_train: False, drop_prob: 0.0}
            a_elbo_v = sess.run([ELBO_t], feed_dict=anomalous_valid_dict)
            anomaly_elbo[start:stop, 0] = a_elbo_v[0]
        # compute the last slice separately since it might have more points
        anomalous_valid_dict = {X: anom_valid.reshape([-1,X_dim])[stop:], is_train: False, drop_prob: 0.0}
        a_elbo_v = sess.run([ELBO_t], feed_dict=anomalous_valid_dict)
        anomaly_elbo[stop:, 0] = a_elbo_v[0]
        anomaly_elbo = np.clip(anomaly_elbo,None,1e3)
        # send the data to the scoring function and call the bayesian opt routine
		# and collect the results
        if bayesian_opt:
            # anomaly_threshold = 150
            anomaly_threshold, bo_results = threshold_selector(normal_elbo,anomaly_elbo,\
                                                               fp_ratio=fp_ratio,fn_ratio=fn_ratio)
            plot_bayes_opt(bo_results,fn_ratio, 'figure 3', save_print_dir, dataset, plots=plots)
        elbo_hist(normal_elbo, anomaly_elbo, anomaly_threshold, 'Validation Set',\
                                                                            'figure 4', save_print_dir, dataset, plots=plots)

        #=================== Evaluation on Test Set ==========================#
        # normalize data
        x_test = ph.p_normalize(x_test,train_mean,train_std)
        anom_test = ph.p_normalize(anom_test,train_mean,train_std)
        #anomalous_test_dict = { X: anom_test.reshape([-1,X_dim]), is_train:False,drop_prob:0.0}
        if verbose:
            print("#=========================Test Set=============================#")
        print('Anomaly threshold: {}', anomaly_threshold)
        start_time = time.time()

        t_normal_elbo = np.zeros([test_size, 1])
        t_anomaly_elbo = np.zeros([anom_test_size, 1])
        # evaluate ELBO on the normal validation-set
        for j in range(n_test_steps - 1):
            start = j * (test_size // n_test_steps);
            stop = (j + 1) * (test_size // n_test_steps)
            normal_test_dict = {X: x_test[start:stop], is_train: False, drop_prob: 0.0}
            x_elbo_t = sess.run([ELBO_t], feed_dict=normal_test_dict)
            t_normal_elbo[start:stop, 0] = x_elbo_t[0]
        # compute the last slice separately since it might have more points
        normal_test_dict = {X: x_test[stop:], is_train: False, drop_prob: 0.0}
        x_elbo_t = sess.run([ELBO_t], feed_dict=normal_test_dict)
        t_normal_elbo[stop:, 0] = x_elbo_t[0]
        t_normal_elbo = np.clip( t_normal_elbo,None,1e3)
        start = stop = 0
        # evaluate ELBO on the anomaly test-set
        for j in range(n_test_steps - 1):
            start = j * (anom_test_size // n_test_steps);
            stop = (j + 1) * (anom_test_size // n_test_steps)
            anomalous_test_dict = {X: anom_test[start:stop], is_train: False, drop_prob: 0.0}
            a_elbo_t = sess.run([ELBO_t], feed_dict=anomalous_test_dict)
            t_anomaly_elbo[start:stop, 0] = a_elbo_t[0]
        # compute the last slice separately since it might have more points
        anomalous_test_dict = {X: anom_test[stop:], is_train: False, drop_prob: 0.0}
        a_elbo_t = sess.run([ELBO_t], feed_dict=anomalous_test_dict)
        t_anomaly_elbo[stop:, 0] = a_elbo_t[0]
        t_anomaly_elbo = np.clip(t_anomaly_elbo,None,1e3)
        # save accuracy rates
        tn = np.sum(t_normal_elbo < anomaly_threshold)
        fp = np.sum(t_normal_elbo > anomaly_threshold)
        tp = np.sum(t_anomaly_elbo > anomaly_threshold)
        fn = np.sum(t_anomaly_elbo  < anomaly_threshold)
        y_pred_n = t_normal_elbo > anomaly_threshold
        y_pred_a = t_anomaly_elbo > anomaly_threshold
        end_time = time.time()
        y_pred = np.concatenate((y_pred_n,y_pred_a),axis=0)
        y_true = np.concatenate((np.zeros([test_size]),np.ones([anom_test_size])), axis=0)
        # calculate the AUC-score
        t_elbo = np.concatenate((t_normal_elbo,t_anomaly_elbo),axis=0)
        # calculate and report total stats
        # scores
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        f01 = (1+(1/fn_ratio)**2) * precision * recall / ((1/fn_ratio**2)*precision + recall)
        auc,thresh=ph.auc_plot(t_elbo,y_true,anomaly_threshold,f01)
        print("AUC:",auc)
        ph.report_stats("\nTest Statistics - Normal Data", t_normal_elbo, verbose=verbose)
        ph.report_stats("\nTest Statistics - Anomaly Data", t_anomaly_elbo, verbose=verbose)
        ph.scores_x(tp,fp,fn,tn, verbose=verbose)
        elbo_hist(t_normal_elbo, t_anomaly_elbo, anomaly_threshold, 'Test Set','figure 5',
                  save_print_dir, dataset, plots=plots)
       # ph.plot_2hist(t_normal_elbo,t_anomaly_elbo, save_print_dir, dataset, title='Normal', f_name="" ,figsize=(5, 5), plots=plots)
        #ph.plot_hist(t_anomaly_elbo, save_print_dir, dataset, title='Anomaly', f_name="" ,figsize=(5, 5), plots=plots)
        # Compute confusion matrix
        cnf_matrix = np.array([[int(tp),int(fn)],
                               [int(fp),int(tn)]])
        np.set_printoptions(precision=2)
        class_names = np.array(['Anomaly','Normal'],dtype='<U10')
        # Plot non-normalized confusion matrix
        if plots:
            ph.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                    title=" Confusion matrix\n"+"Dataset: "+dataset+" - "+mode, plots=plots)
            plt.savefig(save_print_dir+dataset+"_threshold_"+str(round(anomaly_threshold,2))+'_conf_mat.png')
            plt.show()
        if verbose:
            print('total inference time for {} data points (x{} samples each):{:6.3}s '.format(y_true.shape[0],\
                                                                            output_samples,(end_time-start_time)))
    # save all elbo results to a file for post-processing
    np.savez(save_dir + file_name + 'res', descr=run_comment,
             val_norm_elbo=normal_elbo,val_anom_elbo=anomaly_elbo,x_val=x_valid,
             tst_norm_elbo=t_normal_elbo,tst_anom_elbo=t_anomaly_elbo,x_tst=x_test)
    ph.saveres2txt(save_print_dir, file_name, dataset,round(anomaly_threshold,2), tp,fp,tn,fn,f1, auc, f01,precision,recall)
    # return statements
    if bayesian_opt: return [tp, fp, tn, fn], bo_results,normal_elbo,anomaly_elbo,t_normal_elbo,t_anomaly_elbo,save_dir + file_name + 'res'
    else: return [tp, fp, tn, fn],{},normal_elbo,anomaly_elbo,t_normal_elbo,t_anomaly_elbo,save_dir + file_name + 'res'


if __name__ == "__main__":
    conf_matrix,bo_results,n_elbo_valid,a_elbo_valid,n_elbo_test,a_elbo_test,res_np_file  =\
        net(TRAIN_VAE =False, load_model = True, real_data= True, bayesian_opt=False, anomaly_threshold=16,
            fp_ratio=1,fn_ratio=1,verbose=True, plots=True,batch_norm=False, dropout=True)
