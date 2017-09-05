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
from skimage.data import camera
from skimage.filters import roberts
from skimage import transform as transf
from sklearn.metrics import confusion_matrix
import itertools
import project_helper as ph
import importlib
from functools import partial
importlib.reload(ph)
from tensorflow.examples.tutorials.mnist import input_data


def get_data(file_name, split_ratio=0.6, real_data=True,sigma=0.0):
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
        # anomalies
        x_anomalies = mydata['x_anomalies']
        # get sizes
        anomalies_size = x_anomalies.shape[0]
        # split anomaly data
        # train/test
        split = int(anomalies_size * split_ratio)
        anom_valid = x_anomalies[:split, ]
        anom_test = x_anomalies[split:, ]
        return x_train, x_valid, x_test, anom_valid, anom_test
    else:
        return make_data(sigma=sigma)


def make_data(t=10000,n_steps=100,sigma=0.0):
    '''
    Functions that makes fake training data for testing purposes
    Returns: x_train, x_valid, x_test, anom_valid, anom_test
    '''
    t_min, t_max = 0, 100
    resolution = 0.05
     #t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
    def time_series1(t):
        l = t.shape[0]
        l = l//2
        signal1 = np.sin(3*t[0:l,:]) / 3 + 0.25 * np.sin(9*t[0:l,:])
        label1 = np.ones_like(signal1)
        signal2 = np.sin(2 * t[l:,:]) / 3 + 0.25 * np.sin(4 * t[l:,:])
        label2 = np.ones_like(signal2)
        label2 *= 2
        return np.concatenate([signal1,signal2],axis=0)+ np.random.normal(0,sigma,t.shape),np.concatenate([label1,label2.astype(int)],axis=0)
    def time_series2(t):
        signal = 1.5*(np.sin(3.5*t) / 3)  * np.sin(t * 2.1)+ np.random.normal(0,sigma,t.shape)
        label = np.zeros_like(signal)
        return signal,label
    def next_batch(batch_size, n_steps):
        t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
        Ts = t0 + np.arange(0., n_steps) * resolution
        xs_n,label_n = time_series1(Ts)
        xs_a,label_a = time_series2(Ts)
        return xs_n[:, :].reshape(-1, n_steps), label_n[:, :].reshape(-1, n_steps),\
               xs_a[:, :].reshape(-1, n_steps), label_a[:, :].reshape(-1, n_steps)
    x_norm,l_norm, x_anom,l_anom = next_batch(t,n_steps)
    a = np.random.permutation(t)
    return x_norm[a[0:int(t*0.6)],:], x_norm[a[int(t*0.6):int(t*0.8)],:], x_norm[a[int(t*0.8):t],:], \
           l_norm[a[0:int(t * 0.6)], :], l_norm[a[int(t * 0.6):int(t * 0.8)], :], l_norm[a[int(t * 0.8):t], :], \
           x_anom[a[0:int(t*0.5)],:], x_anom[a[int(t*0.5):t],:],l_anom[a[0:int(t*0.5)],:], l_anom[a[int(t*0.5):t],:]

def weighted_loss(anomaly_threshold):
    conf_matrix = net(TRAIN_VAE = False, load_model = True, real_data=False,
                      anomaly_threshold=anomaly_threshold, verbose=False, plots=False )
    tp, fp, tn, fn = conf_matrix[0:4]
    w_loss = 80*fn+20*fp
    return w_loss


def net(TRAIN_VAE = True, load_model = True, real_data=True, batch_norm=False, dropout=False,
        anomaly_threshold = 10, verbose=True, plots=True ):

    # Set random generator seed for reproducibility
    np.random.seed(168)
    tf.set_random_seed(168)
    # Reset the default graph
    tf.reset_default_graph()

    ###############################################################################
    # ======================= file and directory names ========================== #
    ###############################################################################
    run_comment = "lin_CVAE_rnn_"
    dataset = "synthetic"
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
    RNN_size = 32;             n_rnn_layers = 3;
    # hidden and latent space parameters
    z_dim = 100;                h_dim = 512
    # running parameters (hyper-parameters
    epochs  = 41;              batch_size = 100;     early_stopping = 10
    learning_rate = 1e-4;       l2_reg = 3e-3;       drop_probs = 0.2
    # sampling parameters & classification criterion
    n_z_samples = 10;          output_samples = 1;   criterion = 'precision';
    anneal_rate = 5e-5;        fn_ratio = 1
    # observation noise
    sigma = 0.0
    # load data directory
    directory1 = 'LSTM_Data/'
    data_filename = directory1+'synthetic_data.npz'
    # run parameters
    split_ratio = 0.5
    anom_split = str(np.round(1-split_ratio,decimals=3)).replace(".","_")
    file_name = run_comment+'_'+dataset+"_h_dim_"+str(h_dim)+"_zDim_"+str(z_dim)\
              + "_split_ratio"+anom_split
    load_file_name = file_name

    ###############################################################################
    # ========================= get data and variables ========================== #
    ###############################################################################

    # get data
    x_train, x_valid, x_test, l_norm_train, l_norm_valid, l_norm_test, \
    anom_valid, anom_test, l_anom_valid, l_anom_test = get_data(data_filename,split_ratio,real_data=real_data,sigma=0)
    # calculate sizes
    train_size, valid_size, test_size, anom_valid_size, anom_test_size, X_dim =   \
        x_train.shape[0], x_valid.shape[0],x_test.shape[0], anom_valid.shape[0], anom_test.shape[0], x_train.shape[1]
    # other
    num_batches = train_size//batch_size;       epoch = 0;      save_epochs = 0;
    # Training losses containers
    # print(train_size)
    # print(valid_size)
    # print(test_size)
    # print(anom_valid_size)
    # print(anom_test_size)
    best_eval_loss = np.inf
    training_loss = []
    validation_loss= []

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
            aux = tf.placeholder(tf.float32,shape=[None, X_dim],name='auxiliary_variable')
            drop_prob = tf.placeholder(tf.float32, shape=(),name='dropout_prob')
            alpha_KL = tf.placeholder_with_default(input = 1.0,shape=(),name='KL_annealing')
            rc_weight = tf.placeholder_with_default(input = 1.0,shape=(),name='reconstruction_loss_weight')
            l_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
            is_train = tf.placeholder_with_default(input = False,shape=(),name='train_test_state')
        with tf.name_scope("latent"):
            z = tf.placeholder(tf.float32, shape=[None, z_dim],name="latent_vars")
        # introduce convenience function for batch norm
        batch_norm_layer = partial(tf.layers.batch_normalization, training=is_train, momentum=0.95)
        # =============================== Q(z|X) ====================================
        def encode(x, scope='encoder', reuse=False, drop_prob=drop_probs, is_train=is_train):
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
        def decode(z,aux, scope = 'decoder', reuse=False, drop_prob=drop_probs, is_train=is_train):
            '''
            Generative model (decoder)
            Input:      z : latent space data
            Returns:    x : generated data
            '''
            with tf.variable_scope("decoder", reuse=reuse):

                #======  Px(z)  ======#
                inputs  = z #f.concat(axis=1, values=[z,aux])
                # rnn_inputs = tf.layers.dense(inputs, batch_size, activation=None,kernel_initializer=initialize,
                #                     kernel_regularizer=regularize,name='RNN_input_afine_layer')
                # if dropout:
                #     rnn_inputs = tf.layers.dropout(rnn_inputs , training=is_train, rate=drop_probs, seed=128)
                # if batch_norm:
                #     rnn_inputs = batch_norm_layer(rnn_inputs )
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
                #inputs = tf.expand_dims(inputs,-1)
                inputs = tf.stack([inputs ,aux],axis=-1)
                #    [10,50,1]
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
            return x_mu

        # =============================== ELBO ====================================
        def loss(X,x_mu,z_mu,z_logvar,aux,reuse=None, n_z_samples=1,alpha_KL=alpha_KL):
            with tf.name_scope("loss"):
                # E[log P(X|z)]
                recon_loss = 0.5 * tf.reduce_sum(tf.square(x_mu-X),axis=1)/n_z_samples
                for i in range(n_z_samples-1):
                    z_sample = sample_z(z_mu, z_logvar)
                    x_sample= decode(z_sample,aux,reuse=reuse)
                    recon_loss += 0.5 * tf.reduce_sum(tf.square(X-x_mu),axis=1)/n_z_samples
                # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
                kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, axis=1)
                # Regularisation cost
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                # VAE loss
                ELBO = recon_loss + alpha_KL * kl_loss
                vae_loss = tf.add_n([tf.reduce_mean(ELBO)+ tf.reduce_sum(reg_variables)])
                # summary
            with tf.name_scope("Summaries"):
                #tf.summary.scalar("ELBO",ELBO)
                tf.summary.scalar("Batch_loss",tf.reduce_mean(ELBO))
                merger = tf.summary.merge_all()
            return vae_loss,ELBO,merger

        # =============================== TRAINING ====================================
        # embed (encode)
        z_mu, z_logvar = encode(X)
        with tf.name_scope("latent"):
            z_sample = sample_z(z_mu, z_logvar)
        # generate (decode)
        x_mu= decode(z_sample, aux, reuse=None)
        # loss
        vae_loss, ELBO, merger = loss(X,x_mu,z_mu,z_logvar,aux,reuse=True, n_z_samples=1,alpha_KL=alpha_KL)
        with tf.name_scope("optimiser"):
            # updater
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = train_step.compute_gradients(vae_loss)
            clipped_grads = [(tf.clip_by_value(grad, -2,2),var) for grad,var in grads]
            solver = train_step.apply_gradients(clipped_grads)
            bn_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # =============================== TESTING =================================== #
        with tf.name_scope("Testing"):
            z_mu_t, z_logvar_t = encode(X,reuse=True)
            z_sample_t = sample_z(z_mu_t, z_logvar_t)
            x_mu_t = decode(z_sample_t,aux, reuse=True)
            # loss
            _,ELBO_t,_ = loss(X,x_mu_t,z_mu_t,z_logvar_t,aux,reuse=True, n_z_samples=10)

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
            print(" -----Training Started-----")
            count = 0
            for epoch in range(epochs):
                epoch_loss = 0
                epoch_time = time.time()
                #x_epoch = tf.add(x_train,tf.random_normal(x_train.shape ,mean=0.0,stddev=0.05,dtype=tf.float64,seed=None,name=None))
                x_epoch, l_x_train = ph.shuffle_data(x_train,l_norm_train)
                # For each epoch train with mini-batches of size (batch_size)
                for batch_num in range(train_size//batch_size):
                    X_mb = x_epoch[batch_num*batch_size:(batch_num+1)*batch_size,]
                    lx_mb = l_x_train[batch_num*batch_size:(batch_num+1)*batch_size,]
                    # anneal KL cost (full cost after 50.000 batches)
                    kl_a = 2*(-.5+1/(1+np.exp(-count*anneal_rate)))
                    #y_mb = y_train[batch_num*batch_size:(batch_num+1)*batch_size,]
                    # train
                    _, loss,_ = sess.run([solver, vae_loss,bn_update], feed_dict={X: X_mb, aux:lx_mb, is_train: True,
                                                                                  drop_prob: drop_probs,alpha_KL:kl_a})
                    epoch_loss+=loss
                    count+=1
                # print progress
                ph.progress(epoch,(epoch_loss/num_batches), 1,1,time.time()-epoch_time)
                training_loss.append(epoch_loss/num_batches)
                # validate
                if epoch >0 and epoch%1 ==0:
                    vloss = 0
                    vloss, vaeloss = sess.run([vae_loss,merger], feed_dict={ X: x_valid, aux:l_norm_valid,is_train:False,drop_prob:drop_probs} )
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
        # calculate ELBO statistics on the validation set   RECON_loss, KL_loss
        x_elbo = sess.run([ELBO_t], feed_dict={ X: x_valid, aux:l_norm_valid, is_train:False,drop_prob:drop_probs} )
        x_elbo = x_elbo[0]
        x_mean_elbo_valid = np.mean(x_elbo)
        x_std_elbo_valid = np.std(x_elbo)
        x_median_elbo_valid = np.median(x_elbo)
        # print stats
        ph.report_stats("\nValidation Statistics - Normal Data\n",x_elbo, verbose=verbose)
        # calculate ELBO statistics on the anomaly training set
        a_elbo = sess.run([ELBO_t], feed_dict={ X: anom_valid.reshape([-1,X_dim]), aux:l_anom_valid, is_train:False,drop_prob:drop_probs} )
        a_elbo = a_elbo[0]
        a_mean_elbo_valid = np.mean(a_elbo)
        a_std_elbo_valid = np.std(a_elbo)
        a_median_elbo_valid = np.median(a_elbo)
        ph.report_stats("\nValidation Statistics - Anomaly Data\n",a_elbo, verbose=verbose)
        # set threshold value
        elbo_dict = {'n_mean':x_mean_elbo_valid, 'n_std':x_std_elbo_valid,
                     'a_mean':a_mean_elbo_valid, 'a_std':a_std_elbo_valid }

        if a_mean_elbo_valid > x_mean_elbo_valid:
            #anomaly_threshold = ph.select_threshold(elbo_dict, criterion)
            anomaly_threshold,_ = ph.threshold_selector(x_elbo,a_elbo,1,fn_ratio)
            print(anomaly_threshold)

        else:
            print('Training error! Anomaly loss smaller than normal loss! Aborting...')
            anomaly_threshold = ph.select_threshold(elbo_dict, criterion)
        # plot histograms
        ph.elbo_hist(x_elbo, a_elbo, anomaly_threshold, 'Validation Set','',
                  save_print_dir, dataset, plots=plots)
        #=================== Evaluation on Test Set ==========================#
        # normalize data
        x_test = ph.p_normalize(x_test,train_mean,train_std)
        anom_test = ph.p_normalize(anom_test,train_mean,train_std)
        print(x_test.shape)
        print(anom_test.shape)
        if verbose:
            print("#=========================Test Set=============================#")
        print('Anomaly threshold: {}', anomaly_threshold)
        tn,fn,fp,tp = [np.zeros([output_samples]) for i in range(4)]
        y_pred_n = np.zeros([test_size,output_samples])
        y_pred_a = np.zeros([anom_test_size, output_samples])
        start_time = time.time()
        for i in range(output_samples):
            # evaluate ELBO on the normal test-set
            x_elbo = sess.run([ELBO_t], feed_dict={ X: x_test, aux:l_norm_test, is_train:False,drop_prob:drop_probs} )
            x_elbo = x_elbo[0]
            # visualize normal test-set
            x_samples_normal = sess.run([x_mu_t], feed_dict={X: x_test, aux:l_norm_test, is_train:False,drop_prob:drop_probs})
            # evaluate ELBO on the anomaly test-set
            a_elbo = sess.run([ELBO_t], feed_dict={ X: anom_test.reshape([-1,X_dim]), aux:l_anom_test, is_train:False,drop_prob:drop_probs} )
            a_elbo = a_elbo[0]
            # visualize anomaly test-set
            x_samples_anomaly = sess.run([x_mu_t], feed_dict={ X: anom_test.reshape([-1,X_dim]), aux:l_anom_test, is_train:False,drop_prob:drop_probs} )
            # save accuracy rates
            tn[i] = np.sum(x_elbo < anomaly_threshold)
            fp[i] = np.sum(x_elbo > anomaly_threshold)
            tp[i] = np.sum(a_elbo > anomaly_threshold)
            fn[i] = np.sum(a_elbo < anomaly_threshold)
            y_pred_n[:,i] = x_elbo > anomaly_threshold
            y_pred_a[:,i] = a_elbo > anomaly_threshold
        end_time = time.time()
        y_pred = np.concatenate((y_pred_n,y_pred_a),axis=0)
        y_true = np.concatenate((np.zeros([test_size]),np.ones([anom_test_size])), axis=0)
        # visualize operations

        x_class1_class1 = sess.run([x_mu_t],
                                    feed_dict={X: x_test[1].reshape([-1, X_dim]), aux: l_norm_test[1].reshape([-1, X_dim]),
                                               is_train: False, drop_prob: drop_probs})
        x_class1_class2 = sess.run([x_mu_t],
                                    feed_dict={X: x_test[1].reshape([-1,X_dim]), aux: l_norm_test[0].reshape([-1, X_dim]),
                                               is_train: False, drop_prob: drop_probs})
        x_class2_class1 = sess.run([x_mu_t],
                                    feed_dict={X: x_test[0].reshape([-1,X_dim]), aux: l_norm_test[1].reshape([-1, X_dim]),
                                               is_train: False, drop_prob: drop_probs})
        x_class2_class2 = sess.run([x_mu_t],
                                    feed_dict={X: x_test[0].reshape([-1,X_dim]), aux: l_norm_test[0].reshape([-1, X_dim]),
                                               is_train: False, drop_prob: drop_probs})
        x_anom_class1 = sess.run([x_mu_t],
                                    feed_dict={X: anom_test[0].reshape([-1,X_dim]), aux: l_norm_test[1].reshape([-1, X_dim]),
                                               is_train: False, drop_prob: drop_probs})
        x_anom_class2 = sess.run([x_mu_t],
                                    feed_dict={X: anom_test[0].reshape([-1,X_dim]), aux: l_norm_test[0].reshape([-1, X_dim]),
                                               is_train: False, drop_prob: drop_probs})

        sample_dict = {'nc1c1':np.squeeze(x_class1_class1),'nc1c2':np.squeeze(x_class1_class2),'nc2c1':np.squeeze(x_class2_class1),
                       'nc2c2':np.squeeze(x_class2_class2),'ac1':np.squeeze(x_anom_class1),'ac2':np.squeeze(x_anom_class2)}
        # calculate and report total stats
        # scores
        precision = tp.mean() / (tp.mean() + fp.mean())
        recall = tp.mean() / (tp.mean() + fn.mean())
        f1 = 2 * precision * recall / (precision + recall)
        f01 = (1 + (1 / fn_ratio) ** 2) * precision * recall / ((1 / fn_ratio ** 2) * precision + recall)
        ph.report_stats("\nTest Statistics - Normal Data", x_elbo, verbose=verbose)
        ph.report_stats("\nTest Statistics - Anomaly Data", a_elbo, verbose=verbose)
        ph.scores_x(tp,fp,fn,tn, verbose=verbose)
        y_out = ph.uncertainty(y_true,y_pred, verbose=verbose)
        # plot histograms
        ph.elbo_hist(x_elbo, a_elbo, anomaly_threshold, 'Test Set','',
                  save_print_dir, dataset, plots=plots)
        # Compute confusion matrix
        cnf_matrix = np.array([[int(tp.mean()),int(fn.mean())],
                               [int(fp.mean()),int(tn.mean())]])
        np.set_printoptions(precision=2)
        class_names = np.array(['Anomaly','Normal'],dtype='<U10')
        # Plot non-normalized confusion matrix
        if plots:

            ph.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                    title=" Confusion matrix\n"+"Dataset: "+dataset, plots=plots)
            plt.savefig(save_print_dir+dataset+"_split_"+str(split_ratio)+'conf_mat.png')

        if verbose:
            print('total inference time for {} data points (x{} samples each):{:6.3}s '.format(y_true.shape[0],\
                                                                    output_samples,(end_time-start_time)))

        # Save results to file
        ph.save2txt(save_print_dir, file_name, dataset, run_comment, h_dim, z_dim, learning_rate, batch_size, l2_reg,
                 save_epochs, early_stopping, X_dim, train_size, x_valid.shape[0], test_size, save_dir)
        ph.saveres2txt(save_print_dir, file_name, dataset, tn, fn, tp, fp, f1, f01, precision, recall, learning_rate)



    return [tp.mean(), fp.mean(), tn.mean(), fn.mean()],x_samples_normal[0],x_samples_anomaly[0], x_test, anom_test, \
           l_norm_test, l_anom_test, sample_dict
          # x_samples_normal_class1,x_samples_normal_class2,x_samples_anom_class1,x_samples_anom_class2,\


if __name__ == "__main__":
    conf_matrix,xn,xa,xntest,xatest,lnt,lat, sd= \
        net(TRAIN_VAE = False, load_model = True, real_data= False, batch_norm=False,
                                             dropout=True, verbose=False, plots=True) #xnc1,xnc2,xac1,xac2
    # one can use xn (prediction for x_test_normal) and xa (prediction for x_test_anomalous) for visualisation

    fig, ax = plt.subplots(10)#xn.shape[0])
    ax[0].set_title('Normal_Predictions')
    for i in range(10):
        ax[i].plot(xn[i, :], label=str(i))
    plt.show()
    ax[0].set_title('Normal_True')
    for i in range(10):
        ax[i].plot(xntest[i, :], label=str(i))
    plt.legend(('Predicted', 'True'), loc='best')
    plt.show()
    fig, ax = plt.subplots(10)#xn.shape[0])
    ax[0].set_title('Anomaly__Predictions')
    for i in range(10):
        ax[i].plot(xa[i, :], label=str(i))
    plt.show()
    ax[0].set_title('Anomaly__True')
    for i in range(10):
        ax[i].plot(xatest[i, :], label=str(i))
    plt.legend(('Predicted', 'True'), loc='best')
    plt.show()
    ### Preapere plots for report xntest[0] is class 2, xntest[0] is class 1
    fig, ax = plt.subplots(2)#xn.shape[0])
    ax[0].set_title('True: class 1 - Assumed: class 1\n',fontsize=16)
    ax[0].plot(sd['nc1c1'], 'r',label='predicted')
    ax[0].plot(xntest[1, :], label='true')
    ax[0].set_ylabel('Arbitrary units', fontsize=14)
    ax[0].legend(loc='best',fontsize=16)
    ax[1].set_title('\nTrue: class 1 - Assumed: class 2\n',fontsize=16)
    ax[1].plot(sd['nc1c2'], 'r',label='predicted')
    ax[1].plot(xntest[1, :], label='true')
    ax[1].set_xlabel('Time (arbitrary units)', fontsize=14)
    ax[1].set_ylabel('Arbitrary units', fontsize=14)
    ax[1].legend(loc='best', fontsize=16)
    plt.tight_layout()
    fig, ax = plt.subplots(2)#xn.shape[0])
    ax[0].set_title('True: class 2 - Assumed: class 1\n',fontsize=16)
    ax[0].plot(sd['nc2c1'], 'r',label='predicted')
    ax[0].plot(xntest[0, :], label='true')
    ax[0].set_ylabel('Arbitrary units', fontsize=14)
    ax[0].legend(loc='best',fontsize=16)
    ax[1].set_title('\nTrue: class 2 - Assumed: class 2\n',fontsize=16)
    ax[1].plot(sd['nc2c2'], 'r',label='predicted')
    ax[1].plot(xntest[0, :], label='true')
    ax[1].set_xlabel('Time (arbitrary units)', fontsize=14)
    ax[1].set_ylabel('Arbitrary units', fontsize=14)
    ax[1].legend(loc='best', fontsize=16)
    plt.tight_layout()
    fig, ax = plt.subplots(2)#xn.shape[0])
    ax[0].set_title('True: anomalous class - Assumed: class 1\n',fontsize=16)
    ax[0].plot(sd['ac1'], 'r',label='predicted')
    ax[0].plot(xatest[0, :], label='true')
    ax[0].set_ylabel('Arbitrary units', fontsize=14)
    ax[0].legend(loc='best',fontsize=16)
    ax[1].set_title('\nTrue: anomalous class - Assumed: class 2\n',fontsize=16)
    ax[1].plot(sd['ac2'], 'r',label='predicted')
    ax[1].plot(xatest[0, :], label='true')
    ax[1].set_xlabel('Time (arbitrary units)', fontsize=14)
    ax[1].set_ylabel('Arbitrary units', fontsize=14)
    ax[1].legend(loc='best', fontsize=16)
    plt.tight_layout()



