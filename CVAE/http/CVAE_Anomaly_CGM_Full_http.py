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

**** Version 06:
    - Implemented the  CVAE - CGM_Full (Conditional Generative Model)
                                   from K.Sohn et al (CVAE)
    - Implementing the KL of two diagonal Gaussian distributions
      and therefore introducing a second inference network (the prior)
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
from skimage.data import camera
from skimage.filters import roberts
from skimage import transform as transf
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

###############################################################################
##########                      HELPER FUNCTIONS
###############################################################################

def shuffle_data(x,y):
    # randomly shuffle data
    new_indices = np.random.permutation(x.shape[0])
    shuffled_x = x[new_indices,:]
    shuffled_y = y[new_indices,:]
    return shuffled_x, shuffled_y  

def plot_2hist(var1, var2, print_dir, dataset, title='Histogram', f_name="" ,figsize=(5, 5), plots=True):
    # Print a histogram of the calculated ELBO values
    if plots:
        lim = max(max(var1),max(var2))
        plt.figure(figsize=figsize)

        plt.subplot(211)
        axes = plt.gca()
        sns.distplot(var1, kde=False, norm_hist=False, bins=30, color='green', label='true normal')
        plt.ylabel('Counts', fontsize=16)
        axes.set_xlim([-0.1, lim])
        plt.legend()
        plt.title("$\mathcal{L}_{ELBO}$ Histograms\n" + "Assuming {} class\n".format(title), fontsize=16)
        plt.tight_layout()
        plt.subplot(212)
        axes = plt.gca()
        sns.distplot(var2, kde=False, bins=30, norm_hist=False,color='red', label='true anomalies')
        plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=16)
        plt.ylabel('Counts', fontsize=16)
        axes.set_xlim([-0.1, lim*1.1])
        plt.legend()

        plt.tight_layout()
        if f_name != "":
            plt.savefig(print_dir + dataset + f_name)
        plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          f_name="", plots=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if plots:
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title,size=18)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45,size=16)
        plt.yticks(tick_marks, classes,size=16)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        #    thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center", color="black",size=18)
        # color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label',size= 16)
        plt.xlabel('Predicted label',size= 16)
        if f_name != "":
            plt.savefig(save_print_dir + dataset + f_name)
        plt.show()


def plot_hist(var,title='Histogram', f_name="", figsize=(5,5)):
    # Print a histogram of the weights of a layer
    # to inspect for weight saturation
    plt.figure(figsize = figsize)
    plt.hist(var,bins=20)
    plt.title("Dataset: {}, {}".format(dataset,title))
    plt.xlabel('Loss')
    plt.ylabel('Counts')
    plt.tight_layout()
    if f_name!="" :
        plt.savefig(save_print_dir+dataset+f_name)
    plt.show()

def save2txt(save_print_dir ,file_name, dataset, run_comment, h_dim, z_dim, learning_rate, batch_size, l2_reg,
             save_epochs, early_stopping,x_dim,train_size,valid_size,test_size,save_dir):
    '''
    Helper function for saving parameters and hyperparameters
    '''
    res_file_name = save_dir+file_name[:-4]+'res'+'.npz'
    file_name = save_print_dir +file_name + '.txt'
    mf = open(file_name, 'w')
    dt = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    mf.write("File Name:\t{}\nDate:\t{}\n".format(file_name, dt))
    #mf.write("----------------------------------------------------\n".format(dataset, run_comment))
    mf.write("DataSet:\t{}\n".format(dataset))
    #mf.write("----------------------------------------------------\n".format(dataset, run_comment))
    mf.write("h Dim.:\t{}\tz Dim.:\t{}\tx dim:\t{}\n".format(h_dim, z_dim, x_dim))
    mf.write("Learning rate:\t{}\tBatch size:\t{}\tL2 reg:\t{}\n".format(learning_rate,batch_size,l2_reg))
    mf.write("N_epochs:\t{}\tEarly Stopping:\t{}\n".format(save_epochs,early_stopping))
    mf.write("Train size:\t{}\tValid size:\t{}\tTest size:\t{}\n".format(train_size,valid_size,test_size))
    mf.write("np train file:\t{}\n".format(file_name[:-4] + '.npz'))
    mf.write("np result file:\t{}\n\n".format(res_file_name))
    mf.close()
    return

def saveres2txt(save_print_dir, file_name, dataset,tp,fp,tn,fn,f1,f01,precision,recall,learning_rate=0):
    '''
    Helper function for saving results
    '''
    file_name = save_print_dir + file_name + '.txt'
    mf = open(file_name, 'a')
    dt = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    mf.write("\n-------------------------------  Results ---------------------------------\n")
    mf.write("DataSet:\t{}\tlearning rate:\t{}\n".format(dataset, learning_rate))
    mf.write("tp:\t{}\tfp:\t{}\tfn:\t{}\ttn:\t{}\n".format(tp,fp,fn,tn))
    mf.write("f1-score:\t{}\tf0.1-score:\t{}\n".format(f1,f01))
    mf.write("Precision:\t{}\tRecall:\t{}\n".format(precision,recall))
    mf.close()
    return

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 0.1 / tf.sqrt(in_dim / 2.)
    return tf.truncated_normal(shape=size, stddev=xavier_stddev)

def normalize(x):
    """
    Function that normalizes a dataset
    Input:   x = dataset
    Returns: x_mean = row vector of mean per column
             x_std  = row vector of std per column
             x_norm = normalized data set
    """
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    return x_mean, x_std, (x-x_mean)/x_std

def p_normalize(x, x_mean, x_std):
    """
    Function that normalizes data-points
    Input:   x = dataset
             x_mean = row vector of mean per column
             x_std  = row vector of std per column
    Returns: 
             x_norm = normalized data set
    """

    return (x-x_mean)/x_std

def denormalize(x, x_mean, x_std):
    """
    Function that undoes the normalization
    Input:   x = normalized dataset
             x_mean = row vector of mean per column
             x_std  = row vector of std per column
    Returns: unnormalized data set
    """
    return (x*x_std)+x_mean
    
def report_stats(title, res):
    print(title)
    print("min_elbo: {}".format(res.min()))
    print("Mean elbo: {}".format(res.mean()))
    print("max_elbo: {}".format(res.max()))
    print("Std elbo: {}".format(res.std()))
    print("Median elbo: {}".format(np.median(res)))
    return

def progress(epoch,eloss,etime):
    print('Epoch: {}'.format(epoch))
    print('Loss: {:.4}'. format(eloss))
    print("time: {}".format(etime))
    return

###############################################################################
# ======================= file and directory names ========================== #
###############################################################################
run_comment = "CVAE_CGM_Full"
dataset = "http"
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
##########                    Import Data - Transform                ########## 
###############################################################################
TRAIN_VAE = False
load_model =  True
epochs  = 21
# load data
mydata = np.load(dataset+"_normal_anomaly_mixed.npz")
# Split into training, validation and test sets
x_train= mydata['x_train']
x_valid= mydata['x_valid']
x_test= mydata['x_test']
# get labels
y_train= mydata['y_train']
y_valid= mydata['y_valid']
y_test= mydata['y_test']
# train/test
#split = int(anomalies_size//split_ratio)
##  y = [1 0] -> normal
##  y = [0 1] -> anomaly
# get sizes
train_size = x_train.shape[0]
validation_size = x_valid.shape[0]
test_size = x_test.shape[0]
# data dimensionality
X_dim = x_train.shape[1]
y_dim = y_train.shape[1]
# latent space size
z_dim = 10
# hidden layer size
h_dim = 128
# number of samples for the evaluation of likelihood
n_samples = 5
# run parameters
batch_size = 768
num_batches = train_size//batch_size
# learning rate
lr = 3e-4
# hyperparameter for optimisation objectives
alpha = 1
anneal_rate = 5e-4
# regularization constant
gamma = 1e-4
epoch = 0
save_epochs = 0
early_stopping = 50
fn_ratio = 10
# run parameters
#a_split = str(np.round(1-split_ratio,decimals=3)).replace(".","_")
file_name = run_comment+'_'+dataset+"_h_dim_"+str(h_dim)+"_zDim_"+str(z_dim)\
         # + "_split_ratio"+a_split
load_file_name = file_name

# Training losses containers
best_eval_loss = np.inf
training_loss = []
validation_loss= []

# initiallize regularizer and graph
regularize = tf.contrib.layers.l2_regularizer(gamma, scope=None) 
graph = tf.Graph()
with graph.as_default():

# ============================= Placeholders ================================ # 
   
    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, shape=[None, X_dim],name="input_x")
        y = tf.placeholder(tf.float32, shape=[None, y_dim],name="label_y")
        alpha_KL = tf.placeholder_with_default(input=1.0, shape=(), name='KL_annealing')
    with tf.name_scope("latent"):
        z = tf.placeholder(tf.float32, shape=[None, z_dim],name="latent_vars")
        
# =============================== Q(z|X,y) ==================================== #

    with tf.variable_scope("encoder"):
        Q_W1 = tf.get_variable("Q_W1", initializer = xavier_init([X_dim + y_dim, h_dim]),
                           regularizer = regularize)
        Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

        Q_W2_mu = tf.get_variable("Q_W2_mu", initializer =  xavier_init([h_dim, z_dim]),
                           regularizer = regularize)
        Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

        Q_W2_sigma = tf.get_variable("Q_W2_sigma", initializer = xavier_init([h_dim, z_dim]),
                           regularizer = regularize)
        Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))

    def Qz(X,y):
        #with tf.variable_scope("encoder"):
        inputs = tf.concat( axis=1, values=[X, y] )
        h = tf.nn.relu(tf.matmul(inputs, Q_W1) + Q_b1)
        z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
        z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
        return z_mu, z_logvar

# =============================== P(z|X) ==================================== #

    with tf.variable_scope("encoder", reuse = None):
        Qp_W1 = tf.get_variable("Qp_W1", initializer=xavier_init([X_dim, h_dim]),
                               regularizer=regularize)
        Qp_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

        Qp_W2_mu = tf.get_variable("Qp_W2_mu", initializer=xavier_init([h_dim, z_dim]),
                                  regularizer=regularize)
        Qp_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

        Qp_W2_sigma = tf.get_variable("Qp_W2_sigma", initializer=xavier_init([h_dim, z_dim]),
                                     regularizer=regularize)
        Qp_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))

    def Pz(X):
        # with tf.variable_scope("encoder"):
        inputs = X #tf.concat(axis=1, values=[X, y])
        h = tf.nn.relu(tf.matmul(inputs, Qp_W1) + Qp_b1)
        z_mu = tf.matmul(h, Qp_W2_mu) + Qp_b2_mu
        z_logvar = tf.matmul(h, Qp_W2_sigma) + Qp_b2_sigma
        return z_mu, z_logvar

    # =============================== Sampler ====================================

    def sample_z(mu, log_var):
        #with tf.name_scope("latent"):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps
    
    # =============================== P(y|z,x) ====================================
    
    with tf.variable_scope("decoder"):
        
        P_W1 = tf.get_variable("P_W1", initializer = xavier_init([z_dim + X_dim, h_dim]),
                               regularizer = regularize)
        P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
      
        P_W2 = tf.get_variable("P_W2",initializer =xavier_init([h_dim, y_dim]),
                               regularizer = regularize)
        P_b2 = tf.Variable(tf.zeros(shape=[y_dim]))
    
    
    def Px(z,x):
       # with tf.variable_scope("decoder"):
        inputs = tf.concat(axis=1, values=[z, x])
        h = tf.nn.relu(tf.matmul(inputs, P_W1) + P_b1)
#                   tf.nn.sigmoid(tf.matmul(y, P_W_s) + P_b_s))
        logits = tf.matmul(h, P_W2) + P_b2
        #prob = tf.nn.sigmoid(logits)
        return logits
    
    
    # =============================== TRAINING ====================================
    #logits_ = np.zeros((batch_size, n_samples,z_dim))
    # encode CVAE
    with tf.variable_scope("encoder",reuse = True):
        # get the parameters of the posterior
        z_mu, z_logvar = Qz(X,y)
        # get the parameters of the prior network
        zp_mu, zp_logvar = Pz(X)
    with tf.name_scope("latent"):
        z_sample1 = sample_z(z_mu, z_logvar)
    with tf.variable_scope("decoder"):
        logits_1 = Px(z_sample1, X)
    with tf.name_scope("latent"):
        z_sample2 = sample_z(z_mu, z_logvar)
    with tf.variable_scope("decoder",reuse = True):
        logits_2 = Px(z_sample2, X)
    with tf.name_scope("latent"):
        z_sample3 = sample_z(z_mu, z_logvar)
    with tf.variable_scope("decoder",reuse = True):
        logits_3 = Px(z_sample3, X)
    with tf.name_scope("latent"):
        z_sample4 = sample_z(z_mu, z_logvar)
    with tf.variable_scope("decoder",reuse = True):
        logits_4 = Px(z_sample4, X)
    with tf.name_scope("latent"):
        z_sample5 = sample_z(z_mu, z_logvar)
    with tf.variable_scope("decoder",reuse = True):
        logits_5 = Px(z_sample5, X)
    with tf.name_scope("loss"):
        # E[log P(X|z)]
        L_1 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_1)
        L_2 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_2)
        L_3 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_3)
        L_4 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_4)
        L_5 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_5)
        # reconstruction loss
        recon_loss = (L_1+L_2+L_3+L_4+L_5)/5
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl_loss = 0.5 * tf.reduce_sum((tf.exp(z_logvar)/tf.exp(zp_logvar))\
                                      + (z_mu-zp_mu)**2 - 1. - z_logvar + zp_logvar, 1)
        # Regularisation cost
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # VAE loss
        vae_loss = recon_loss + alpha_KL * kl_loss
        vae_loss = tf.add_n([tf.reduce_mean(vae_loss, axis=0) + tf.reduce_sum(reg_variables)])
        # summary
    with tf.name_scope("Summaries"):
        #tf.summary.scalar("ELBO",ELBO)
        tf.summary.scalar("Vae_loss",vae_loss)
        merger = tf.summary.merge_all()
    with tf.name_scope("optimiser"):
        # updater
        solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(vae_loss) 
    
    # =============================== TESTING =================================== #
    with tf.name_scope("Testing"):
        z_mu_t, z_logvar_t = Qz(X,y)
        zp_mu_t, zp_logvar_t = Pz(X)
        z_sample_t1 = sample_z(z_mu_t, z_logvar_t)
        logits_t1 = Px(z_sample_t1, X)
        z_sample_t2 = sample_z(z_mu_t, z_logvar_t)
        logits_t2 = Px(z_sample_t2, X)        
        z_sample_t3 = sample_z(z_mu_t, z_logvar_t)
        logits_t3 = Px(z_sample_t3, X)
        z_sample_t4 = sample_z(z_mu_t, z_logvar_t)
        logits_t4 = Px(z_sample_t4, X)
        z_sample_t5 = sample_z(z_mu_t, z_logvar_t)
        logits_t5 = Px(z_sample_t5, X)
        # E[log P(X|z)]
        L_t1 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_t1)
        L_t2 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_t2)
        L_t3 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_t3)
        L_t4 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_t4)
        L_t5 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_t5)
        # reconstruction loss
        L_t = (L_t1+L_t2+L_t3+L_t4+L_t5)/5
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl_loss_t = 0.5 * tf.reduce_sum((tf.exp(z_logvar_t)/tf.exp(zp_logvar_t))\
                                      + (z_mu_t-zp_mu_t)**2 - 1. - z_logvar_t + zp_logvar_t, 1)
        # VAE loss
        ELBO_t = L_t + kl_loss_t
    
    # =============================== Session =============================== #
    # normalize data
    train_mean, train_std, x_train = normalize(x_train)
    x_valid = p_normalize(x_valid,train_mean, train_std)
    x_test = p_normalize(x_test, train_mean, train_std)
    # train
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

        print(" -----Training Started-----")
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_time = time.time()
            x_train,y_train = shuffle_data(x_train,y_train)
            count = 0
            # For each epoch train with mini-batches of size (batch_size) 
            for batch_num in range(train_size//batch_size):
                X_mb = x_train[batch_num*batch_size:(batch_num+1)*batch_size,]
                y_mb = y_train[batch_num*batch_size:(batch_num+1)*batch_size,]
                kl_a = 2 * (-.5 + 1 / (1 + np.exp(-count * anneal_rate)))
                # train
                _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb, y:y_mb,alpha_KL:kl_a})
                epoch_loss+=loss
                count+=1
            # print progress
            progress(epoch,epoch_loss/num_batches, time.time()-epoch_time)
            training_loss.append(epoch_loss/num_batches)
            if epoch >0 and epoch%1 ==0:
                vloss = 0
                vloss, vaeloss = sess.run([vae_loss,merger], feed_dict={ X: x_valid, y: y_valid} )
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
                    + ", z: "+str(z_dim)+", Lr: "+str(lr)+ ", L2: "+str(gamma)\
                    + ", batch_size: " + str(batch_size)+", epochs"+str(save_epochs)   
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
                tf.train.Saver().restore(sess,save_dir+load_file_name)
                print("Done!\n")
            except Exception:
                print("Could not find saved file. Training from scratch")

    # calculate ELBO statistics on the validation set
    x_elbo = sess.run([ELBO_t], feed_dict={ X: x_valid, y:y_valid} )
    x_elbo = x_elbo[0]
    # plot_hist(x_elbo,title='ELBO distribution\n validation set',\
    #           f_name="train_normal", figsize=(8,5))
    x_mean_elbo_valid = np.mean(x_elbo)
    x_std_elbo_valid = np.std(x_elbo)
    x_median_elbo_valid = np.median(x_elbo)
    # plt.show()
    # print stats
    report_stats("\nValidation Statistics \n",x_elbo)
    
    #=================== Evaluation on Test Set ==========================#
    
    print("#============================================================#")
    # Predict label based on normal assumption, i.e. yi = [1 0]
    y_test = y_test.astype(dtype=np.bool)
    y_norm = np.ones_like(y_test)
    y_norm[:,1] = 0
    x_elbo_norm = sess.run([ELBO_t], feed_dict={ X: x_test, y: y_norm} )
    x_elbo_norm = x_elbo_norm[0]
    # Predict label based on anomaly assumption, i.e. yi = [0 1]
    y_anom = np.ones_like(y_test)
    y_anom[:,0] = 0
    x_elbo_anom = sess.run([ELBO_t], feed_dict={ X: x_test, y: y_anom} )
    x_elbo_anom = x_elbo_anom[0]
    # concatenate the two ELBO vectors
    x_conc = np.concatenate((x_elbo_norm.reshape([-1,1]), x_elbo_anom.reshape([-1,1])),axis=1)
    # get predicted label by selecting the column with the smallest elbo
    y_pred = np.zeros_like(y_test).astype(dtype=np.bool)
    y_pred[:,0] = [True if a<b else False for a,b in x_conc]
    y_pred[:,1] = [True if a>b else False for a,b in x_conc]
    # compare with base truth 
    # p = positive = normal, n = negative = anomaly
    tp = sum(y_test[:,0] & y_pred[:,0])
    tn = sum(y_test[:,1] & y_pred[:,1])
    fp = sum(y_test[:,1] & y_pred[:,0])
    fn = sum(y_test[:,0] & y_pred[:,1])
    # score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    f01 = (1+(1/fn_ratio)**2) * precision * recall / ((1/fn_ratio**2)*precision + recall)
    print("#=====================================#")
    print("True negatives:{}".format(tn))
    print("False positives:{}".format(fp))
    print("Precision:{}".format(precision))
    print("Recall:{}".format(recall))
    print("f1-score:{}".format(f1))
    print("f01-score:{}".format(f01))
    # Compute confusion matrix
    cnf_matrix = np.array([[tn,fp],
                           [fn,tp]])
    np.set_printoptions(precision=1)
    class_names = np.array(['Anomaly','Normal'],dtype='<U10')
    # Plot non-normalized confusion matrix

    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization',
                          f_name=file_name)
    #
    # plot_hist(x_elbo_norm,title='ELBO distribution assuming normal label \n test set',\
    #       f_name="train_normal", figsize=(8,5))
    #
    # plot_hist(x_elbo_anom,title='ELBO distribution assuming anomaly label \n test set',\
    #       f_name="train_normal", figsize=(8,5))
    x_elbo_norm_norm = x_elbo_norm[y_test[:,1]==0]  ;x_elbo_norm_anom = x_elbo_norm[y_test[:,1]==1]
    x_elbo_anom_norm = x_elbo_anom[y_test[:,1]==0]  ;x_elbo_anom_anom = x_elbo_anom[y_test[:,1]==1]

    plot_2hist(x_elbo_norm_norm, x_elbo_norm_anom, save_print_dir, dataset, title='normal',
               f_name="elbo_normal" ,figsize=(5, 5), plots=True)
    plot_2hist(x_elbo_anom_norm, x_elbo_anom_anom, save_print_dir, dataset, title='anomalous',
               f_name="elbo_anomaly", figsize=(5, 5), plots=True)
    # Save results to file
    save2txt(save_print_dir, file_name, dataset, run_comment, h_dim, z_dim, lr, batch_size, gamma,
             save_epochs, early_stopping, X_dim, train_size, x_valid.shape[0], test_size, save_dir)
    saveres2txt(save_print_dir, file_name, dataset, tn, fn, tp, fp, f1, f01, precision, recall, lr)

