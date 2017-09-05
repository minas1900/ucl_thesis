# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 01:37:15 2017

@author: Heraclitus

###############################################################################
                              TENSORBOARD
tensorboard --logdir=./logs                          
,or, 
python -m tensorflow.tensorboard

Then: localhost:6006

"""
# General Imports
import numpy as np
import os.path
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
# Command Line Interface Imports
import argparse
import sys

###############################################################################
#   0. Parameter Initialization  - Data Import                                #
###############################################################################
# Set random generator seed for reproducibility
np.random.seed(128)
# Import MNIST example data 
from tensorflow.examples.tutorials.mnist import input_data

###############################################################################
#   1. Class definition                                                       #
###############################################################################

# CLass Definitions

class VAE(object):
    """ Generic class for Variational Autoencoder
    """
    def __init__( self, params):
                 #sess) :
        """
        'params' variable contains initialization values for:
            input_size,
            latent_size,
            p_x = 'Categorical',
            p_z = 'Gaussian',
            q_z = 'Gaussian',
            he1_size = 500,
            hd1_size = 500,
            dropout_prob = 0.0,
            batch_norm = False,
            l2_reg = 0.0,
            trainable = True,
            model_dir = None
        """
        # Initialize variables
        self.input_size = params['input_size']
        self.latent_size = params['latent_size']
        self.he1_size = params['h1_size']
        self.hd1_size = params['h2_size']
        self.dropout_prob = params['dropout_prob']
        self.batch_norm = params['batch_norm']
        if params['l2_reg'] > 0.0:
            self.l2_reg = tf.contrib.layers.l2_regularizer(params['l2_reg'])
        else:
            self.l2_reg=None 
        self.p_x = params['p_x']
        self.q_z = params['q_z']
        self.p_z = params['p_z']
        self.trainable = True
        self.batch_size = params['batch_size']
        #self.sess = sess
        # Build model
        self.initializer = tf.contrib.layers.xavier_initializer(
            uniform=False, seed=None , dtype = tf.float32 )  
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Define placeholders
            self.x = tf.placeholder(tf.float32,shape=[None,self.input_size],name="x")
            self.z = tf.placeholder(tf.float32,shape=[None,self.latent_size],name="z")
            if self.batch_norm :
                self.train_mode = tf.placeholder(tf.bool, name="train_mode")
            self.define_vars()
            self.make_model()
            #self.summarizer()
            self.saver = tf.train.Saver()
            self.session = tf.Session()
        
    def define_vars(self):
        """Define model variables - use scoping for easier reference in 
        Tensorboard
        """
        # ENCODER
        self.global_step = tf.Variable(0, trainable = False, name="SGD_counter0",\
                                       dtype=tf.int32 ) 
        with tf.variable_scope("encoder_weights") as scope:
            self.we_1_h = tf.get_variable("we_1_h",
                                       shape = [self.input_size, self.he1_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )
            self.be_1_h = tf.get_variable("be_1_h",
                                       shape = [self.he1_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )            
            self.we_2_sigma2 = tf.get_variable("we_2_var",
                                       shape = [self.he1_size, self.latent_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )
            self.be_2_sigma2 = tf.get_variable("be_2_var",
                                       shape = [self.latent_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )  
            self.we_3_mu = tf.get_variable("we_3_mu",
                                       shape = [self.he1_size, self.latent_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )
            self.be_3_mu = tf.get_variable("be_3_mu",
                                       shape = [self.latent_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )              
        # DECODER       
        with tf.variable_scope("decoder_weights") as scope:
            self.wd_1_h = tf.get_variable("wd_1_h",
                                       shape = [self.latent_size, self.hd1_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )
            self.bd_1_h = tf.get_variable("bd_1_h",
                                       shape = [self.hd1_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )            
            self.wd_2_o = tf.get_variable("wd_2_o",
                                       shape = [self.hd1_size, self.input_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )
            self.bd_2_o = tf.get_variable("bd_2_o",
                                       shape = [self.input_size],
                                       dtype=tf.float32, 
                                       trainable= self.trainable,
                                       initializer = self.initializer,
                                       regularizer = self.l2_reg )  
    def sampler(self, mu,sigma):
        """
        sample z
        """
        with tf.name_scope("sampler") as scope:
            z = tf.add(tf.multiply(sigma,tf.random_normal(\
                        shape = [self.batch_size, self.latent_size],\
                        mean=0.0, stddev=1.0), name="x_sigma"),mu, name = "plus_mu_sample_z")
        return z
        
    def encoder(self,x):
        """Build the encoder
        """
        with tf.name_scope("encoder"):  #,reuse = False
            # nodes
            #scope.reuse_variables()
            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.train_mode, )
            he1 = tf.add(tf.matmul(x,self.we_1_h,name="x_h1_weights"),\
                                    self.be_1_h, name="plus_bias")
            if self.batch_norm:
                he1 = tf.layers.batch_normalization(he1, training=tf.cast(self.train_mode,tf.bool))         
            he1 = tf.nn.relu(he1,name="hidden")
            
            log_var = tf.add(tf.matmul(he1,self.we_2_sigma2,name="x_var_weights"),\
                                    self.be_2_sigma2,name="logsigma2")
            if self.batch_norm:
                log_var = tf.layers.batch_normalization(log_var, training=tf.cast(self.train_mode,tf.bool))               
            
            mu = tf.add(tf.matmul(he1,self.we_3_mu,name="x_mu_weights"),\
                        self.be_3_mu, name = "plus_mu")
            if self.batch_norm:
                mu = tf.layers.batch_normalization(mu, training=tf.cast(self.train_mode,tf.bool))
                
            sigma = tf.exp(0.5*log_var,name="sigma2")
            # prepare summaries
        with tf.name_scope("Summaries"):
            tf.summary.histogram("w_hidden",self.we_1_h)
            tf.summary.histogram("b_hidden",self.be_1_h)
            tf.summary.histogram("w_sigma2",self.we_2_sigma2)
            tf.summary.histogram("b_sigma2",self.be_2_sigma2)
            tf.summary.histogram("w_mu",self.we_3_mu)
            tf.summary.histogram("b_mu",self.be_3_mu)
            tf.summary.histogram("hidden_activations",he1)
        return log_var, sigma, mu, self.sampler(mu,sigma)
               
    def decoder(self,z):
        """Build the decoder
        """
        with tf.name_scope("decoder") as scope:
            # nodes
            #scope.reuse_variables()
            hd1 = tf.nn.relu(tf.add(tf.matmul(z,self.wd_1_h,name="x_h1_weights"),\
                                    self.bd_1_h,name="plus_bias"),name="hidden")
            x_hat = tf.sigmoid(tf.add(tf.matmul(hd1,self.wd_2_o,name="x_weights"),\
                                      self.bd_2_o,name="plus_bias_logits") ,name="probits")
            with tf.name_scope("Summaries"):
                #with tf.variable_scope("encoder"):
                tf.summary.histogram("w_hidden",self.wd_1_h)
                tf.summary.histogram("b_hidden",self.bd_1_h)
                tf.summary.histogram("w_output",self.wd_2_o)
                tf.summary.histogram("b_output",self.bd_2_o)
                tf.summary.histogram("hidden_activations",hd1)
   
        return x_hat        
                 

    def make_model(self):
        """Building the model
        """
        # encode data
        with tf.variable_scope("encoder",reuse = False):
            log_var, sigma, mu, z = self.encoder(self.x)
        # decode latent variable
        with tf.variable_scope("decoder",reuse = False): 
            x_hat = self.decoder(z)
        # calculate optimisation objective
        with tf.name_scope("loss"): 
            # KL - loss (analytic calculation)
            with tf.name_scope("KL"):
                KL = 0.5 * tf.reduce_sum( 1 + log_var - sigma**2 - mu**2 , 1)
            # Estimate Eq(z/x)[logp(x|z)] - 1 sample estimate
            with tf.name_scope("xent"):    
                xent_loss = self.input_size * tf.contrib.keras.losses.binary_crossentropy(self.x, x_hat)            
            # Total loss
            self.loss = tf.reduce_mean(xent_loss - KL)
        # summary saver
        #with tf.name_scope("Summaries"):
            tf.summary.scalar('entropy', self.loss)
        self.merged_sums  = tf.summary.merge_all()
            
    def train_model(self, x_train, x_valid, num_epochs = 10, batch_size = 100,\
                    learning_rate= 1e-3, beta1=0.9, beta2=0.999,epsilon = 1e-8,\
                    l2_reg = 0.0, report_every = 1, train_mode = True,\
                    load_model = False, load_dir=None, save_dir=None,\
                    write_path=None, plot_samples = None):
        """
        Method for training the model
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        if l2_reg > 0.0:
            self.l2_reg = tf.contrib.layers.l2_regularizer(l2_reg)
        else:
            self.l2_reg=None  
#        if self.batch_norm:
#            self.train_mode = train_mode

        ''' Save Session and Summary '''
        #merged_sum  = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/", self.graph)
        
        if save_dir is None:
            if not os.path.exists("./saved_models/"):
                os.mkdir("./saved_models/")
            save_dir = 'saved_models/VAE_{}-{}_{}.cpkt'.format(
                    learning_rate,self.batch_size,time.time())
        # initialize graph
        with self.graph.as_default():
            # Optimiser
            self.optim = tf.train.AdamOptimizer( learning_rate = self.learning_rate,
                                                beta1 = self.beta1, beta2 = self.beta2)
            # Training Operator
            #self.train_op = self.optim.minimize(self.loss)
            self.train_op = tf.contrib.layers.optimize_loss( loss = self.loss,\
                            global_step = self.global_step, learning_rate = self.learning_rate,
                            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,\
                            beta1 = self.beta1, beta2 = self.beta2),\
                            clip_gradients= 1e3,\
                            #learning_rate_decay_fn = ,\
                            name ="advanced_training_operator", \
                            summaries=["learning_rate", "loss", "gradient_norm"])
            # extra dependencies for batch norm
            if self.batch_norm:
                self.batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # Initialiser
            init = tf.global_variables_initializer()

        # start training session
        with self.session as sess:
            sess.run(init)
            if load_model:
                print("Loading saved model")
                if load_dir is None: 
                        raise ValueError("Filename and path not supplied! Aborting...")
                else:
                    try:
                        self.saver.restore(sess,load_dir)
                        print("Done! Continuing training...\n")
                    except Exception:
                        print("Could not find saved file. Training from scratch")

            train_size = x_train.shape[0]
            valid_size = x_valid.shape[0]
            best_eval_loss = np.inf
            counter = 0
            # Start Training
            print("\nTraining Started")
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                # shuffle the data
                np.random.shuffle(x_train)
                #train_images,train_labels = shuffle_data(train_images,train_labels)
                #if you have to shuffle the labels also you must use a custom 
                #shuffling function
                # pick minibatches
                training_loss = 0.0
                for batch in range(train_size//batch_size):
                    if batch < train_size//batch_size:
                        x_batch = x_train[batch*batch_size:(batch+1)*batch_size,]
                    else:
                        x_batch = x_train[batch*batch_size:,]
                    # reshape for feeding into network

                    # Run optimisation
                    if self.batch_norm:
                        # Build feed dictionary
                        train_feed = {self.x: x_batch, self.train_mode: [True] }
                        _,batch_loss,_ = sess.run([self.train_op, self.loss,
                                                 self.batch_norm_ops],
                                               feed_dict = train_feed)
                    else:
                        # Build feed dictionary
                        train_feed = {self.x : x_batch}
                        _,batch_loss = sess.run([self.train_op, self.loss],
                                               feed_dict = train_feed)
                    training_loss += batch_loss
                epoch_end_time = time.time()
                if epoch % report_every == 0:
                    eval_loss = 0.0
                    for batch in range(valid_size//batch_size):
                        if batch < valid_size//batch_size:
                            x_batch = x_valid[batch*batch_size:(batch+1)*batch_size,]
                        else:   
                            x_batch = x_valid[batch*batch_size:,]
                        # reshape for feeding into network
                        
                        # Run optimisation
                        #print(self.loss.eval().shape)
                        if self.batch_norm:
                            # Build feed dictionary
                            valid_feed = {self.x : x_batch, self.train_mode: [False]}
                            batch_loss,lsummary,_ = sess.run([self.loss, self.merged_sums,\
                                                self.batch_norm_ops],\
                                                feed_dict = valid_feed)  
                        else: 
                            # Build feed dictionary
                            valid_feed = {self.x : x_batch}
                            batch_loss,lsummary = sess.run([self.loss, self.merged_sums],\
                                                feed_dict = valid_feed)                                 
                        eval_loss += batch_loss
                        writer.add_summary(lsummary, counter)
                        counter+=1
                    if eval_loss < best_eval_loss:
                        # update best result and save checkpoint
                        best_eval_loss = eval_loss
                        self.saver.save(sess, save_dir)
                        
                    print('------------------')
                    print('epoch: %d'%(epoch))
                    print('Epoch training time: {:6.2f}s'.format(epoch_end_time-epoch_start_time))
                    print('training loss: {:3.6f}'.format(self.batch_size*training_loss/train_size))
                    print('validation set loss: {:3.6f}'.format(self.batch_size*eval_loss/valid_size)) 
                    #print('cross entropy loss: {:3.6f}'.format(train_loss))

                    
            writer.close()
                # If requested, sample n values of each variable and plot
            if plot_samples is not None:
                if self.latent_size ==2:
                    self.plotsamples2D(plot_samples)
                    self.plotstochasticsamples2D(plot_samples)
                else:
                    print("Latent size >2. Plot not possible (!)")
       
      
    def plotsamples2D(self,n):
        """
        Method for plotting samples from the latent space
        """
        # initialize image
        image = np.zeros((n*28,n*28))
        # populate image
        plt.figure(1,(10,10))
        for z_x in range(n):
            for z_y in range(n):
                z = np.array([norm.ppf((z_x/n) + 1/(2*n)),
                              norm.ppf((z_y/n) + 1/(2*n))],dtype=np.float32).reshape(1,2)
                sample = np.array(self.session.run([self.decoder(z)],feed_dict={self.z:z}))
                #tf.expand_dims(z, 1)
                image[z_x*28:(z_x+1)*28,z_y*28:(z_y+1)*28] = sample.reshape((28,28))
        image *= 255
        plt.imshow(image.astype(np.uint8),cmap="gray")
        plt.show()
    
    def plotstochasticsamples2D(self,n):
        """
        Method for plotting samples from the latent space
        """
        # initialize image
        image = np.zeros((n*28,n*28))
        # populate image
        plt.figure(2,(10,10))
        z_samples = np.random.normal(0,1,[n,n,2]).astype(dtype=np.float32)
        for z_x in range(n):
            for z_y in range(n):
                z = z_samples[z_x,z_y,:].reshape(1,2) #np.array([norm.ppf((z_x/n) + 1/(2*n)),
                              #norm.ppf((z_y/n) + 1/(2*n))],dtype=np.float32).reshape(1,2)
                sample = np.array(self.session.run([self.decoder(z)],feed_dict={self.z:z}))
                #tf.expand_dims(z, 1)
                image[z_x*28:(z_x+1)*28,z_y*28:(z_y+1)*28] = sample.reshape((28,28))
        image *= 255
        plt.imshow(image.astype(np.uint8),cmap="gray")
        plt.show()
                
    def plotsamplesHiD(self,n):
        """
        Method for plotting samples from the latent space projected (with PCA)
        to a 2-D space (i.e. selecting the 2 principal components)
        """
        # initialize image
        image = np.zeros((n*28,n*28,n))
        # populate image

        for z_x in range(n):
            for z_y in range(n):
                for z_z in range(n):
                    z = np.array([norm.ppf((z_x/n) + 1/(2*n)),
                                  norm.ppf((z_y/n) + 1/(2*n)),
                                  norm.ppf((z_z/n) + 1/(2*n))],dtype=np.float32).reshape(1,3)
                    sample = np.array(self.session.run([self.decoder(z)],feed_dict={self.z:z}))
                    
                    image[z_x*28:(z_x+1)*28,z_y*28:(z_y+1)*28,z_z] = sample.reshape((28,28))
        image *= 255
        for i in range(n):
            plt.figure(i+1,(8,8))
            plt.imshow(image[:,:,i].astype(np.uint8),cmap="gray")
            plt.show()         
        
        
        
    def test_model():
        """
        Method for carying out predictions with the model
        """
        pass
        
    def save_model():
        """
        Method for saving a model
        """
        pass
    def load_model():
        """
        Method for loading a trained model
        """
        pass
        
    def train_summaries():
        """
        Method for collecting training summaries in tensorboard
        """
        pass
    

    

        
        
