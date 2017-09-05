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

def hidden_layer(x, name="",scope=None,reuse=False,trainable = True,\
				input_size=784,num_units=512, act_fun =tf.nn.relu,\
				batch_norm=False,keep_prob=0.0,l2_reg=0.0,train_mode = True):
    """
    Generic helper function for generating hidden layers  
    """
    # Variable Initializers
    w_init = tf.contrib.layers.xavier_initializer(
    uniform=False, seed=None , dtype = tf.float32 ) 
    b_init = tf.constant_initializer(0.)
    # Regularizer
    if l2_reg > 0.0:
        l2 = tf.contrib.layers.l2_regularizer(l2_reg)
    else:
        l2 = None
    # Define variables
    namw = "w_"+name	
    namb = "b_"+name
    with tf.variable_scope(scope) as sc:
        
        namw = tf.get_variable(name=str(namw) ,shape = [input_size,num_units],\
									dtype = tf.float32, trainable = trainable,\
									initializer = w_init, regularizer = l2 )

        namb = tf.get_variable( name=str(namb) ,shape = [num_units],\
							dtype = tf.float32, trainable = trainable,\
							initializer = b_init, regularizer = None )							
    # Define network
        if batch_norm:
            x = tf.layers.batch_normalization( x, training= train_mode )
        activ = tf.add(tf.matmul( x,namw,name="x_weights"),\
								namb, name="plus_bias")
        if batch_norm:
            activ = tf.layers.batch_normalization(activ, training=train_mode)         
        out = tf.nn.relu(activ,name="hidden_out")
        if keep_prob>0.0:
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                out = tf.nn.dropout(out, keep_prob)
                tf.summary.scalar('dropout_keep_probability', keep_prob)
    # write summaries
#        with tf.name_scope("Summaries"):
#            tf.summary.histogram("w_"+name,"w_"+name)
#            tf.summary.histogram("b_"+name,"b_"+name)
#            tf.summary.histogram("hidden_activations",tf.nn.relu(activ))				
    return out

	
def linear_layer(x, name="",scope=None,reuse=False,trainable = True,\
				input_size=784,num_units=512, act_fun =tf.identity,\
				batch_norm=False,keep_prob=0.0,l2_reg=0.0,train_mode = True):
    """
    Generic helper function for generating hidden layers  
    """
    # Variable Initializers
    w_init = tf.contrib.layers.xavier_initializer(
    uniform=False, seed=None , dtype = tf.float32 ) 
    b_init = tf.constant_initializer(0.)
    # Regularizer
    if l2_reg > 0.0:
        l2 = tf.contrib.layers.l2_regularizer(l2_reg)
    else:
        l2 = None
    # Define variables	
    with tf.variable_scope(scope) as sc:
        namw = "w_"+name
        namw = tf.get_variable( name=str(namw) ,shape = [input_size,num_units],\
									dtype = tf.float32, trainable = trainable,\
									initializer = w_init, regularizer = l2 )
        namb = "b_"+name
        namb = tf.get_variable( name=str(namb) ,shape = [num_units],\
							dtype = tf.float32, trainable = trainable,\
							initializer = b_init, regularizer = None )							
    # Define network
        if batch_norm:
            x = tf.layers.batch_normalization( x, training= train_mode )
        activ = tf.add(tf.matmul( x,namw,name="x_weights"),\
								namb, name="plus_bias")
        if batch_norm:
            activ = tf.layers.batch_normalization(activ, training=train_mode)         
        out = tf.nn.relu(activ,name="hidden_out")
        if keep_prob>0.0:
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                out = tf.nn.dropout(out, keep_prob)
                tf.summary.scalar('dropout_keep_probability', keep_prob)
    # write summaries
#        with tf.name_scope("Summaries"):
#            tf.summary.histogram("w_"+name,"w_"+name)
#            tf.summary.histogram("b_"+name,"b_"+name)
#            tf.summary.histogram("hidden_activations",tf.nn.relu(activ))				
    return out
