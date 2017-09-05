# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 00:39:57 2017

@author: Heraclitus
"""
# General Imports
import numpy as np
import os.path
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from VAE_01 import VAE
# Command Line Interface Imports
import argparse
import sys
# Import MNIST example data 
from tensorflow.examples.tutorials.mnist import input_data

def binarize(images, threshold=0.1):
    """
    Binarize the images using a threshold 
    Input:  images: images as flat 1D numpy arrays, [num_images*image_pixels] 
            threshold: the value above which a pixel will be set to 1 
                       (otherwise it will be set to 0)
    Output: the binarized image
    """
    return (threshold < images).astype("float32")    

def flagsparser():
    parser = argparse.ArgumentParser(description='Command Line Functionality')
    parser.add_argument('-I', '--input', dest='input_size', action='store', default=784, type=int)
    parser.add_argument('-Z', '--latent', dest='latent_size', action='store', default=2, type=int)
    parser.add_argument('-H1', '--encoder_h', dest='h1_size', action='store', default=512, type=int)
    parser.add_argument('-H2', '--decoder_h', dest='h2_size', action='store', default=512, type=int)
    parser.add_argument('-Px', '--distribution_px', dest='p_x', action='store', default='Categorical', type=str)
    parser.add_argument('-Pz', '--distribution_pz', dest='p_z', action='store', default='Gaussian', type=str)
    parser.add_argument('-Qz', '--distribution_qz', dest='q_z', action='store', default='Gaussian', type=str)
    parser.add_argument('-B', '--batch_size', dest='batch_size', action='store', default=100, type=int)
    parser.add_argument('-E', '--epochs', dest='num_epochs', action='store', default=5, type=int)
    parser.add_argument('-Lr', '--l_rate', dest='learning_rate', action='store', default=1e-3, type=float)  
    parser.add_argument('-B1', '--beta1', dest='beta1', action='store', default=0.9, type=float)  
    parser.add_argument('-B2', '--beta2', dest='beta2', action='store', default=0.999, type=float)  
    parser.add_argument('-Ep', '--epsilon', dest='epsilon', action='store', default=1e-8, type=float)     
    parser.add_argument('-Rr', '--report_rate', dest='report_every', action='store', default=1, type=int)     
    parser.add_argument('-Dr', '--dropout', dest='dropout_prob', action='store', default=0.0, type=float,
                        help='If >0.0 it is the keep probability for training with dropout.')
    parser.add_argument('-L2', '--l2_reg', dest='l2_reg', action='store', default=0.001, type=float)
    parser.add_argument('-BN', '--batch_norm', dest='batch_norm', default=False)
    parser.add_argument('-LM', '--load', dest='load_model', default=False)
    parser.add_argument('-LD', '--load_dir', dest='load_dir', action='store', default="./saved_models/")
    parser.add_argument('-s', '--save', dest='save_flag', default=True)
    parser.add_argument('-sd', '--savedir', dest='save_dir', action='store', default='./saved_models/')
    parser.add_argument('-sr', '--save_res', dest='res_flag', default=True)
    parser.add_argument('-srd', '--resdir', dest='res_dir', action='store', default='res/')

    args, unparsed = parser.parse_known_args()     
    if unparsed:
        print('The following arguments were not understood:{}\nDefault values will be used'.format(unparsed))
    # Create a file_name to use for saving models
    params = args.h1_size, args.latent_size, args.h2_size, args.dropout_prob, args.batch_size, \
             args.l2_reg, args.num_epochs, args.learning_rate, args.beta1, args.beta2
    file_name = 'H1_{}-Z_{}-H2_{}-Dr_{}-B_{}-L2_{}-E_{}-Lr_{}-B1_{}-B2_{}'.format(*params)

  
    if args.res_flag is True:
        if not os.path.exists(args.res_dir):
            os.mkdir(args.res_dir)
        write_path = ''.join([args.res_dir, 'res_', file_name, '.txt'])
        write_dir = open(write_path, 'a')
    else:
        write_dir = None

    # Set model save destination
    if args.save_flag is True:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        save_path = ''.join([args.save_dir, 'model_', file_name, '.ckpt'])
    else:
        save_path = None

#    # Print parameters at head of results file
    args_dict = vars(args)
    print('---------- Run Parameters ----------', file=write_dir)
    [print("{}:\t{}".format(key, args_dict[key]), file=write_dir) for key in sorted(args_dict.keys())]

    return args, args_dict, write_dir, save_path


def main():
    
     # Load Data
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    # mnist: class which stores the training, validation, and test sets as NumPy arrays. 
    # also provides a function for iterating through data minibatches.
    # Binirize and split into training, and test sets
    train_images = binarize(mnist.train.images)
    #    train_labels = mnist.train.labels
    valid_images = binarize(mnist.test.images)
    #    valid_labels = mnist.test.labels
    params, params_dict, write_dir, save_path =flagsparser()       
     
    _VAE_=VAE(params_dict)      
    
    _VAE_.train_model(train_images,valid_images,learning_rate= 3e-4,num_epochs =1,
                      l2_reg = 0.001, load_model = True, train_mode= False, 
                      load_dir=save_path, save_dir=save_path, write_path=write_dir, 
                      plot_samples=15)
    return _VAE_

if __name__ == "__main__":
    
    tf.set_random_seed(125)
    
    main()
        
        