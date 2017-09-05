import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import math
from skimage.data import camera
from skimage.filters import roberts
from skimage import transform as transf
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.examples.tutorials.mnist import input_data
from bayes_opt import BayesianOptimization
import datetime
import seaborn as sns
tf.reset_default_graph()


###############################################################################
##########                      HELPER FUNCTIONS
###############################################################################

def binarize(image, thresshold=0.1):
    return (image > thresshold).astype(np.float32)


def get_edges(image, thresshold=0.1):
    s = image.shape[0]
    image = np.reshape(image, (s, 28, 28))
    ret = np.zeros_like(image)
    for i in range(s):
        ret[i, :, :] = binarize(roberts(image[i, :, :]), thresshold)
    return np.reshape(ret, (s, 28 * 28))


def rot90(image, angle=math.pi / 4):
    s = image.shape[0]
    image = np.reshape(image, (s, 28, 28))
    #    image = np.reshape(image,(28,28))
    ret = np.zeros_like(image)
    t_matrix = transf.SimilarityTransform(scale=1, rotation=angle,
                                          translation=(12, -5))
    for i in range(s):
        ret[i, :, :] = binarize(transf.warp(image[i, :, :], t_matrix))
    return np.reshape(ret, (s, 28 * 28))


def shuffle_data(x, y):
    # randomly shuffle data
    x = np.array(x)
    y = np.array(y)
    new_indices = np.random.permutation(x.shape[0])
    shuffled_x = x[new_indices]+np.random.normal(0,0.05,size=x.shape)
    shuffled_y = y[new_indices]
    return shuffled_x, shuffled_y

def shuffle_x_data(x):
    # randomly shuffle data
    new_indices = np.random.permutation(x.shape[0])
    shuffled_x = x[new_indices, :]
    return shuffled_x

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        plt.pause(0.0001)
    return fig


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

def plot_2hist(var1, var2, print_dir, dataset, title='Histogram', f_name="" ,figsize=(5, 5), plots=True,lim=25):
    # Print a histogram of the calculated ELBO values
    if plots:
        #lim = max(max(var1[:-1]),max(var2[:-1]))
        plt.figure(figsize=figsize)

        plt.subplot(211)
        axes = plt.gca()
        sns.distplot(var1, kde=False, norm_hist=False, bins=20, color='green', label='true normal')
        plt.ylabel('Counts', fontsize=16)
        axes.set_xlim([-0.2, lim])
        plt.legend()
        plt.title("$\mathcal{L}_{ELBO}$ Histograms\n" + "Assuming {} class\n".format(title), fontsize=16)
        plt.tight_layout()
        plt.subplot(212)
        axes = plt.gca()
        sns.distplot(var2, kde=False, bins=50, norm_hist=False,color='red', label='true anomalies')
        plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=16)
        plt.ylabel('Counts', fontsize=16)
        axes.set_xlim([-0.2, lim])
        plt.legend()
        #plt.tight_layout()
        if f_name != "":
            plt.savefig(print_dir + dataset + f_name)
        plt.show()

def elbo_hist( normal_elbo,anomaly_elbo,anomaly_threshold, title, filename, print_dir, dataset, plots=True):
    if plots:
        plt.figure()
        #sns.distplot(normal_elbo, kde=True, color='blue',  label='Normal')
        plt.hist(normal_elbo, bins=50, histtype='bar', normed=True,color='b', label='Normal')
        plt.hist(anomaly_elbo, bins=50, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Anomalous')
        #sns.distplot(anomaly_elbo, kde=True, color='red', label='Anomalous')
        plt.axvline(x = anomaly_threshold, linewidth=4, color ='g', label='Threshold')
        tit="Dataset: {} - {}".format(dataset,title)+"\nNormalised Histogram"
        plt.title(tit,size= 18)
        plt.xlabel("Evidence Lower Bound, $\mathcal{L}_{ELBO}$",size= 16)
        plt.ylabel("Empirical Density",size= 16)
        plt.axis([0, max(anomaly_elbo)*1.1, 0, 0.8])
        plt.legend(loc='upper left')
        plt.grid(True,which='both')
        if filename != "":
            plt.savefig(print_dir + dataset + filename)
        plt.show()


def plot_hist(var, print_dir, dataset, title='Histogram', f_name="" ,figsize=(5, 5), plots=True):
    # Print a histogram of the weights of a layer
    # to inspect for weight saturation
    if plots:
        plt.figure(figsize=figsize)
        plt.hist(var, bins=50)
        plt.title("Dataset: {}, {}".format(dataset, title))
        plt.xlabel('Loss')
        plt.ylabel('Counts')
        plt.tight_layout()
        if f_name != "":
            plt.savefig(print_dir + dataset + f_name)
        plt.show()


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 0.1 / tf.sqrt(in_dim / 2.)
    return tf.truncated_normal(shape=size, stddev=xavier_stddev)

def bias_init(size):
    return tf.constant(0.1,shape=size)

def zeros_init(size):
    return tf.zeros(shape=size)

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
    return x_mean, x_std, (x - x_mean) / x_std


def p_normalize(x, x_mean, x_std):
    """
    Function that normalizes data-points
    Input:   x = dataset
             x_mean = row vector of mean per column
             x_std  = row vector of std per column
    Returns:
             x_norm = normalized data set
    """

    return (x - x_mean) / x_std


def denormalize(x, x_mean, x_std):
    """
    Function that undoes the normalization
    Input:   x = normalized dataset
             x_mean = row vector of mean per column
             x_std  = row vector of std per column
    Returns: unnormalized data set
    """
    return (x * x_std) + x_mean

def report_stats(title, res, verbose=True):
    if verbose:
        print(title)
        print("min_elbo: {}".format(res.min()))
        print("Mean elbo: {}".format(res.mean()))
        print("max_elbo: {}".format(res.max()))
        print("Std elbo: {}".format(res.std()))
        print("Median elbo: {}".format(np.median(res)))
    return

def progress(epoch, eloss,rloss,klloss, etime):
    print('Epoch: {}'.format(epoch))
    print('Loss: {:.4}'.format(eloss))
    print("time: {}".format(etime))
    print("Reconstruction loss: {}".format(rloss))
    print("KL divergence: {}".format(klloss))
    return



def select_threshold(elbo_dict, criterion='precision'):
    """
    Python function that selects the anomaly threshold criterion based on the \n
    criterion : precision/ recall and the mean/std of the normal/anomaly elbos
    in the validation set
    Input:   criterion:      precision/ recall/ balanced
             dict:           mean/std dictionary (normal/anomaly)
    Returns: threshold
    """
    d = elbo_dict
    if criterion == 'precision':
        threshold = max(d['a_mean'] - 2 * d['a_std'],
                        d['n_mean'] + 5 * d['n_std'])
    elif criterion == 'recall':
        threshold = min(d['a_mean'] - 5 * d['a_std'],
                        d['n_mean'] + 2 * d['n_std'])
    else:
        threshold = d['n_mean'] + abs(d['n_mean'] - d['a_mean'])
    # threshold = max( d['a_mean'] - 3 * d['a_std'],
    #                         d['n_mean'] + 3 * d['n_std'])
    return threshold

def scores(y_true, y_pred, verbose = True):
    '''
    :param y_true: true labels (anomaly/normal)
    :param y_pred: predicted labels
    :return: tp,fp,fn,tn, precision, recall, f1 score
    '''
    # p = positive = normal, n = negative = anomaly
    tp = sum(y_true[:,0] & y_pred[:,0])
    tn = sum(y_true[:,1] & y_pred[:,1])
    fp = sum(y_true[:,1] & y_pred[:,0])
    fn = sum(y_true[:,0] & y_pred[:,1])
    # score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    if verbose:
        print("#=====================================#")
        print("True negatives:{}".format(tn))
        print("False positives:{}".format(fp))
        print("Precision:{}".format(precision))
        print("Recall:{}".format(recall))
        print("f1-score:{}".format(f1))
    return tp,fp,fn,tn,precision,recall,f1

def scores_x(tp,fp,fn,tn, verbose=True):
    precision = tp.mean() / (tp.mean() + fp.mean())
    recall = tp.mean() / (tp.mean() + fn.mean())
    f1 = 2 * precision * recall / (precision + recall)
    if verbose:
        print("#=====================================#")
        print("True positives:\t{} +/-:{:6.2f} ".format(tp.mean(), 2 * tp.std()))
        print("False positives:\t{} +/-:{:6.2f}".format(fp.mean(), 2 * fp.std()))
        print("False negatives:\t{} +/-:{:6.2f}".format(fn.mean(), 2 * fn.std()))
        print("True negatives:\t{} +/-:{:6.2f}".format(tn.mean(), 2 * tn.std()))
    return precision, recall, f1

def uncertainty(y_true, y_pred, verbose=True):
    yp_std = y_pred.std(axis=1).reshape([-1,1])
    yp_mean = y_pred.mean(axis=1).reshape([-1,1])
    index = np.arange(y_true.shape[0]).reshape([-1,1])
    y_out = np.concatenate((index,y_true.reshape([-1,1]),yp_mean,yp_std),axis=1)
    y_out =y_out[y_out[:,3].argsort()[::-1]]
    if verbose:
        print("Predictions with highest uncertainty:")
        for i in range(20):
            print('Data point:{:10.0f}, True Label:{}, Predicted Label:{}, Prediction Std:{:6.4f}'.format(\
                y_out[i,0],y_out[i,1],y_out[i,2],y_out[i,3]))
    return y_out




def nontrainable():
    trainables = tf.trainable_variables()
    all_vars   = tf.all_variables()
    return [v for v in all_vars if v not in trainables]

def trainable():
    trainables = tf.trainable_variables()
    all_vars   = tf.global_variables()# tf.all_variables()
    return [v for v in all_vars if v in trainables]

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

# initialize multiple RNN cells with state is tuple
# state_per_layer_list = tf.unpack(init_state, axis=0)
# rnn_tuple_state = tuple(
#     [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
#      for idx in range(num_layers)]
# )

# https://medium.com/@erikhallstrm/using-the-tensorflow-multilayered-lstm-api-f6e7da7bbe40

# batch norm - from the books' web-site introduction to DNN
# with tf.name_scope("train"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(extra_update_ops):
#         training_op = optimizer.minimize(loss)

# [v.name for v in tf.trainable_variables()]
# [v.name for v in tf.global_variables()]

def unpool(updates, mask, ksize=[1, 2, 2, 1], output_shape=None, name=''):
    with tf.variable_scope(name):
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

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

def threshold_selector(normal_elbo,anomaly_elbo,fp_ratio,fn_ratio):
    """
    Perform BayesianOptimization to select the threshold value
    """
    gp_params = {"alpha": 1e-5}
    min_t = min(normal_elbo)
    max_t = np.max(anomaly_elbo)
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
            {'threshold': (min_t, max_t)})
    VAE_BO.explore({'threshold': [min_t, (min_t+max_t)/2, max_t]})
    VAE_BO.maximize(n_iter=10, **gp_params)
    return VAE_BO.res['max']['max_params']['threshold'], VAE_BO

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
        plt.legend(loc='upper left',size= 16)
        plt.grid(True,which='both')
        if filename != "":
            plt.savefig(print_dir + dataset + filename)
        plt.show()