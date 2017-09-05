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
    new_indices = np.random.permutation(x.shape[0])
    shuffled_x = x[new_indices, :]
    shuffled_y = y[new_indices, :]
    return shuffled_x, shuffled_y


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


def plot_hist(var, title='Histogram', f_name="", figsize=(5, 5)):
    # Print a histogram of the weights of a layer
    # to inspect for weight saturation
    plt.figure(figsize=figsize)
    plt.hist(var, bins=20)
    plt.title("Dataset: {}, {}".format(dataset, title))
    plt.xlabel('Loss')
    plt.ylabel('Counts')
    plt.tight_layout()
    if f_name != "":
        plt.savefig(save_print_dir + dataset + f_name)
    plt.show()


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


def report_stats(title, res):
    print(title)
    print("min_elbo: {}".format(res.min()))
    print("Mean elbo: {}".format(res.mean()))
    print("max_elbo: {}".format(res.max()))
    print("Std elbo: {}".format(res.std()))
    print("Median elbo: {}".format(np.median(res)))
    return


def progress(epoch, eloss, etime):
    print('Epoch: {}'.format(epoch))
    print('Loss: {:.4}'.format(eloss))
    print("time: {}".format(etime))
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

def scores(y_true, y_pred):
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
    print("#=====================================#")
    print("True negatives:{}".format(tn))
    print("False positives:{}".format(fp))
    print("Precision:{}".format(precision))
    print("Recall:{}".format(recall))
    print("f1-score:{}".format(f1))
    return tp,fp,fn,tn,precision,recall,f1


def auc_plot(t_elbo,y_t,threshold=1,f01=1e-5):
    # convert y_true to 1-d if it is one-hot encoded
    try:
        y_t.shape[1]
        y_true = y_t[:,0]==0  # will be 1 if anomaly 0 if normal
    except IndexError:
        y_true=y_t
    # convert elbo to probability
    y_prob = t_elbo / max(t_elbo)
    print("Max_Prob",max(y_prob))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label="ROC curve\nAUC = {:8.7f}\nf0.1 score = {:8.7f}".format(auc_score,f01))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',size= 16)
    plt.ylabel('True Positive Rate',size= 16)
    plt.title('Receiver operating characteristic curve',size= 18)
    plt.legend(loc="lower right")
    plt.show()
    return auc_score,thresholds