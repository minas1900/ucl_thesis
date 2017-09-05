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
from sklearn import metrics
import itertools
from scipy.interpolate import spline
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


def plot_hist(var, print_dir, dataset, title='Histogram', f_name="" ,figsize=(5, 5), plots=True):
    # Print a histogram of the calculate ELBO values
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

def plot_2hist(var1, var2, print_dir, dataset, title='Histogram', f_name="" ,figsize=(5, 5), plots=True):
    # Print a histogram of the calculate ELBO values
    if plots:
        plt.figure(figsize=figsize)
        plt.subplot(211)
        axes = plt.gca()
        sns.distplot(var1, kde=False, norm_hist=True, bins=50, color='green', label='normal data')
        plt.ylabel('Emp. Density', fontsize=14)
        axes.set_ylim([0, 0.5])
        plt.subplot(212)
        axes = plt.gca()
        sns.distplot(var2, kde=False, bins=50, norm_hist=True,color='green', label='anomalous data')
        plt.suptitle("Calculated Test Set, $\mathcal{L}_{ELBO}$ Histograms"+"\n - Dataset: {}\n".format(dataset), fontsize=18)
        plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=14)
        plt.ylabel('Emp. Density', fontsize=14)
        axes.set_ylim([0, 0.5])
        #plt.tight_layout()
        if f_name != "":
            plt.savefig(print_dir + dataset + f_name)
        plt.show()


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

def save2txt(save_print_dir ,file_name, dataset, run_comment, h_dim, z_dim, num_z_samples, learning_rate, batch_size, drop_probs, l2_reg,
             save_epochs, early_stopping,x_dim,train_size,valid_size,test_size,anom_train,anom_valid,anom_test,save_dir):
    '''
    Helper function for saving parameters and hyperparameters
    '''
    res_file_name = save_dir+file_name[:-4]+'res'+'.npz'
    file_name = save_print_dir +file_name + '.txt'
    mf = open(file_name, 'w')
    dt = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    mf.write("File Name:\t{}\nDate:\t{}\n".format(file_name, dt))
    #mf.write("----------------------------------------------------\n".format(dataset, run_comment))
    mf.write("DataSet:\t{}\tModel:\t{}\n".format(dataset, run_comment))
    #mf.write("----------------------------------------------------\n".format(dataset, run_comment))
    mf.write("h Dim.:\t{}\tz Dim.:\t{}\tx dim:\t{}\tnum z samples:\t{}\n".format(h_dim, z_dim, x_dim, num_z_samples))
    mf.write("Learning rate:\t{}\tBatch size:\t{}\tDropout prob:\t{}\tL2 reg:\t{}\n".format(learning_rate,
                                                                                            batch_size, drop_probs,
                                                                                            l2_reg))
    mf.write("N_epochs:\t{}\tEarly Stopping:\t{}\n".format(save_epochs,early_stopping))
    mf.write("Train size:\t{}\tValid size:\t{}\tTest size:\t{}\n".format(train_size,valid_size,test_size))
    mf.write("Train anomalies:\t{}\tValid anomalies:\t{}\tTest anomalies:\t{}\n".format(anom_train,anom_valid,anom_test))
    mf.write("np train file:\t{}\n".format(file_name[:-4] + '.npz'))
    mf.write("np result file:\t{}\n\n".format(res_file_name))
    mf.close()
    return

def saveres2txt(save_print_dir, file_name, dataset,anomaly_threshold, tp,fp,tn,fn,f1, auc,f01,precision,recall):
    '''
    Helper function for saving results
    '''
    file_name = save_print_dir + file_name + '.txt'
    mf = open(file_name, 'a')
    dt = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    mf.write("\n-------------------------------  Results ---------------------------------\n")
    mf.write("DataSet:\t{}\tAnomaly threshold:\t{}\n".format(dataset, anomaly_threshold))
    mf.write("tp:\t{}\tfp:\t{}\tfn:\t{}\ttn:\t{}\n".format(tp,fp,fn,tn))
    mf.write("AUC:\t{}\tf1-score:\t{}\tf0.1-score:\t{}\n".format(auc,f1,f01))
    mf.write("Precision:\t{}\tRecall:\t{}\n".format(precision,recall))
    mf.close()
    return

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



def unsupervised_elbo_analysis(v_elbo, t_elbo, y_test,fn_ratio=10):
    v_length = v_elbo.shape[0]
    length = t_elbo.shape[0]
    #rates = [0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.08,0.1]
    rates = [0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8]
    tn,fn,tp,fp,precision,recall,f1,f01 = [[] for i in range(8)]
    count=0
    sorted_v_elbo = np.sort(v_elbo,axis=0,kind='mergesort')
    for anom_rate in rates :
        idx = int(np.ceil(v_length * anom_rate))
        anomaly_threshold = sorted_v_elbo[-idx]
        # save accuracy rates
        tn.append(sum([t_elbo[j] < anomaly_threshold and y_test[j, 0] == 1 for j in range(length)]))
        fp.append(sum([t_elbo[j] > anomaly_threshold and y_test[j, 0] == 1 for j in range(length)]))
        tp.append(sum([t_elbo[j] > anomaly_threshold and y_test[j, 0] == 0 for j in range(length)]))
        fn.append(sum([t_elbo[j] < anomaly_threshold and y_test[j, 0] == 0 for j in range(length)]))
        # calculate and report total stats
        # scores
        precision.append(tp[count] / (tp[count] + fp[count]))
        recall.append(tp[count] / (tp[count] + fn[count]))
        f1.append(2 * precision[count] * recall[count] / (precision[count] + recall[count]))
        f01.append((1 + (1 / fn_ratio) ** 2) * precision[count] * recall[count] / ((1 / fn_ratio ** 2) * precision[count] + recall[count]))
        count+=1
    # bring back the auc score
    return precision,recall,f1, f01, tn,fn,tp,fp,rates

def select_elbo_region(v_elbo, t_elbo, y_test,e_min,e_max,fn_ratio=10):

    length = t_elbo.shape[0]
    tn = sum([ (t_elbo[j] < e_min or t_elbo[j]> e_max) and y_test[j, 0] == 1 for j in range(length)])
    fp = sum([ (t_elbo[j] >= e_min and t_elbo[j] <= e_max) and y_test[j,0] == 1 for j in range(length)])
    tp = sum([ (t_elbo[j] >= e_min and t_elbo[j] <= e_max) and y_test[j,0] == 0 for j in range(length)])
    fn = sum([ (t_elbo[j] < e_min or t_elbo[j]> e_max) and y_test[j, 0] == 0 for j in range(length)])
    # calculate and report total stats
    # scores
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    f01 = (1 + (1 / fn_ratio) ** 2) * precision * recall / ((1 / fn_ratio ** 2) * precision + recall)

    # bring back the auc score
    return precision,recall,f1, f01, tn,fn,tp,fp


def post_plot_unsuper(v_elbo,t_elbo,threshold,e_min, e_max):
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    axes = plt.gca()
    sns.distplot(v_elbo, bins=50,kde=False, color='green')
    # fig.legend('Anomalies','normal')
    # plt.legend(handles=[l1,l2])
    plt.title("Calculated Validation Set $\mathcal{L}_{ELBO}$ Histogram", fontsize=18)
    plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.tight_layout()
    plt.axvline(e_min, color='r', linestyle='--', label='zoom region')
    plt.axvline(e_max, color='r', linestyle='--')
    plt.legend(fontsize=14)
    #axes.set_ylim([0, 0.8])
    axes.set_xlim([0, np.max(v_elbo,axis=0)])
    plt.show()
    plt.subplot(222)
    axes = plt.gca()
    sns.distplot(v_elbo,bins=50, kde=False, color='green')
    # fig.legend('Anomalies','normal')
    # plt.legend(handles=[l1,l2])
    plt.title("Validation set: Zoom into secondary $\mathcal{L}_{ELBO}$ peak", fontsize=18)
    plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.tight_layout()
    plt.axvline(e_min, color='r', linestyle='--')
    plt.axvline(e_max, color='r', linestyle='--')
    plt.legend(fontsize=14)
    plt.ylim([0,500])
    #axes.set_ylim([0, 0.003])
    axes.set_xlim([e_min, e_max])
    plt.show()
    plt.subplot(223)
    axes = plt.gca()
    sns.distplot(t_elbo,bins=50 , kde=False, color='blue')
    # fig.legend('Anomalies','normal')
    # plt.legend(handles=[l1,l2])
    plt.title("Calculated Test Set $\mathcal{L}_{ELBO}$ Histogram", fontsize=18)
    plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.tight_layout()
    plt.axvline(e_min, color='r', linestyle='--', label='zoom region')
    plt.axvline(e_max, color='r', linestyle='--')
    plt.legend(fontsize=14)
    #axes.set_ylim([0, 0.8])
    axes.set_xlim([0, np.max(v_elbo,axis=0)])
    plt.show()
    plt.subplot(224)
    axes = plt.gca()
    sns.distplot(t_elbo,bins=50, kde=False, color='blue')
    # fig.legend('Anomalies','normal')
    # plt.legend(handles=[l1,l2])
    plt.title("Test Set: Zoom into secondary $\mathcal{L}_{ELBO}$ peak", fontsize=18)
    plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.tight_layout()
    plt.axvline(e_min, color='r', linestyle='--')
    plt.axvline(e_max, color='r', linestyle='--')
    plt.legend(fontsize=14)
    plt.ylim([0,500])
    #axes.set_ylim([0, 0.003])
    axes.set_xlim([e_min, e_max])
    plt.show()

    return

def post_plot_unsuper2(v_elbo,threshold):
    plt.figure()
    axes = plt.gca()
    sns.distplot(v_elbo, bins=50,kde=False, color='green')
    # fig.legend('Anomalies','normal')
    # plt.legend(handles=[l1,l2])
    plt.title("Calculated Validation Set $\mathcal{L}_{ELBO}$ Histogram", fontsize=18)
    plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.tight_layout()
    plt.axvline(threshold, color='r', linestyle='--', label='threshold')
    plt.legend(fontsize=14)
    #axes.set_ylim([0, 0.8])
    axes.set_xlim([0, np.max(v_elbo,axis=0)])
    plt.show()
    return

def post_plot_unsuper3(v_elbo,threshold):
    lim = -30
    v_elbo = np.sort(v_elbo, axis=0)[:lim]
    plt.figure()
    axes = plt.gca()
    plt.hist(v_elbo, bins=50,log=False,normed=True, color='green')
    # fig.legend('Anomalies','normal')
    # plt.legend(handles=[l1,l2])
    plt.title("Calculated Validation Set $\mathcal{L}_{ELBO}$ Histogram", fontsize=18)
    plt.xlabel('Evidence Lower Bound, $\mathcal{L}_{ELBO}$', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    plt.tight_layout()
    plt.axvline(threshold, color='r', linestyle='--', label='threshold')
    plt.legend(fontsize=16)
    #axes.set_ylim([0, 0.8])
    axes.set_xlim([0, np.median(v_elbo,axis=0)*5])
    plt.show()
    return

def post_plot_AE(v_elbo,t_elbo):
    lim = -1
    #v_elbo = np.sort(v_elbo, axis=0)[:lim]
    plt.figure()
    plt.subplot(121)
    #axes = plt.gca()
    plt.hist(v_elbo, log=False,normed=False, color='green',bins=50)#bins=10,
    plt.title("Validation set", fontsize=18)
    plt.xlabel('Reconstruction Loss', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    #plt.tight_layout()
    plt.xlim([0, 1])
    plt.show()
    plt.subplot(122)
    #axes = plt.gca()
    plt.hist(t_elbo, log=False,normed=False, color='green',bins=250)#bins=50,
    plt.title("Test set", fontsize=18)
    plt.xlabel('Reconstruction Loss', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    plt.xlim([0, 1])
    #plt.suptitle("Reconstruction Error Histogram - http dataset", fontsize=18)
    plt.show()
    return