#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:evaluate.py
@date:2023/1/19 14:27
@desc:It mainly includes some evaluation of the model, including 50 percent cross validation, 61 cross validation, and so on
"""
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import StratifiedKFold

tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)

def plot_roc(label, pred, n_classes=2):
    """
    plot roc
    :return:
    """
    len_label = len(label)
    colors = ['aqua', 'darkorange', 'cornflowerblue','jet','turbo']
    # roc_auc={}
    plt.figure()
    for i in range(len_label):
        fpr = dict()
        tpr = dict()
        roc_auc = {}
        lw = 2
        label_i = label[str(i)].reshape(-1, 2).detach().numpy()
        pred_i = pred[str(i)].reshape(-1, 2).detach().numpy()
        for n in range(n_classes):
            fpr[n], tpr[n], _ = roc_curve(label_i[:, n], pred_i[:, n])
            roc_auc[n] = auc(fpr[n], tpr[n])

        fpr["micro"], tpr["micro"], _ = roc_curve(label_i.ravel(), pred_i.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.plot(fpr[i][1], tpr[i][1], color=colors[i],
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


def plot_multi_roc(label, pred, n_classes=2):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(label.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # todo:Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # todo:Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
def kf_5_plot(pred,label,i):
    fpr, tpr, thresholds = roc_curve(label[:, 1], pred[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # calculate auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # To plot, just plt.plot(fpr,tpr) is needed, and the variable roc_auc simply records the value of auc, calculated by the auc() function
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
def plot_luck():
    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # Calculate the average AUC value
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()




if __name__ == '__main__':
    pass
