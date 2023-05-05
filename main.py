#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:processing.py
@date:2022/12/22 13:49
@desc:'这个模块是将原始的数据进行转化的，比如，将inital_data中的关联数据转化为矩阵的数据'
"""
import argparse
import copy
import gc
import time

import numpy as np
import torch
import visdom
from matplotlib import pyplot as plt
from matplotlib.pyplot import axes
from numpy import interp, mean
from sklearn.metrics import roc_curve, average_precision_score, recall_score, f1_score, accuracy_score, auc, \
    precision_recall_curve, precision_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from torch import optim, nn, sigmoid, softmax
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pre.dataloader import get_train_data, TrainSet, get_data, KFTrainSet

from model.model import TDSMDA
from utils import write_res, get_dict_list

parser = argparse.ArgumentParser(description='Run miRNA-disease')
parser.add_argument('--batch_size', default=32,
                    help='number of samples in one batch')
parser.add_argument('--heads', nargs='?', default=6,
                    help='number of multi-head')
parser.add_argument('--layers', nargs='?', default=6,
                    help='number of multi-layers')
parser.add_argument('--epochs', nargs='?', default=50,
                    help='number of epochs in SGD')
parser.add_argument('--lr', nargs='?', default=0.00002,
                    help='learning rate for the SGD')
parser.add_argument('--device', nargs='?', default='gpu',
                    help='training device')
args = parser.parse_args()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


def case_study(model, case_236_loader, case_50_loader, case_240_loader, index=None):
    # todo：model init
    model_params = torch.load('./model/best_model_dict-{}.pth'.format(index))
    model.load_state_dict(model_params)
    # todo: model testing
    res_logistic = []
    y_label, y_pred = [], []
    name_list = ['Lung Neoplasms', 'Breast Neoplasms', 'Lymphoma']
    i = 0
    axes = plt.subplots(1, 1, )
    for _loader in [case_236_loader, case_50_loader, case_240_loader]:
        res_logistic, res_label = [], []
        temp_data = np.array(_loader.dataset.md_feature)
        temp_label = np.array(_loader.dataset.md_label)
        for md_data, md_label in _loader:
            with torch.no_grad():
                # md_data = md_data
                logistics = model(md_data)
                """
                so this should be sorting the logistic scores that you get and then we get the asscoiated top 50,
                and this should be the score for getting the corresponding miRNA
                """
                #
                # temp_logistic=np.array(logistics)
                res_logistic.append(logistics)
                res_label += md_label

        res_logistic = torch.FloatTensor([_logistic.cpu().detach().numpy() for _logistic in res_logistic]).reshape(-1,
                                                                                                                   2)
        res_label = torch.FloatTensor([_label.cpu().detach().numpy() for _label in res_label])
        res_label = res_label.reshape(-1, 2)
        np.savetxt("dataSet/case study/{}.csv".format(name_list[i]), res_logistic, delimiter=',')


def main(data_type, case=None):
    # todo:set seed point
    torch.cuda.manual_seed(1234)
    viz = visdom.Visdom()
    data = get_train_data(data_type)
    # data = get_train_data('graph')
    # todo:bio+graph
    # data = get_data()
    md_train_set = TrainSet(data, 'train', data_type)
    md_test_set = TrainSet(data, 'test', data_type)
    md_val_set = TrainSet(data, 'val', data_type)

    md_train_loader = DataLoader(md_train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    md_test_loader = DataLoader(md_test_set, batch_size=32, shuffle=False, drop_last=True)
    md_val_loader = DataLoader(md_val_set, batch_size=64, shuffle=False, drop_last=True)
    # TODO:case study
    if case:
        case_236_set = TrainSet(data, 'case_163', data_type)
        case_50_set = TrainSet(data, 'case_50', data_type)
        case_240_set = TrainSet(data, 'case_240', data_type)
        case_236_loader = DataLoader(case_236_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
        case_50_loader = DataLoader(case_50_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
        case_240_loader = DataLoader(case_240_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    inchanel = data['md']['md_data'].shape[1]
    d_model = args.batch_size
    d_model = inchanel
    model = TDSMDA(d_model, inchanel, args.heads, args.layers, args.layers).cuda()

    # todo:study rate:gamma=0.1, last_epoch=- 1, verbose=False
    opt = optim.Adam(model.parameters(), lr=args.lr)
    # opt = optim.SGD(model.parameters(), lr=args.lr,momentum=0.1) # after testing,this opt is not suitable
    lambda1 = lambda epoch: epoch // 30
    lambda2 = lambda epoch: 0.99 ** epoch
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda2], last_epoch=-1, )
    # schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20,30],gamma=0.1)
    criteon = nn.MSELoss(reduction='mean')

    # mm, dd = data['mm'], data['dd']
    train_data = data['md']['md_data'].reshape(-1, 2, inchanel)
    label = data['md']['md_label'].reshape(-1, 2, 2)
    train_data, label = shuffle(train_data, label)

    train_func(md_train_loader, opt, model, criteon, viz, schedule, md_test_loader, md_val_loader)

    # todo:5-folding cross verification
    # k_fold_train_roc(train_data, label, viz, inchanel)
    # k_fold_train_pr(train_data, label, viz, inchanel) # pr
    # todo:case_study：which calls the trained model directly and runs the case_data data directly
    # case_study(model, case_236_loader, case_50_loader, case_240_loader)


def train_func(md_train_loader, opt, model, criteon, viz, schedule, md_test_loader, val_loader,
               case_loader=None, index=None):
    """
    model training
    :param md_train_loader: train data,shape=(10840，512)
    :param opt:Adam
    :param model: transformer
    :param criteon: loss function
    :param init_loss: init loss value
    :param schedule: follow up the learning rate regularly
    :param viz: plot photo
    :param md_test_loader:test data
    :return:
    """
    viz.line([0], [-1], win='LOSS', opts=dict(title='LOSS'))
    viz.line([0], [-1], win='val', opts=dict(title='val'))
    best_acc = 0
    for epoch in tqdm(list(range(1, args.epochs + 1))):
        total_loss = 0
        for md_data, md_label in md_train_loader:
            md_data = md_data.cuda()
            md_label = md_label.cuda()
            opt.zero_grad()
            res = model(md_data)
            loss = criteon(res, md_label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        # todo:val model
        val_acc, best_label, best_pred, best_pred_t = evalute(model, val_loader, 'val', index)
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            # torch.save(model, './model/best_model-{}.pth'.format(index)) # direct save model
            torch.save(model.state_dict(), './model/best_model_dict-{}.pth'.format(index))  # save the model params

        viz.line([total_loss], [epoch], win='LOSS', update='append')
        viz.line([best_acc], [epoch], win='val', update='append')
        schedule.step()
    # todo:model test
    test_acc, best_label, best_pred, true_pred = evalute(model, md_test_loader, 'test', index)

    return test_acc, best_label, best_pred, true_pred


def evalute(model, loader, eval_type, viz, index=None):
    """
    Calculate the ratio of correct data to incorrect data
    :param model:
    :param loader:
    :return:
    """
    if eval_type == 'test':
        model_params = torch.load('./model/best_model_dict-{}.pth'.format(index))
        model.load_state_dict(model_params)
        model.eval()

    for md_data, md_label in loader:
        # temp_data = np.array(md_data)
        # temp_label = np.array(md_data)
        with torch.no_grad():
            md_data = md_data
            md_tgt = md_data[-1, :, :].reshape(1, md_data.shape[1], md_data.shape[2])
            # logits = model(md_data, md_label)
            logits = model(md_data)
            true_pred = copy.deepcopy(logits).reshape(-1, 2).cpu().numpy()
            # pred = softmax(logits, dim=1)
            logits[logits > 0.5] = 1
            logits[logits <= 0.5] = 0
            pred = logits.reshape(-1, 2)
            md_label = md_label.reshape(-1, 2)

            # todo:some else performace evaluation index
            pred = pred.cpu().numpy()

            false_positive_rate, true_positive_rate, thresholds = roc_curve(md_label[:, 1], true_pred[:, 1],
                                                                            pos_label=1)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            ave = average_precision_score(np.array(md_label)[:, 1], true_pred[:, 1])
            precision = precision_score(np.array(md_label)[:, 1], pred[:, 1])
            re_score = recall_score(np.array(md_label)[:, 1], pred[:, 1])
            f1_score_ = f1_score(np.array(md_label)[:, 1], pred[:, 1])
            acc = accuracy_score(np.array(md_label)[:, 1], pred[:, 1])
            precision_, recall_, _ = precision_recall_curve(np.array(md_label)[:, 1], true_pred[:, 1])
            aupr = auc(recall_, precision_)
            # todo:get the best performance
            if best_acc < acc:
                best_pred_t = true_pred
                best_label = md_label
                best_pred = pred
                best_acc = acc
                best_auc = roc_auc
                best_f1score = f1_score_
                best_recall = re_score
                best_avg_score = ave
                best_aupr = aupr
                best_precision = precision

    if eval_type == 'test':
        write_res(best_acc, best_auc, best_f1score, best_recall, best_avg_score, best_aupr, list(best_pred),
                  list(best_label), best_precision)

    # return acc, best_label, best_pred, best_pred_t
    return best_acc, best_label, best_pred, best_pred_t


def k_fold_train_roc(train_data, label, viz, inchanel):
    """
    5-folding training
    :param train_data:train data
    :param label:train data label
    :param model:model
    :param criteon:loss function
    :param schedule:The learning rate decreases automatically
    :param opt:adam
    :param viz:visdom
    :return:
    """
    # Using StratifiedShuffleSplit methods stratified sampling resolution data, which USES 5 fold cross-validation training model
    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    index = 0

    for _train, _test in kf.split(train_data):
        viz.line([0], [-1], win='LOSS', opts=dict(title='LOSS'))
        # todo:Redefine the correlation of the model

        model = TDSMDA(inchanel, inchanel, args.heads, args.layers, args.layers).cuda()

        # model = torch.load('model.pth')
        # opt = optim.SGD(model.parameters(), lr=args.lr)
        # schedule = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9)
        opt = optim.Adam(model.parameters(), lr=args.lr)
        lambda2 = lambda epoch: 0.99 ** epoch
        schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda2], last_epoch=-1, )
        criteon = nn.MSELoss(reduction='mean')
        # todo:train data set
        train_set = train_data[_train]
        train_label = label[_train]
        train = KFTrainSet(train_set, train_label)
        train_loader = DataLoader(train, batch_size=64, shuffle=True, drop_last=True)  # 1000,128
        # todo:test data set
        test_set = train_data[_test]
        test_label = label[_test]
        test = KFTrainSet(test_set, test_label)
        test_loader = DataLoader(test, batch_size=32, shuffle=False, drop_last=True)
        # todo:Validation set, where half of the data is randomly sampled from the test machine
        resample_id = np.random.choice(_test, size=int(len(_test) / 2), replace=True)
        val_set = train_data[resample_id]
        val_label = label[resample_id]
        val = KFTrainSet(val_set, val_label)
        val_loader = DataLoader(val, batch_size=32, shuffle=True, drop_last=True)
        # todo:model training
        test_acc, best_label, best_pred, true_pred = train_func(train_loader, opt, model, criteon,
                                                                viz, schedule, test_loader, val_loader, index)


def k_fold_train_pr(train_data, label, viz, inchanel):
    """
    5-folding training
    :param train_data:train data
    :param label:train data label
    :param model:model
    :param criteon:loss function
    :param schedule:The learning rate decreases automatically
    :param opt:adam
    :param viz:visdom
    :return:
    """

    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    index = 0
    y_label, y_pred = [], []
    axes = plt.subplots(1, 1, )
    for _train, _test in kf.split(train_data):
        viz.line([0], [-1], win='LOSS', opts=dict(title='LOSS'))

        model = TDSMDA(args.batch_size, inchanel, 6).cuda()
        # model = MLP(inchanel)
        # model = torch.load('model.pth')
        opt = optim.SGD(model.parameters(), lr=args.lr)
        schedule = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9)
        criteon = nn.MSELoss(reduction='mean')

        train_set = train_data[_train]
        train_label = label[_train]
        train = KFTrainSet(train_set, train_label)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)  # 1000,128

        test_set = train_data[_test]
        test_label = label[_test]
        test = KFTrainSet(test_set, test_label)
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True)

        resample_id = np.random.choice(_test, size=int(len(_test) / 2), replace=True)
        val_set = train_data[resample_id]
        val_label = label[resample_id]
        val = KFTrainSet(val_set, val_label)
        val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_acc, best_label, best_pred, true_pred = train_func(train_loader, opt, model, criteon, viz, schedule,
                                                                test_loader, val_loader, index)


if __name__ == '__main__':
    main('graph_bio_not_pca')
    # main('bio_not_pca')
    # main('graph')
