import math
import time

import numpy as np
import pandas as pd
import torch
import xlrd

from config.conf import Config

conf = Config()


def cal_data():
    """
    Calculate the mean and standard deviation
    :return:
    """
    data_list = {'acc': [0.9259, 0.9226, 0.9306, 0.9375, 0.9312],
                 'Pre': [0.8571, 0.8803, 0.8636, 0.8592, 0.8750],
                 'recall': [0.8929, 0.8750, 0.8973, 0.8889, 0.8991],
                 'f1_score': [0.9231, 0.9167, 0.9143, 0.9118, 0.9242],
                 'auc': [0.9582, 0.9531, 0.9587, 0.9682, 0.9602]}
    for _list in data_list.keys():
        avg = sum(data_list[_list]) / 5
        sum_data = 0
        for data in data_list[_list]:
            temp_i = data - avg
            sum_data += temp_i * temp_i
        std = math.sqrt(sum_data / 5)
        print(_list, avg, math.sqrt(std))


def write_res(acc, auc, f1_score, recall, avg_score, best_aupr, pred, label, precision):
    """best_auc,best_f1score,best_recall,best_avg_score
    The result is recorded in log
    :return:
    """
    # todo:生成当前时间
    t = time.localtime()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", t)

    with open('out/result.txt', 'a+') as f:
        f.write('==================================================\n')
        # f.write('%s'%model_name)
        f.write(
            "%s:acc:%.4f,auc:%.4f,f1_score:%.4f,reacall:%.4f,avg_score:%.4f,aupr:%.4f,precision:%.4f "
            "result as follows:\n" % (
                current_time, acc, auc, f1_score, recall, avg_score, best_aupr, precision))
        f.write("pred:{}\n".format(pred))
        f.write("label:{}\n".format(label))
        f.write('\n')
        print(
            "%s:acc:%.4f,auc:%.4f,f1_score:%.4f,reacall:%.4f,avg_score:%.4f,aupr:%.4f,precision:%.4f \n"
            "result as follows:\n" % (
                current_time, acc, auc, f1_score, recall, avg_score, best_aupr, precision,))


def convert_to_windows(data, n_window):
    """
    Input: Then the result is to input the two-dimensional data, and then to make 10 copies of the same dimension of the data, the final result
    """
    windows = []
    w_size = n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)


def get_mi_name(mi_index):
    """
    Based on the mi_index that was passed in
    :param mi_index:
    :return:
    """
    pd_data = pd.read_csv(conf.init_m_name_path, header=None).values
    # todo:Get the data for each row, and then get the corresponding mi_name based on index
    for _data in pd_data:
        _data_list = _data[0].split('\t')
        if str(mi_index) in _data_list[0]:
            return _data_list[1]
        else:
            continue
    return None


def case_study(path, path1):
    """
    Matching of mirna data for excel tables
    :return:
    """
    # todo:Read the exce form, the form for predicting disease
    pd_data = pd.read_csv(path, header=None)
    print(1)
    mi_col = pd_data.values[:, 2]
    mi_name_list = []
    for i in mi_col:
        # todo:Then go to mi_rna and find the corresponding mirna_name,
        mi_name = get_mi_name(int(i))
        #  And then write it into the table
        mi_name_list.append(mi_name)
    pd_mi = pd.DataFrame(mi_name_list)
    pd_data.insert(loc=3, column='name', value=pd_mi)
    pd_data.to_csv('dataSet/case study/{}_verify.csv'.format(path1))


def case_data_mark(path, name):
    """
    !!! There is a certain error rate in method marking, which is manually marked by oneself at present
    It is used to search for the corresponding value of mirna_name from the three data of hmdd, dbdemc and mir2disease. If it exists,
Just in the next column of this number add hmdd,dbdemc and mir2disease.
    :return:
    """
    dbdemc_path = './dataSet/out_data/case_study/verify/dbDEMC.txt'
    mir2_path = './dataSet/out_data/case_study/verify/mir2disease.txt'

    hmdd_list, dbdemc_list, mir2_list = [], [], []
    if name == 'Breast Neoplasms':
        hmdd_path = './dataSet/out_data/case_study/verify/hmdd/Breast Neoplasms.txt'
        pd_dbdemc = open(dbdemc_path, 'r', encoding='utf-8')
        name = 'breast'
        for _dbdemc in pd_dbdemc.readlines():
            if name in _dbdemc:
                temp_dbdemc = _dbdemc.split('\t')[0]
                dbdemc_list.append(temp_dbdemc[:12].strip())

        pd_mir2 = open(mir2_path, 'r', encoding='utf-8')
        for _mir2 in pd_mir2.readlines():
            if name in _mir2:
                temp_mir2 = _mir2.split('\t')[0]
                mir2_list.append(temp_mir2[:12].strip())
    elif name == 'heart failure':
        hmdd_path = './dataSet/out_data/case_study/verify/hmdd/heart failure.txt'
        pd_dbdemc = open(dbdemc_path, 'r', encoding='utf-8')
        name = 'heart'
        for _dbdemc in pd_dbdemc.readlines():
            if name in _dbdemc:
                temp_dbdemc = _dbdemc.split('\t')[0]
                dbdemc_list.append(temp_dbdemc[:12].strip())

        pd_mir2 = open(mir2_path, 'r', encoding='utf-8')
        for _mir2 in pd_mir2.readlines():
            if name in _mir2:
                temp_mir2 = _mir2.split('\t')[0]
                mir2_list.append(temp_mir2[:12].strip())
    else:
        hmdd_path = './dataSet/out_data/case_study/verify/hmdd/Lymphoma.txt'
        # todo:Obtain the mirnas corresponding to diseases in dbdemc
        pd_dbdemc = open(dbdemc_path, 'r', encoding='utf-8')
        name = 'lymphoma'
        for _dbdemc in pd_dbdemc.readlines():
            temp_dbdemc = _dbdemc.split('\t')[0]
            dbdemc_list.append(temp_dbdemc[:12].strip())
        # todo:The corresponding mirnas in Mir2disease were obtained
        pd_mir2 = open(mir2_path, 'r', encoding='utf-8')
        for _mir2 in pd_mir2.readlines():
            if name in _mir2:
                temp_mir2 = _mir2.split('\t')[0]
                mir2_list.append(temp_mir2[:12].strip())

    # todo:Read the three pieces of data separately

    pd_hmdd = open(hmdd_path, 'r', encoding='utf-8')
    for _hmdd in pd_hmdd.readlines():
        _hmdd = _hmdd.replace(u'\ue232', ' ')
        _hmdd = _hmdd.replace(u'\xa0', ' ')
        _hmdd = _hmdd.replace(u'\u2217', ' ')
        _hmdd = _hmdd.replace(u'\u223c', ' ')
        _hmdd = _hmdd.replace(u'\u2011', ' ')

        print(_hmdd)
        hmdd_list.append(_hmdd.split('\t')[0])
    pd_hmdd.close()

    # todo:Read the data to be validated
    pd_data = pd.read_csv(path)
    mark_list = []
    for _data in pd_data[['name']].values:
        if _data[0] in hmdd_list:
            hmdd_mark = 'hmdd'
        else:
            hmdd_mark = 'unfinded'
        if _data[0] in dbdemc_list:
            dbdemc_mark = 'dbdemc'
        else:
            dbdemc_mark = 'unfinded'
        if _data[0] in mir2_list:
            mir2_mark = 'mir2disease'
        else:
            mir2_mark = 'unfinded'
        mark_list.append(hmdd_mark + ';' + dbdemc_mark + ';' + mir2_mark)
    pd_mi = pd.DataFrame(mark_list)
    pd_data.insert(loc=6, column='evidence', value=pd_mi)
    pd_data.to_csv('./dataSet/out_data/case_study/{}_verify.csv'.format(name))


def get_dict_list(best_dict):
    """
    Rewrites the values of value in the dictionary data and puts them into an inverted list, then returns the data
    :param best_pred_dict:
    :return:
    """
    res_list = []
    for _data in best_dict:
        res_list.append(_data[1])
    return res_list


if __name__ == '__main__':
    cal_data()
    # case_data_mark('./dataSet/out_data/case_study/Breast Neoplasms.csv', "Breast Neoplasms")
    # case_data_mark('./dataSet/out_data/case_study/heart failure.csv', "heart failure")
    # case_data_mark('./dataSet/out_data/case_study/Lymphoma.csv', "lymphoma")
    # case_study('dataSet/case study/Breast Neoplasms.csv','Breast Neoplasms')
    # case_study('dataSet/case study/Lung Neoplasms.csv','Lung Neoplasms')
    # case_study('dataSet/case study/Lymphoma.csv','Lymphoma')
