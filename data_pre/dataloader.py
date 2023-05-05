#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:dataloader.py
@date:2022/12/22 15:44
@desc:used to load training data sets
    struct data：
        1、The graph is used to represent learning, and the m-m and d-d associations are used to extract features from the data
        2、Then the similarity data and the data extracted from the above features are integrated
"""
import glob
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
from karateclub import GL2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from config.conf import Config



conf = Config()


def get_train_data(data_type=None, is_case=None):
    """
    Gets the data in the input_data file and returns it in a dictionary
    :return:res_dict
    """
    res_dict = {}
    # if is_case is True:
    #     m_d = pd.read_csv(conf.input_md_case_path, header=None)
    # else:
    m_d = pd.read_csv(conf.input_md_path, header=None)
    m_p = np.load(conf.input_mp_path + '.npy')
    p_d = np.load(conf.input_pd_path + '.npy')

    # todo:similarity
    s_d = pd.read_csv(conf.input_sd_path, header=None)
    s_m = pd.read_csv(conf.input_sm_path, header=None)


    # todo:get edge data，
    dd_edge = get_edge_index(s_d)
    mm_edge = get_edge_index(s_m)
    pp_edge = pd.read_csv(conf.init_pp_path, error_bad_lines=False, header=None)
    mp_edge = pd.read_csv(conf.init_mp_path, error_bad_lines=False, header=None)
    dp_edge = pd.read_csv(conf.init_pd_path, error_bad_lines=False, header=None)
    # todo:The figure shows learning Graph2vec to extract the feature data of mirseq
    path = conf.input_mir_path
    mir_dag = glob.glob(os.path.join(path, "*.csv"))
    graph_mm = torch.zeros(495, 129)
    for _dag in mir_dag:
        mm = pd.read_csv(_dag)
        mm.sort_values('type', inplace=True)
        mm_values = torch.tensor(mm.values)
        temp_mm = np.array(mm_values)
        graph_mm = torch.add(graph_mm, mm_values)
    graph_mm = graph_mm.data[:, 1:]
    # todo:pca was used to extract the original s_d data set
    pca = PCA(n_components=128)
    pca_sd = torch.from_numpy(pca.fit_transform(s_d))

    if data_type == 'graph_bio_not_pca':
        # todo:The feature data is extracted by combining similarity data and graph representation learning
        # pca = PCA(n_components=128)
        # pca_sm = torch.from_numpy(pca.fit_transform(s_m))
        # pca_sd = torch.from_numpy(pca.fit_transform(s_d))
        sm = torch.tensor(s_m.values)
        sd = torch.tensor(s_d.values)
        mm = torch.hstack((graph_mm, sm))
        dd = torch.hstack((pca_sd, sd))
        # pp_graph = torch.hstack((pp_graph, pp_graph))
        # mm = torch.add(mm_graph, pca_sm)
        # dd = torch.add(dd_graph, pca_sd)
        # pp_graph = torch.add(pp_graph, pp_graph)
        # todo:Construct sample set, train sample set, test sample set
        md_data, md_label, case_163, case_50, case_240, \
        case_163_label, case_50_label, case_240_label = struct_train(mm, dd, m_d)

        # mp_data, mp_label = struct_p_train(mm, pp_graph, m_p)
        # pd_data, pd_label = struct_pd_train(dd, pp_graph, p_d)
        """
        # todo:save the data set 
        np.save('../dataSet/out_data/bio_graph_notpca/md_data', md_data)
        np.save('../dataSet/out_data/bio_graph_notpca/md_label', md_label)
        np.save('../dataSet/out_data/bio_graph_notpca/m_d', m_d)

        np.save('../dataSet/out_data/bio_graph_notpca/mp_data', mp_data)
        np.save('../dataSet/out_data/bio_graph_notpca/mp_label', mp_label)
        np.save('../dataSet/out_data/bio_graph_notpca/m_p', m_p)

        np.save('../dataSet/out_data/bio_graph_notpca/pd_data', pd_data)
        np.save('../dataSet/out_data/bio_graph_notpca/pd_label', pd_label)
        np.save('../dataSet/out_data/bio_graph_notpca/pd', p_d)

        np.save('../dataSet/out_data/bio_graph_notpca/mm', mm)
        np.save('../dataSet/out_data/bio_graph_notpca/dd', dd)
        np.save('../dataSet/out_data/bio_graph_notpca/pp', pp_graph)"""
        # todo:shuffle the md_data,md_label
        md_data, md_label = shuffle(md_data, md_label)
        # todo:Carry out feature engineering on feature data
        # pca_md=PCA(n_components=256)
        # case_240 = torch.FloatTensor(pca_md.fit_transform(case_240))
        # md_data =  pca_md.fit_transform(md_data)
        # case_163 =  torch.FloatTensor(pca_md.fit_transform(case_163))
        # case_50 = torch.FloatTensor(pca_md.fit_transform(case_50))
        # todo:The data is normalized
        # scaler = MinMaxScaler()
        # md_data = scaler.fit_transform(md_data)
        # md_data =  torch.FloatTensor(md_data)

        res = {'md': {'md_data': md_data, 'md_label': md_label, 'm_d': m_d,
                      'case_163': case_163, 'case_163_label': case_163_label,
                      'case_50': case_50, 'case_50_label': case_50_label,
                      'case_240': case_240, 'case_240_label': case_240_label},
               'mm': s_m, 'dd': s_d}
        # 'mp': {'mp_data': mp_data, 'mp_label.npy': mp_label, 'm_p': m_p},
        # 'pd': {'pd_data': pd_data, 'pd_label': pd_label, 'p_d': p_d}}
    # elif data_type=='graph_bio_not_pca':
    #     sm = torch.tensor(s_m.values)
    #     sd = torch.tensor(s_d.values)
    #     mm = torch.hstack((mm_graph, sm))
    #     dd = torch.hstack((dd_graph, sd))
    elif data_type == 'bio_not_pca':
        s_m = torch.tensor(s_m.values)
        s_d = torch.tensor(s_d.values)
        md_data, md_label, case_163, case_50, case_240, case_163_label, \
        case_50_label, case_240_label = struct_train(s_m, s_d, m_d)
        res = {'mm': s_m, 'dd': s_d, 'md': {'m-d': m_d, 'md_data': md_data, 'md_label': md_label}}
    else:  # only the graph:
        md_data, md_label, case_163, case_50, case_240, case_163_label, case_50_label, case_240_label = struct_train(
            graph_mm, pca_sd, m_d)
        res = {'dd': pca_sd, 'mm': graph_mm, 'md': {'md_data': md_data, 'md_label': md_label}}
    return res


def get_data(data_type=None):
    """
    get the train data
    :return:res = {'md': {'md_data': md_data, 'md_label': md_label, 'm_d': m_d},
    #        'mp': {'mp_data': mp_data, 'mp_label.npy': mp_label.npy, 'm_p': m_p},
    #        'pd': {'pd_data': pd_data, 'pd_label': pd_label, 'p_d': p_d}}
    """
    md_data = np.load(conf.md_data_path, allow_pickle=True)
    md_label = np.load(conf.md_label_path, allow_pickle=True)
    m_d = np.load(conf.m_d_path, allow_pickle=True)

    s_d = pd.read_csv(conf.input_sd_path, header=None)
    s_m = pd.read_csv(conf.input_sm_path, header=None)
    dd_edge = get_edge_index(s_d)
    mm_edge = get_edge_index(s_m)
    pp_edge = pd.read_csv(conf.init_pp_path, error_bad_lines=False, header=None)

    mp_data = np.load(conf.mp_data_path, allow_pickle=True)
    mp_label = np.load(conf.mp_label_path, allow_pickle=True)
    m_p = np.load(conf.m_p_path, allow_pickle=True)

    pd_data = np.load(conf.pd_data_path, allow_pickle=True)
    pd_label = np.load(conf.pd_label_path, allow_pickle=True)
    p_d = np.load(conf.p_d_path, allow_pickle=True)

    mm = np.load(conf.mm_path, allow_pickle=True)
    dd = np.load(conf.dd_path, allow_pickle=True)
    pp = np.load(conf.pp_path, allow_pickle=True)

    # todo:shuffle data set
    md_data, md_label = shuffle(md_data, md_label)
    res = {'md': {'md_data': md_data, 'md_label': md_label, 'm_d': m_d},
           'mp': {'mp_data': mp_data, 'mp_label': mp_label, 'm_p': m_p},
           'pd': {'pd_data': pd_data, 'pd_label': pd_label, 'p_d': p_d},
           'mm': mm, 'dd': dd, 'pp': pp}
    if data_type:
        res = {'mm': mm, 'dd': dd, 'pp': pp, 'm-d': m_d, 'mp': mp_data, 'pd': pd_data, 'md_data': md_data}

    return res


def get_mulit_data():
    mm = np.load(conf.mm_path, allow_pickle=True)
    dd = np.load(conf.dd_path, allow_pickle=True)
    pp = np.load(conf.pp_path, allow_pickle=True)
    s_d = pd.read_csv(conf.input_sd_path, header=None)
    s_m = pd.read_csv(conf.input_sm_path, header=None)

    dd_edge = get_edge_index(s_d)
    mm_edge = get_edge_index(s_m)
    pp_edge = pd.read_csv(conf.init_pp_path, error_bad_lines=False, header=None)

    md_data = pd.read_csv(conf.input_md_path, header=None)
    mp_edge = pd.read_csv(conf.init_mp_path, error_bad_lines=False, header=None)
    dp_edge = pd.read_csv(conf.init_pd_path, error_bad_lines=False, header=None)
    md_edge = get_md_index(md_data)
    np_md = np.array(md_edge)
    md_label = torch.LongTensor([1 for i in range(len(md_edge.T))])
    mp_label = torch.LongTensor([1 for i in range(len(mp_edge))])
    dp_label = torch.LongTensor([1 for i in range(len(dp_edge))])
    return {'mm': mm, 'dd': dd, 'pp': pp, 'mm_edge': mm_edge, 'dd_edge': dd_edge, 'pp_edge': pp_edge,
            'md_edge': md_edge, 'mp_edge': mp_edge, 'dp_edge': dp_edge,
            'md_lael': md_label, 'mp_label': mp_label, 'pd_label': dp_label}


class TrainSet(Dataset):
    def __init__(self, data, mode, data_type):
        inchanel = data['md']['md_data'].shape[1]
        # note：And this is 10 before, and here it's set to 2 because of case-study
        self.md_feature = data['md']['md_data'].reshape(-1, 2, inchanel)
        self.md_label = data['md']['md_label'].reshape(-1, 2, 2)

        # self.md_feature = data['md']['md_data'][:10840]
        # self.md_label = data['md']['md_label'][:10840]

        self.md_feature, self.md_label = shuffle(self.md_feature, self.md_label)

        if mode == 'train':  # 80%
            self.md_feature = self.md_feature[:int(0.7 * len(self.md_feature))]
            self.md_label = self.md_label[:int(0.7 * len(self.md_label))]

        elif mode == 'val':
            self.md_feature = self.md_feature[int(0.7 * len(self.md_feature)):int(0.8 * len(self.md_feature))]
            self.md_label = self.md_label[int(0.7 * len(self.md_label)):int(0.8 * len(self.md_label))]
        elif mode == 'test':
            self.md_feature = self.md_feature[int(0.8 * len(self.md_feature)):]
            self.md_label = self.md_label[int(0.8 * len(self.md_label)):]
        elif mode == 'case_163':
            self.md_feature = data['md']['case_163'].reshape(-1, 1, inchanel)
            self.md_label = data['md']['case_163_label'].reshape(-1, 1, 2)
        elif mode == 'case_50':
            self.md_feature = data['md']['case_50'].reshape(-1, 1, inchanel)
            self.md_label = data['md']['case_50_label'].reshape(-1, 1, 2)

        elif mode == 'case_240':
            self.md_feature = data['md']['case_240'].reshape(-1, 1, inchanel)
            self.md_label = data['md']['case_240_label'].reshape(-1, 1, 2)
        else:
            self.md_feature = data['md']['case_data'].reshape(-1, 2, inchanel)
            self.md_label = data['md']['case_label'].reshape(-1, 2, 2)
        # self.md_feature,self.md_label = shuffle(self.md_feature,self.md_label)

    def __len__(self):
        return len(self.md_label)

    def __getitem__(self, item):
        return self.md_feature[item], self.md_label[item]


class KFTrainSet(Dataset):
    def __init__(self, data, label):
        # self.data = data.reshape(-1, 10, 512)
        # self.label = label.reshape(-1, 10, 2)
        self.data, self.label = shuffle(data, label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


def get_md_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            if matrix[i][j] != 0:  # ！=0，is associate data
                edge_index[0].append(i)  # disease
                edge_index[1].append(j)  # mirna
    return torch.LongTensor(edge_index)


def get_edge_index(matrix):
    """
    Input matrix of similarity, similarity data,! =0, then add coordinates
    :param matrix:
    :return:
    """
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:  # ！=0，is associate data
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def get_p_edge(i, matrix):
    """
    get the protein edge
    :param i: Enter the index for edge 0
    :param matrix:edge data
    :return: [(1,2),(1,3),(1,150)......]
    """
    res_list = []
    for edge in matrix.values:
        if i == edge[0]:
            res_list.append(tuple(edge))
    return list(set(res_list))


def get_edges(i, matrix):
    """
    Enter m-m,d-d matrix data, enter the edges between points
    :param matrix:
    :return:
    """
    res = []

    for j in matrix.T:
        if i == j[0]:
            res.append((i, j[1]))
        else:
            break
    return res


def ex_data(data):
    """
    Feature data was extracted using GL2vec
    :return:(495/383，128)
    """
    temp_data = np.array(data)
    graphs = []
    for i in range(data.min(), data.max() + 1):
        G = nx.Graph()
        G.add_nodes_from([i for i in range(data.min(), data.max() + 1)])

        G.add_edges_from(get_edges(i, data))
        graphs.append(G)
    gl = GL2Vec()

    # print(gl)
    print('start fit.....')
    gl.fit(graphs)
    print('ending.....')
    print('start embedding.....')
    res = gl.get_embedding()  # (1,128),
    print('end embedding.....\n', res.shape)

    return torch.from_numpy(res)


def ex_p_data(data):
    """
    Using graph representation learning gl2vec to extract p-p feature data
    :param data:
    :return:
    """
    # todo:Define the number of nodes, on one side if the network is homogeneous, but on a heterogeneous network, add the maximum values of the two
    if data.max()[0] == data.max()[1]:
        max_node = data.max()[0]
        len_min = data.min()[0]
        len_max = data.max()[0]
    else:
        len_min = min(data.min())
        len_max = min(data.max())
        max_node = data.max()[0] + data.max()[1]
    temp_data = np.array(data)
    graphs = []
    for i in range(len_min, len_max + 1):
        G = nx.Graph()
        G.add_nodes_from([i for i in range(data.min()[0], max_node + 2)])

        G.add_edges_from(get_p_edge(i, data))
        graphs.append(G)
    gl = GL2Vec()

    # print(gl)
    print('start fit.....')
    gl.fit(graphs)
    print('ending.....')
    print('start embedding.....')
    res = gl.get_embedding()  # (1,128),
    print('end embedding.....\n', res.shape)

    return torch.from_numpy(res)


def struct_train(sm_1, sm_2, matrix):
    """
    The edge data is learned by using graph representation to extract features and then fused with sim data
    :param sm_1:miRNA
    :param sm_2:disease
    :return:
    """
    know, unknow1 = [], []
    # todo:Gets the edge data of relation 1
    col = matrix.shape[1]
    row = matrix.shape[0]
    for i in range(0, col):  # col
        for j in range(0, row):  # row
            if matrix[i][j] == 1:
                know.append((i, j))
            else:
                unknow1.append((i, j))
    temp_know = np.array(know)
    # todo:get data and label
    pos_list, neg_list, label = [], [], []
    case_236, case_50, case_240, case_236_label, case_50_label, case_240_label = [], [], [], [], [], []
    case_pos_236, case_pos_50, case_pos_240 = [], [], []
    mi_236_name, mi_50_name, mi_240_name = [], [], []
    # pos data set
    for k in range(0, len(know)):
        # if  is bio ,then pos = sm_1[know[k][1], :].tolist() + sm_2[know[k][0], :].tolist()
        pos = sm_1[know[k][1], :].tolist() + sm_2[know[k][0], :].tolist()
        # todo:The case study was the use of Brain Neoplasms, Breast Neoplasms, and Lymphoma
        if know[k][0] == 236:
            pos = sm_1[know[k][1], :].tolist() + sm_2[know[k][0], :].tolist()
            case_pos_236.append(pos)
            mi_236_name.append(know[k][1])
            case_236_label.append([1, 0])
        elif know[k][0] == 50:
            pos = sm_1[know[k][1], :].tolist() + sm_2[know[k][0], :].tolist()
            case_pos_50.append(pos)
            mi_50_name.append(know[k][1])
            case_50_label.append([1, 0])
        elif know[k][0] == 240:
            pos = sm_1[know[k][1], :].tolist() + sm_2[know[k][0], :].tolist()
            case_pos_240.append(pos)
            mi_240_name.append(know[k][1])
            case_240_label.append([1, 0])
        pos_list.append(pos)
        label.append([1, 0])
    # net data set
    unknow = random.sample(unknow1, len(know))
    temp_unknow = np.array(unknow)
    temp_unknow1 = np.array(unknow1)
    for n in range(5430):
        neg = sm_1[unknow[n][1], :].tolist() + sm_2[unknow[n][0], :].tolist()
        neg_list.append(neg)
        label.append([0, 1])
    train_data = pos_list + neg_list
    # todo:Take the data from the case study
    case_236_neg, case_50_neg, case_240_neg = [], [], []
    for n in range(len(unknow1)):
        if unknow1[n][0] == 236:
            neg = sm_1[unknow1[n][1], :].tolist() + sm_2[unknow1[n][0], :].tolist()
            case_236_neg.append(neg)
            mi_236_name.append(unknow1[n][1])
            case_236_label.append([0, 1])
        elif unknow1[n][0] == 50:
            neg = sm_1[unknow1[n][1], :].tolist() + sm_2[unknow1[n][0], :].tolist()
            case_50_neg.append(neg)
            mi_50_name.append(unknow1[n][1])
            case_50_label.append([0, 1])
        elif unknow1[n][0] == 240:
            neg = sm_1[unknow1[n][1], :].tolist() + sm_2[unknow1[n][0], :].tolist()
            case_240_neg.append(neg)
            mi_240_name.append(unknow1[n][1])
            case_240_label.append([0, 1])
        else:
            continue
    case_236 = case_pos_236 + case_236_neg
    case_50 = case_pos_50 + case_50_neg
    case_240 = case_pos_240 + case_240_neg
    np.savetxt('./dataSet/case study/mir236.csv', mi_236_name)
    np.savetxt('./dataSet/case study/mir50.csv', mi_50_name)
    np.savetxt('./dataSet/case study/mir240.csv', mi_240_name)
    return torch.FloatTensor(train_data), torch.FloatTensor(label), \
           torch.FloatTensor(case_236), torch.FloatTensor(case_50), torch.FloatTensor(case_240), \
           torch.FloatTensor(case_236_label), torch.FloatTensor(case_50_label), torch.FloatTensor(case_240_label)


def struct_pd_train(sm1, sm2, matrix):
    """
    The edge data is learned by using graph representation to extract features and then fused with sim data
    :param data:
    :return:
    """
    know, unknow = [], []
    # todo:Gets the edge data of relation 1
    col = matrix.shape[1]
    row = matrix.shape[0]
    for i in range(0, row):  # 列
        for j in range(0, col):  # 行
            if matrix[i][j] == 1:
                know.append((i, j))
            else:
                unknow.append((i, j))
    temp_know = np.array(know)
    # todo:get data and label
    pos_list, neg_list, label = [], [], []
    # pos data set
    kn_count, unkn_count = 0, 0
    know = random.sample(know, 5700)
    for k in range(0, 5430 + 1 + kn_count):
        try:
            pos = sm1[know[k][1], :].tolist() + sm2[know[k][0], :].tolist()
        except:
            kn_count += 1
            continue
        pos_list.append(pos)
        label.append([1, 0])
    # neg data set
    unknow = random.sample(unknow, 5700)
    temp_unknow = np.array(unknow)
    for n in range(0, 5430 + 1 + unkn_count):
        try:
            neg = sm1[unknow[n][1], :].tolist() + sm2[unknow[n][0], :].tolist()
        except:
            unkn_count += 1
            continue
        neg_list.append(neg)
        label.append([0, 1])
    train_data = pos_list + neg_list
    return torch.FloatTensor(train_data), torch.FloatTensor(label)


def struct_p_train(sm1, sm2, matrix):
    """
    The edge data is learned by using graph representation to extract features and then fused with sim data
    :param data:
    :return:
    """
    know, unknow = [], []
    # todo:Gets the edge data of relation 1
    col = matrix.shape[1]
    row = matrix.shape[0]
    for i in range(0, row):
        for j in range(0, col):
            if matrix[i][j] == 1:
                know.append((i, j))
            else:
                unknow.append((i, j))
    temp_know = np.array(know)

    pos_list, neg_list, label = [], [], []

    know = random.sample(know, 5700)
    kn_count, unkn_count = 0, 0
    for k in range(0, 5430 + 1 + kn_count):
        try:
            pos = sm1[know[k][0], :].tolist() + sm2[know[k][1], :].tolist()
        except:
            kn_count += 1
            continue
        pos_list.append(pos)
        label.append([1, 0])

    unknow = random.sample(unknow, 5700)
    temp_unknow = np.array(unknow)
    for n in range(0, 5430 + 1 + unkn_count):
        try:
            neg = sm1[unknow[n][0], :].tolist() + sm2[unknow[n][1], :].tolist()
        except:
            unkn_count += 1
            continue
        neg_list.append(neg)
        label.append([0, 1])
    train_data = pos_list + neg_list
    return torch.FloatTensor(train_data), torch.FloatTensor(label)


if __name__ == '__main__':
    get_train_data('graphbio')
    # get_mulit_data()
    # np.zeros()
    # get_train_data()
    # np.save('../dataSet/out_data/md_data', 1)
