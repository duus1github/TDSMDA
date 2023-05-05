#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:processing.py
@date:2022/12/22 13:49
@desc:'This module converts raw data, for example, the associated data in inital_data into matrix data.'
"""

import numpy as np

from config.conf import Config
import pandas as pd
import json
from data_pre.graph2vec import graph2vec_main

conf = Config()


def find_name(i, martix):
    """
    Enter a value to see if it is included in the list
    :param i: search value
    :param martix: list data
    :return:
    """
    for data in martix:
        if i == data:
            return True


def change_2_matrix(file_name, row_file, col_file, res_file):
    """
    The correlation pair relation is transformed into the matrix correlation relation, the correlation =1, the unrelated =0
    :param file_name:
    :return:matrix
    """
    pd_data = pd.read_csv(file_name, header=None)
    row_data = pd.read_csv(row_file, error_bad_lines=False, header=None)
    col_data = pd.read_csv(col_file, error_bad_lines=False, header=None)
    len_row = len(row_data)
    len_col = len(col_data)
    pd_row = pd_data.T.values[0]
    pd_col = pd_data.T.values[1]

    res_matrix = np.eye(len_row, len_col)
    # print(pd_data)
    for i in pd_row:
        for j in pd_col:
            # print(i,type(i))
            # print(j,type(j))
            res_matrix[i][j] = 1
            # time.sleep(50)
    print(res_matrix)
    np.save(res_file, res_matrix)
    return res_matrix


def change_2_graph(index, seq):
    """
    Enter the seq sequence data and then use k-mer to get the graph of DAG in 3,2,1 step size respectively
    :param seq:UGCCAGUCUC
    :return:
    """
    res_dict = {}
    for step in range(1, 4):
        step_dict = {}
        edges_list = []
        features_dict = {}
        for i in range(0, len(seq)):
            seq_1 = seq[i:i + step]
            temp = [seq_1, seq_1]
            edges_list.append(temp)
            features_dict[i] = seq_1
            step_dict["edges"] = edges_list
            step_dict["feature"] = features_dict

            with open(f'../dataSet/input_data/mir_dag/{step}/{index}.json', 'w+') as f:
                f.write(json.dumps(step_dict))
    print(index, 'write finish')


def pre_mriseq():
    """
    Process data for miRNA_seq
    :return:
    """
    # todo:Here is the first seq data processing into dictionary format data
    pd_mir_seq = pd.read_csv(conf.mir_seq_path)
    print(pd_mir_seq)
    # for index in range(0, pd_mir_seq.shape[0]):
    #     seq = pd_mir_seq.mirSeq[index]
    #     change_2_graph(index,seq)
    # todo:Call graph2vec_main to generate the data
    graph2vec_main()
    print('End of file processing')


if __name__ == '__main__':
    pre_mriseq()
    # change_2_matrix(conf.init_pd_path,conf.init_p_name_path,conf.init_d_name_path,conf.input_pd_path)
