#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:conf.py
@time:2022/08/18
@desc:'存放一些项目的配置信息'

"""
import os.path


class Config(object):
    base_path = os.path.dirname(os.path.dirname(__file__))

    init_mp_path = base_path + '/dataSet/inital_data/m_p.txt'
    init_pd_path = base_path + '/dataSet/inital_data/p_d.txt'
    init_pp_path = base_path + '/dataSet/inital_data/p_p_out.txt'
    init_m_name_path = base_path+'/dataSet/inital_data/m_name.txt'
    init_d_name_path = base_path+'/dataSet/inital_data/d_name.txt'
    init_p_name_path = base_path+'/dataSet/inital_data/p_name.txt'
    mir_seq_path = base_path + '/dataSet/inital_data/mirSequence.csv'

    input_md_case_path = base_path+'/dataSet/input_data/m-d-case.csv'
    input_md_path = base_path+'/dataSet/input_data/m-d.csv'
    input_sd_path = base_path+'/dataSet/input_data/s_d.csv'
    input_sm_path = base_path+'/dataSet/input_data/s_m.csv'
    input_mp_path = base_path+'/dataSet/input_data/m_p'
    input_pd_path = base_path+'/dataSet/input_data/p_d'
    input_mpd_path = base_path+'/dataSet/input_data/mpd'
    input_mir_path = base_path + '/dataSet/out_data/mir_dag/'

    md_data_path =base_path+'/dataSet/out_data/bio_graph_notpca/md_data.npy'
    md_label_path =base_path+'/dataSet/out_data/bio_graph_notpca/md_label.npy'
    m_d_path =base_path+'/dataSet/out_data/bio_graph_notpca/m_d.npy'

    mp_data_path =base_path+'/dataSet/out_data/bio_graph_notpca/mp_data.npy'
    mp_label_path =base_path+'/dataSet/out_data/bio_graph_notpca/mp_label.npy'
    m_p_path =base_path+'/dataSet/out_data/bio_graph_notpca/m_p.npy'

    pd_data_path =base_path+'/dataSet/out_data/bio_graph_notpca/pd_data.npy'
    pd_label_path =base_path+'/dataSet/out_data/bio_graph_notpca/pd_label.npy'
    p_d_path =base_path+'/dataSet/out_data/bio_graph_notpca/pd.npy'

    mm_path =base_path+'/dataSet/out_data/bio_graph_notpca/mm.npy'
    dd_path =base_path+'/dataSet/out_data/bio_graph_notpca/dd.npy'
    pp_path= base_path+'/dataSet/out_data/bio_graph_notpca/pp.npy'

if __name__ == '__main__':
    cn = Config()
