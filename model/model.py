#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:model.py
@date:2022/12/22 16:32
@desc:'The model layer,
    train data：
        Put the above 3D data into transformer+mpl for training.'
"""
import math

from torch.nn import TransformerEncoder, LayerNorm
from torch.nn import TransformerDecoder
import torch
import torch.nn as nn

eps = 1e-7


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, layer_norm_eps=1e-5):  # 512,256
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)
        self.d_model = d_model

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, layer_norm_eps=1e-5):
        super(TransformerDecoderLayer, self).__init__()
        self.linear = nn.Linear(2, 256)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # memory = torch.LongTensor(memory.numpy())
        # memory = self.linear(memory)
        tgt = tgt.cuda()
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TDSMDA(nn.Module):
    def __init__(self, d_model, chanel, nhead, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.2, layer_norm_eps=1e-5):
        super(TDSMDA, self).__init__()
        encoder_layer = TransformerEncoderLayer(chanel, nhead, dim_feedforward=2048, dropout=dropout,
                                                layer_norm_eps=layer_norm_eps)
        encoder_norm = LayerNorm(chanel, eps=layer_norm_eps, )
        self.encoder1 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(chanel, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_norm = LayerNorm(chanel, eps=layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        decoder_layers1 = TransformerDecoderLayer(d_model=chanel, nhead=nhead, dim_feedforward=dim_feedforward,
                                                  dropout=dropout)
        decoder_norm = LayerNorm(chanel, eps=layer_norm_eps)
        self.decoder1 = TransformerDecoder(decoder_layers1, num_decoder_layers, decoder_norm)

        decoder_layers2 = TransformerDecoderLayer(d_model=chanel, nhead=nhead, dim_feedforward=2048,
                                                  dropout=dropout)  # 2048
        decoder_norm = LayerNorm(chanel, eps=layer_norm_eps)
        self.decoder2 = TransformerDecoder(decoder_layers2, num_decoder_layers, decoder_norm)

        self.chanel = chanel
        self.nhead = nhead
        self.d_model = d_model
        self.fcn = nn.Sequential(nn.Linear(2 * chanel, chanel), nn.ReLU())

        self.fcn2 = nn.Sequential(nn.Linear(chanel, 2 * chanel), nn.ReLU(),
                                  nn.Linear(2 * chanel, 2), nn.Sigmoid())
        # nn.Linear(128, 2), nn.Sigmoid())
        self.classific = nn.Sequential(
            nn.Linear(chanel, 2),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 2),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def encode(self, src):
        src = src.cuda()
        # en_c = en_c.cuda()
        # src1 = torch.cat((src, en_c), dim=2)
        src2 = src * int(math.sqrt(self.d_model))
        # src3 = self.fcn(src2)
        memory = self.encoder1(src2)
        # tgt = tgt.repeat(1, 1, 2)
        return memory

    def forward(self, md):
        """
        Direct use of transformer model, only encoder and decoder composition
        :param src: It's the data from the coder
        :param tgt: The data of the decoder
        :return:
        """

        # c = torch.zeros_like(md)
        c = torch.zeros(md.shape).cuda()
        md_memory = self.encode(md)
        md_memory = md_memory.cuda()
        output1 = self.decoder1(md, md_memory)
        output1 = output1.cuda()
        md = md.cuda()
        c = (output1 - md) ** 26
        # output2 = self.encode(md)
        # output3 = self.decoder2(md, output1)
        output3 = self.decoder2(md, md_memory)
        output4 = self.classific(output3)

        return output4


if __name__ == '__main__':
    pass
