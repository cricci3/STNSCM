# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : GCN.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/27 16:07
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Description  : Graph Convolutional Network implementation for the
               Spatial-Temporal Neural Network with Structural
               Context Network (STNSCN)
==================================================================
TODO List:
   Date                       Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class dyn_gconv(nn.Module):
    def __init__(self):
        super(dyn_gconv, self).__init__()

    def forward(self, A, x):
        # Graph convolution to replace linear transformation in RNN
        x = torch.einsum('bhw,bwc->bhc', (A, x))
        return x.contiguous()


class static_gconv(nn.Module):
    def __init__(self):
        super(static_gconv, self).__init__()

    def forward(self, A, x):
        x = torch.einsum('hw,bwc->bhc', (A, x))
        return x.contiguous()


class gconv(nn.Module):

    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, A, x):
        x = torch.einsum('hw, bwtc->bhtc', (A, x))
        return x.contiguous()


class linear(nn.Module):
    '''Using 1D convolution to replace linear transformation'''

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out)

    def forward(self, x):

        return F.relu(self.mlp(x), inplace=True)

class GCN(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout_prob, graph_num, type=None):
        super(GCN, self).__init__()
        if type == 'RNN':
            self.dyn_gconv = dyn_gconv()
            self.static_gconv = static_gconv()
            self.mlp = linear((gdep + 1) * c_in, c_out)

            # Create weight parameters based on the number of graphs
            # graph_num+1+1: +1 for original input weight, +1 for dynamic graph
            self.weight = nn.Parameter(torch.FloatTensor(graph_num+1+1), requires_grad=True)
            self.weight.data.fill_(1.0)

        elif type == 'common':
            self.gconv = gconv()
            self.mlp = linear((gdep + 1) * c_in, c_out)

            # +1 for the weight parameter needed for input
            self.weight = nn.Parameter(torch.FloatTensor(graph_num+1), requires_grad=True)
            self.weight.data.fill_(1.0)


        self.dropout = nn.Dropout(dropout_prob)
        self.graph_num = graph_num
        self.gdep = gdep
        self.type = type

    def forward(self, x, norm_adj, dyn_norm_adj=None):
        h = x
        out = [h]

        weight = F.softmax(self.weight, dim=0)     # Normalize weights to probabilities
        # weight = self.weight    # Without probability normalization

        if self.type == 'RNN':
            for _ in range(self.gdep):
                h_next = weight[0] * x
                for i in range(0, len(norm_adj)):
                    h_next += weight[i+1] * self.static_gconv(norm_adj[i], h)
                if dyn_norm_adj is not None:
                    h_next += weight[-1] * self.dyn_gconv(dyn_norm_adj, h)

                h = h_next
                out.append(h)

        elif self.type == 'common':
            for _ in range(self.gdep):  # Diffusion steps
                h = self.weight[0] * x
                for i in range(1, len(norm_adj)):
                    h += self.weight[i] * self.gconv(norm_adj[i], h)
                out.append(h)

        ho = torch.cat(out, dim=-1)

        ho = self.mlp(ho)  # Process results with fully connected layer using sigmoid activation function


        return ho