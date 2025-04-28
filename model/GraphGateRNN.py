# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : GraphGateRNN.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/27 20:37
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :

==================================================================
TODO List:
   Date                     Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""
import torch.nn.functional as F
import torch
import torch.nn as nn

from model.DynamicGraph import DynamicGraphGenerate
from model.GCN import GCN


class GraphGateRNN(nn.Module):
    '''
    RNN Cell containing a dynamic graph generation module and multi-graph GCN.
    '''

    def __init__(self, in_channels,
                 time_channels,
                 hidden_channels,
                 dropout_type='zoneout',
                 gcn_depth=2,
                 num_of_weeks=1,
                 num_of_days=1,
                 num_of_hours=1,
                 dropout_prob=0.3,
                 fusion_mode=2,
                 node_num=54,
                 static_norm_adjs=None,
                 alpha=1,
                 norm='D-1'):
        super(GraphGateRNN, self).__init__()

        self.in_channels = in_channels
        self.time_channels = time_channels
        self.hidden_channels = hidden_channels

        self.fusion_mode = fusion_mode
        self.dropout_type = dropout_type

        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.num_of_hours = num_of_hours

        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        if self.fusion_mode == 'mix':
            # Process periodic flow and time features jointly
            self.input_FC = nn.Linear((in_channels + time_channels), hidden_channels)
            self.fusion_x_time = nn.Linear(hidden_channels * (num_of_weeks + num_of_days + num_of_hours), hidden_channels)
            # Gating mechanism
            self.gate_FC1 = nn.Linear(hidden_channels, hidden_channels)
            self.info_FC1 = nn.Linear(hidden_channels, hidden_channels)

        elif self.fusion_mode == 'split':
            # Process periodic features separately
            self.week_x_FC = nn.Linear(in_channels * num_of_weeks, hidden_channels)
            self.week_time_FC = nn.Linear(time_channels * num_of_weeks, hidden_channels)
            self.week_FC = nn.Linear(hidden_channels * 2, hidden_channels)

            self.day_x_FC = nn.Linear(in_channels * num_of_days, hidden_channels)
            self.day_time_FC = nn.Linear(time_channels * num_of_days, hidden_channels)
            self.day_FC = nn.Linear(hidden_channels * 2, hidden_channels)

            self.hour_x_FC = nn.Linear(in_channels * num_of_hours, hidden_channels)
            self.hour_time_FC = nn.Linear(time_channels * num_of_hours, hidden_channels)
            self.hour_FC = nn.Linear(hidden_channels * 2, hidden_channels)

            # Gating mechanism
            self.fusion_x_time2 = nn.Linear(hidden_channels * 3, hidden_channels)
            self.gate_FC2 = nn.Linear(hidden_channels, hidden_channels)
            self.info_FC2 = nn.Linear(hidden_channels, hidden_channels)

        self.dynGraph = DynamicGraphGenerate(hidden_channels,
                                             hidden_channels,
                                             dropout_prob,
                                             node_num=node_num,
                                             reduction=16,
                                             alpha=alpha,
                                             norm=norm)
        self.static_norm_adjs = static_norm_adjs
        # RNN gated graph convolutions
        self.GCN_update1 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_update2 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_reset1 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_reset2 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_cell1 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_cell2 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.layerNorm = nn.LayerNorm([self.hidden_channels])

    def input_process(self, x, x_time, fusion_mode):
        # If x_time is None, this is the second RNN cell, no need for input processing
        if x_time is None:
            return x

        batch_size, node_num, _, in_channels = x.shape

        if fusion_mode == 'mix':
            # Process flow and time features jointly
            x = torch.split(x, 1, dim=2)
            time = torch.split(x_time, 1, dim=2)

            fusion = [torch.cat([x_i, time_i], dim=-1) for x_i, time_i in zip(x, time)]
            x = torch.cat(fusion, dim=2)

            x = self.input_FC(x)
            x = x.reshape(batch_size, node_num, -1)
            x = self.fusion_x_time(x)

            residual = x
            gate_x = torch.sigmoid(self.gate_FC1(residual))
            info = torch.tanh(self.info_FC1(residual))
            x = x + torch.mul(gate_x, info)  # Hadamard product
            x = self.dropout(x)

        elif fusion_mode == 'split':
            # Process flow and time features separately
            week_feature = x[:, :, :self.num_of_weeks, :]
            week_time = x_time[:, :, :self.num_of_weeks, :]

            day_feature = x[:, :, self.num_of_weeks:self.num_of_weeks + self.num_of_days, :]
            day_time = x_time[:, :, self.num_of_weeks:self.num_of_weeks + self.num_of_days, :]

            hour_feature = x[:, :, self.num_of_weeks + self.num_of_days:self.num_of_weeks + self.num_of_days + self.num_of_hours, :]
            hour_time = x_time[:, :, self.num_of_weeks + self.num_of_days:self.num_of_weeks + self.num_of_days + self.num_of_hours, :]

            week_feature = self.week_x_FC(week_feature.reshape(batch_size, node_num, -1))
            week_time = self.week_time_FC(week_time.reshape(batch_size, node_num, -1))
            week = F.relu(self.week_FC(torch.cat([week_feature, week_time], dim=-1)), inplace=True)

            day_feature = self.day_x_FC(day_feature.reshape(batch_size, node_num, -1))
            day_time = self.day_time_FC(day_time.reshape(batch_size, node_num, -1))
            day = F.relu(self.day_FC(torch.cat([day_feature, day_time], dim=-1)), inplace=True)

            hour_feature = self.hour_x_FC(hour_feature.reshape(batch_size, node_num, -1))
            hour_time = self.hour_time_FC(hour_time.reshape(batch_size, node_num, -1))
            hour = F.relu(self.hour_FC(torch.cat([hour_feature, hour_time], dim=-1)), inplace=True)

            x = torch.cat([week, day, hour], dim=-1)
            x = self.fusion_x_time2(x)

            residual = x
            gate_x = torch.sigmoid(self.gate_FC2(residual))
            info = torch.tanh(self.info_FC2(residual))
            x = x + torch.mul(gate_x, info)  # Hadamard product
            x = self.dropout(x)

        return x

    def forward(self, input, input_time, Hidden_State, encoder_hidden=None):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        x = input
        x_time = input_time
        if x_time is None:
            batch_size, node_num, in_channels = input.shape
        else:
            batch_size, node_num, _, in_channels = input.shape

        x = self.input_process(x, x_time, self.fusion_mode)
        if Hidden_State.numel() != batch_size * node_num * self.hidden_channels:
            Hidden_State = torch.zeros(batch_size, node_num, self.hidden_channels, device=DEVICE)
        else:
            if Hidden_State.dim() == 2:
                Hidden_State = Hidden_State.view(batch_size, node_num, -1)
            elif Hidden_State.dim() == 3:
                pass
            else:
                raise ValueError(f"Unexpected Hidden_State shape: {Hidden_State.shape}. Expected dimensions: 2 or 3.")

        Hidden_State = Hidden_State.to(DEVICE)
        x = x.to(DEVICE)

        if encoder_hidden is not None:
            Hidden_State = Hidden_State + encoder_hidden

        combined = torch.cat((x, Hidden_State), -1)

        dyn_norm_adj, dyn_adj = self.dynGraph(x, Hidden_State)
        dyn_norm_adjT = dyn_norm_adj.transpose(1, 2)

        norm_adjs = [adj for adj in self.static_norm_adjs]
        norm_adjTs = [adj.T for adj in self.static_norm_adjs]

        update_gate = torch.sigmoid(self.GCN_update1(combined, norm_adjs, dyn_norm_adj) +
                                    self.GCN_update2(combined, norm_adjTs, dyn_norm_adjT))

        reset_gate = torch.sigmoid(self.GCN_reset1(combined, norm_adjs, dyn_norm_adj) +
                                   self.GCN_reset2(combined, norm_adjTs, dyn_norm_adjT))

        temp = torch.cat((x, torch.mul(reset_gate, Hidden_State)), -1)
        Cell_State = torch.tanh(self.GCN_cell1(temp, norm_adjs, dyn_norm_adj) +
                                self.GCN_cell2(temp, norm_adjTs, dyn_norm_adjT))

        next_Hidden_State = torch.mul(update_gate, Hidden_State) + torch.mul(1.0 - update_gate, Cell_State)

        next_hidden = self.layerNorm(next_Hidden_State)

        output = next_hidden
        if self.dropout_type == 'zoneout':
            next_hidden = self.zoneout(prev_h=Hidden_State,
                                       next_h=next_hidden,
                                       rate=self.dropout_prob,
                                       training=self.training)

        return output, next_hidden

    def zoneout(self, prev_h, next_h, rate, training=True):
        """
        Applies zoneout regularization.
        """
        if training:
            d = torch.zeros_like(next_h).bernoulli_(rate)
            next_h = d * prev_h + (1 - d) * next_h
        else:
            next_h = rate * prev_h + (1 - rate) * next_h

        return next_h