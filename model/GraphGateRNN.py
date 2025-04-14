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
   Date      	           Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""
import torch.nn.functional as F
import torch
import torch.nn as nn

from model.DynamicGraph import DynamicGraphGenerate
from model.GCN import GCN


class GraphGateRNN(nn.Module):
    '''
    RNNCell,包含图生成模块和多图GCN
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

        if self.fusion_mode=='mix':
            # 周期性的流量和时间联合处理
            self.input_FC = nn.Linear((in_channels+time_channels), hidden_channels)
            self.fusion_x_time = nn.Linear(hidden_channels*(num_of_weeks+num_of_days+num_of_hours), hidden_channels)
            # 门控
            self.gate_FC1 = nn.Linear(hidden_channels, hidden_channels)
            self.info_FC1 = nn.Linear(hidden_channels, hidden_channels)

        elif self.fusion_mode=='split':
            # 周期性分开处理
            self.week_x_FC = nn.Linear(in_channels*num_of_weeks, hidden_channels)
            self.week_time_FC = nn.Linear(time_channels*num_of_weeks, hidden_channels)
            self.week_FC = nn.Linear(hidden_channels*2, hidden_channels)

            self.day_x_FC = nn.Linear(in_channels*num_of_days, hidden_channels)
            self.day_time_FC = nn.Linear(time_channels*num_of_weeks, hidden_channels)
            self.day_FC = nn.Linear(hidden_channels*2, hidden_channels)

            self.hour_x_FC = nn.Linear(in_channels*num_of_hours, hidden_channels)
            self.hour_time_FC = nn.Linear(time_channels*num_of_weeks, hidden_channels)
            self.hour_FC = nn.Linear(hidden_channels*2, hidden_channels)

            # 门控
            self.fusion_x_time2 = nn.Linear(hidden_channels*3, hidden_channels)
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
        # RNN门控图卷积
        self.GCN_update1 = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_update2 = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_reset1 = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_reset2 = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_cell1 = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_cell2 = GCN(hidden_channels*2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.layerNorm = nn.LayerNorm([self.hidden_channels])

    def input_process(self, x, x_time, fusion_mode):

        # 当x_time 为空的时候说明是第二个RNNCell，不需要对输入做处理
        if x_time == None:                # 当第二个cell是没有时间维度
            return x

        # 第一个cell有时间，并且时间维度是各个周期之和
        batch_size, node_num, _, in_channels = x.shape

        if fusion_mode=='mix':
            # 流量特征和时间特征统一处理
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
            x = x + torch.mul(gate_x, info)               # 哈达玛积
            x = self.dropout(x)

        elif fusion_mode=='split':
            # 流量特征和时间特征分开处理
            week_feature = x[:, :, :self.num_of_weeks, :]
            week_time = x_time[:, :, :self.num_of_weeks, :]

            day_feature = x[:, :, self.num_of_weeks:self.num_of_weeks+self.num_of_days, :]
            day_time = x_time[:, :, self.num_of_weeks:self.num_of_weeks+self.num_of_days, :]

            hour_feature = x[:, :, self.num_of_weeks + self.num_of_days:self.num_of_weeks + self.num_of_days + self.num_of_hours, :]
            hour_time = x_time[:, :, self.num_of_weeks + self.num_of_days:self.num_of_weeks + self.num_of_days + self.num_of_hours, :]

            # 不同时间的分开处理，再concat
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

            # gate机制
            residual = x
            gate_x = torch.sigmoid(self.gate_FC2(residual))
            info = torch.tanh(self.info_FC2(residual))
            x = x + torch.mul(gate_x, info)               # 哈达玛积
            x = self.dropout(x)

        return x

    def forward(self, input, input_time, Hidden_State, encoder_hidden=None):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        x = input
        x_time = input_time
        if x_time == None:  # When second cell has no time dimension
            batch_size, node_num, in_channels = input.shape
        else:  # First cell has time, and time dimension is the sum of cycles
            batch_size, node_num, _, in_channels = input.shape

        x = self.input_process(x, x_time, self.fusion_mode)

        # Debug the Hidden_State shape
        print(f"Hidden_State shape: {Hidden_State.shape}, size: {Hidden_State.numel()}")
        print(f"Expected shape product: {batch_size * node_num * self.hidden_channels}")
        
        # Check if Hidden_State has the proper dimensions
        if Hidden_State.numel() != batch_size * node_num * self.hidden_channels:
            # If dimensions don't match, initialize with correct shape
            print(f"WARNING: Hidden_State size mismatch. Reinitializing with correct shape.")
            Hidden_State = torch.zeros(batch_size, node_num, self.hidden_channels, device=DEVICE)
        else:
            # Reshape if needed
            if Hidden_State.dim() == 2:  # If it's [batch_size*node_num, hidden_channels]
                Hidden_State = Hidden_State.view(batch_size, node_num, -1)
            elif Hidden_State.dim() == 3:  # If it's already [batch_size, node_num, channels]
                pass  # No need to reshape
            else:
                # Handle other cases or raise meaningful error
                raise ValueError(f"Unexpected Hidden_State shape: {Hidden_State.shape}. Expected dimensions: 2 or 3.")

        # Make sure Hidden_State is on the right device
        Hidden_State = Hidden_State.to(DEVICE)
        x = x.to(DEVICE)

        # Process the input
        if encoder_hidden is not None:
            Hidden_State = Hidden_State + encoder_hidden

        combined = torch.cat((x, Hidden_State), -1)

        # Graph generation gate - different graph for different batches, dimension [batch, node_num, node_num]
        dyn_norm_adj, dyn_adj = self.dynGraph(x, Hidden_State)
        dyn_norm_adjT = dyn_norm_adj.transpose(1, 2)

        # Multiple graph collection
        norm_adjs = [adj for adj in self.static_norm_adjs]
        norm_adjTs = [adj.T for adj in self.static_norm_adjs]

        # Update gate using diffusion convolution
        update_gate = torch.sigmoid(self.GCN_update1(combined, norm_adjs, dyn_norm_adj) +
                                    self.GCN_update2(combined, norm_adjTs, dyn_norm_adjT))

        # Reset gate
        reset_gate = torch.sigmoid(self.GCN_reset1(combined, norm_adjs, dyn_norm_adj) +
                                self.GCN_reset2(combined, norm_adjTs, dyn_norm_adjT))

        # Generate candidate set
        temp = torch.cat((x, torch.mul(reset_gate, Hidden_State)), -1)
        Cell_State = torch.tanh(self.GCN_cell1(temp, norm_adjs, dyn_norm_adj) +
                                self.GCN_cell2(temp, norm_adjTs, dyn_norm_adjT))  # Current temporary hidden layer state

        # Update hidden representation
        next_Hidden_State = torch.mul(update_gate, Hidden_State) + torch.mul(1.0 - update_gate, Cell_State)

        # Channel dimension normalization
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
        Applies zoneout regularization
        """
        if training:
            d = torch.zeros_like(next_h).bernoulli_(rate)
            next_h = d * prev_h + (1 - d) * next_h
        else:
            next_h = rate * prev_h + (1 - rate) * next_h

        return next_h