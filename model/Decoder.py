# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : Decoder.py
Project      : AAAI2023-STNSCN
Created Date : 2021/8/28 13:52
Author       : Yu Zhao 
Email        : yzhao@buaa.edu.cn
==================================================================
Description  : Decoder module for the Spatial-Temporal Neural Network
               with Structural Context Network (STNSCN)
==================================================================
TODO List:
   Date                       Comments                        Finish
   
   
   
---------   --------------------------------------------   -------   
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from model.GraphGateRNN import GraphGateRNN


class Decoder(nn.Module):

    def __init__(self, in_channels, time_channels, hidden_channels, output_channels, gcn_depth, alpha,
                 num_of_weeks, num_of_days, num_of_hours, num_for_predict, dropout_prob,
                 dropout_type, fusion_mode,
                 node_num,
                 static_norm_adjs,
                 norm,
                 use_curriculum_learning, cl_decay_steps):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout_prob)
        self.w_length = num_of_weeks * num_for_predict
        self.d_length = num_of_days * num_for_predict
        self.h_length = num_of_hours * num_for_predict

        self.static_norm_adjs = static_norm_adjs

        self.in_channels = in_channels
        self.time_channels = time_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.RNN_layer = 1

        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps

        self.RNNCell = nn.ModuleList([
                                         GraphGateRNN(in_channels,
                                                      time_channels,
                                                      hidden_channels,
                                                      dropout_type=dropout_type,
                                                      gcn_depth=gcn_depth,
                                                      alpha=alpha,
                                                      num_of_weeks=0,
                                                      num_of_days=0,
                                                      num_of_hours=1,
                                                      dropout_prob=dropout_prob,
                                                      node_num=node_num,
                                                      fusion_mode=fusion_mode,
                                                      norm=norm,
                                                      static_norm_adjs=static_norm_adjs)
                                     ])

        self.fc_final = nn.Linear(self.hidden_channels, self.output_channels)

    def forward(self, decoder_input, target_time, target_cl, Hidden_State, task_level=2, global_step=None):
        '''
        Forward pass of the decoder model

        :param decoder_input: [batch_size, node_num, time_len=1, in_channels]
        :param target_time:   [batch_size, node_num, num_for_target, time_channels]
        :param target_cl:     [batch_size, node_num, num_for_target, in_channels]
        :param Hidden_State:  [batch_size, RNN_layer, node_num, hidden_channels] or
                              [batch_size, RNN_layer, node_num, num_pred, hidden_channels]
        :param task_level:    <= num_for_target
        :param global_step:   Used to adjust whether the decoder's input is the ground truth label
        :return: [batch_size, node_num, num_for_target, out_channels]
        '''
        batch_size, node_num, time_len, dim = decoder_input.shape
        Hidden_State = [Hidden_State[:, l, :, :] for l in range(self.RNN_layer)]

        outputs_final = []

        # Curriculum learning: gradually increase input sequence length and use label as decoder input with certain probability
        # Otherwise, use the previous time step's decoder output as the input for the next time step
        for i in range(task_level):
            # Initial time input
            cur_time = target_time[:, :, i:i + 1, :]

            for j, rnn_cell in enumerate(self.RNNCell):
                # Note: decoder input dimension is [batch, node_num, 1, output_channels]
                cur_h = Hidden_State[j]
                cur_out, cur_h = rnn_cell(decoder_input, cur_time, cur_h)

                Hidden_State[j] = cur_h
                decoder_input = F.relu(cur_out, inplace=True)
                # decoder_input = cur_out
                cur_time = None

            # Decoder output after FC becomes the final result
            decoder_output = self.fc_final(cur_out)

            # In decoder, the output of the previous time step becomes the input for the next time step
            decoder_input = decoder_output.view(batch_size, node_num, 1, self.output_channels)

            # Collect each decoder output result
            outputs_final.append(decoder_output)

            # Curriculum learning: with certain probability, use the ground truth as the decoder input
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                prob = self._compute_sampling_threshold(global_step)
                if global_step < self.cl_decay_steps:
                    prob = 0.5
                if c < prob:
                    # target_cl = [batch, node_num, num_for_target, 2]
                    decoder_input = target_cl[:, :, i:i + 1, :]

        # [batch, num_node, num_for_target/task_level, output_dim]
        outputs_final = torch.stack(outputs_final, dim=2)

        outputs_final = outputs_final.view(batch_size, node_num, task_level, self.output_channels)

        del Hidden_State

        return outputs_final

    def _compute_sampling_threshold(self, global_step):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(global_step / self.cl_decay_steps))