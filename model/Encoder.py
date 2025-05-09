# Encoder with ablation flags: use_graph_gate_rnn and use_dynamic_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GraphGateRNN import GraphGateRNN
from model.GCN import GCN

class Encoder(nn.Module):
    def __init__(self, in_channels, time_channels, hidden_channels, gcn_depth, alpha,
                 num_of_weeks, num_of_days, num_of_hours, num_for_predict, dropout_prob,
                 dropout_type, fusion_mode, node_num, static_norm_adjs, norm, device,
                 use_dynamic_graph=True, use_graph_gate_rnn=True):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.time_channels = time_channels
        self.hidden_channels = hidden_channels
        self.seq_length = num_for_predict
        self.device = device

        self.w_length = num_of_weeks * num_for_predict
        self.d_length = num_of_days * num_for_predict
        self.h_length = num_of_hours * num_for_predict

        self.use_graph_gate_rnn = use_graph_gate_rnn

        self.static_norm_adjs = static_norm_adjs

        if use_graph_gate_rnn:
            self.rnn_cell = GraphGateRNN(
                in_channels,
                time_channels,
                hidden_channels,
                dropout_type,
                gcn_depth,
                num_of_weeks,
                num_of_days,
                num_of_hours,
                dropout_prob,
                fusion_mode,
                node_num,
                static_norm_adjs,
                alpha,
                norm,
                use_dynamic_graph=use_dynamic_graph
            )
        else:
            self.gcn = GCN(
                c_in=in_channels,
                c_out=hidden_channels,
                gdep=gcn_depth,
                dropout_prob=dropout_prob,
                graph_num=len(static_norm_adjs),
                type='common'
            )

    def forward(self, input, x_time, seq_length):
        batch_size, node_num, time_len, dim = input.shape
        Hidden_State = self.initHidden(batch_size, node_num, self.hidden_channels)

        week_feature = input[:, :, :self.w_length, :]
        week_time = x_time[:, :, :self.w_length, :]

        day_feature = input[:, :, self.w_length:self.w_length + self.d_length, :]
        day_time = x_time[:, :, self.w_length:self.w_length + self.d_length, :]

        hour_feature = input[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]
        hour_time = x_time[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]

        outputs = []
        hiddens = []

        for i in range(seq_length):
            input_cur = torch.cat([
                week_feature[:, :, i:i + 1, :],
                day_feature[:, :, i:i + 1, :],
                hour_feature[:, :, i:i + 1, :]
            ], dim=2)

            input_time = torch.cat([
                week_time[:, :, i:i + 1, :],
                day_time[:, :, i:i + 1, :],
                hour_time[:, :, i:i + 1, :]
            ], dim=2)

            if self.use_graph_gate_rnn:
                cur_out, Hidden_State = self.rnn_cell(input_cur, input_time, Hidden_State)
            else:
                cur_input = input_cur[:, :, 0, :].unsqueeze(2)  # shape: [B, N, 1, D]
                cur_out = self.gcn(cur_input, self.static_norm_adjs)  # shape: [B, N, 1, H]
                cur_out = cur_out.squeeze(2)  # remove time dim → shape: [B, N, H]
                Hidden_State = cur_out  # Overwrite hidden state

            outputs.append(cur_out.unsqueeze(2))
            hiddens.append(Hidden_State.unsqueeze(1).unsqueeze(3))

        outputs = torch.cat(outputs, dim=2)
        hiddens = torch.cat(hiddens, dim=3)

        return outputs, hiddens

    def initHidden(self, batch_size, num_nodes, hidden_dim):
        return torch.zeros((batch_size, num_nodes, hidden_dim), device=self.device)