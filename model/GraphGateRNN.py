# GraphGateRNN with support for disabling dynamic graph generation
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DynamicGraph import DynamicGraphGenerate
from model.GCN import GCN

class GraphGateRNN(nn.Module):
    def __init__(self, in_channels, time_channels, hidden_channels,
                 dropout_type='zoneout', gcn_depth=2,
                 num_of_weeks=1, num_of_days=1, num_of_hours=1,
                 dropout_prob=0.3, fusion_mode='mix', node_num=54,
                 static_norm_adjs=None, alpha=1, norm='D-1',
                 use_dynamic_graph=True):
        super(GraphGateRNN, self).__init__()

        self.in_channels = in_channels
        self.time_channels = time_channels
        self.hidden_channels = hidden_channels

        self.fusion_mode = fusion_mode
        self.dropout_type = dropout_type
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        self.use_dynamic_graph = use_dynamic_graph
        self.static_norm_adjs = static_norm_adjs

        if self.fusion_mode == 'mix':
            self.input_FC = nn.Linear((in_channels + time_channels), hidden_channels)
            self.fusion_x_time = nn.Linear(hidden_channels * (num_of_weeks + num_of_days + num_of_hours), hidden_channels)
            self.gate_FC1 = nn.Linear(hidden_channels, hidden_channels)
            self.info_FC1 = nn.Linear(hidden_channels, hidden_channels)
        elif self.fusion_mode == 'split':
            self.week_x_FC = nn.Linear(in_channels * num_of_weeks, hidden_channels)
            self.week_time_FC = nn.Linear(time_channels * num_of_weeks, hidden_channels)
            self.week_FC = nn.Linear(hidden_channels * 2, hidden_channels)

            self.day_x_FC = nn.Linear(in_channels * num_of_days, hidden_channels)
            self.day_time_FC = nn.Linear(time_channels * num_of_days, hidden_channels)
            self.day_FC = nn.Linear(hidden_channels * 2, hidden_channels)

            self.hour_x_FC = nn.Linear(in_channels * num_of_hours, hidden_channels)
            self.hour_time_FC = nn.Linear(time_channels * num_of_hours, hidden_channels)
            self.hour_FC = nn.Linear(hidden_channels * 2, hidden_channels)

            self.fusion_x_time2 = nn.Linear(hidden_channels * 3, hidden_channels)
            self.gate_FC2 = nn.Linear(hidden_channels, hidden_channels)
            self.info_FC2 = nn.Linear(hidden_channels, hidden_channels)

        if self.use_dynamic_graph:
            self.dynGraph = DynamicGraphGenerate(hidden_channels, hidden_channels, dropout_prob,
                                                 node_num=node_num, reduction=16, alpha=alpha, norm=norm)

        self.GCN_update1 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_update2 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_reset1 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_reset2 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_cell1 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.GCN_cell2 = GCN(hidden_channels * 2, hidden_channels, gcn_depth, dropout_prob, len(static_norm_adjs), type='RNN')
        self.layerNorm = nn.LayerNorm([hidden_channels])

    def forward(self, input, input_time, Hidden_State, encoder_hidden=None):
        batch_size, node_num, *_ = input.shape
        x = self.input_process(input, input_time, self.fusion_mode)

        if Hidden_State.shape != (batch_size, node_num, self.hidden_channels):
            Hidden_State = torch.zeros(batch_size, node_num, self.hidden_channels, device=input.device)

        if encoder_hidden is not None:
            Hidden_State = Hidden_State + encoder_hidden

        combined = torch.cat((x, Hidden_State), -1)

        if self.use_dynamic_graph:
            dyn_norm_adj, dyn_adj = self.dynGraph(x, Hidden_State)
            dyn_norm_adjT = dyn_norm_adj.transpose(1, 2)
        else:
            dyn_norm_adj = dyn_norm_adjT = None

        norm_adjs = self.static_norm_adjs
        norm_adjTs = [adj.T for adj in self.static_norm_adjs]

        update_gate = torch.sigmoid(
            self.GCN_update1(combined, norm_adjs, dyn_norm_adj) +
            self.GCN_update2(combined, norm_adjTs, dyn_norm_adjT)
        )

        reset_gate = torch.sigmoid(
            self.GCN_reset1(combined, norm_adjs, dyn_norm_adj) +
            self.GCN_reset2(combined, norm_adjTs, dyn_norm_adjT)
        )

        temp = torch.cat((x, reset_gate * Hidden_State), -1)
        cell_state = torch.tanh(
            self.GCN_cell1(temp, norm_adjs, dyn_norm_adj) +
            self.GCN_cell2(temp, norm_adjTs, dyn_norm_adjT)
        )

        next_Hidden_State = update_gate * Hidden_State + (1.0 - update_gate) * cell_state
        next_hidden = self.layerNorm(next_Hidden_State)

        if self.dropout_type == 'zoneout':
            d = torch.zeros_like(next_hidden).bernoulli_(self.dropout_prob)
            next_hidden = d * Hidden_State + (1 - d) * next_hidden

        return next_hidden, next_hidden

    def input_process(self, x, x_time, fusion_mode):
        if x_time is None:
            return x

        batch_size, node_num, _, _ = x.shape

        if fusion_mode == 'mix':
            x = torch.cat([x, x_time], dim=-1)
            x = self.input_FC(x)
            x = x.reshape(batch_size, node_num, -1)
            x = self.fusion_x_time(x)

            residual = x
            gate_x = torch.sigmoid(self.gate_FC1(residual))
            info = torch.tanh(self.info_FC1(residual))
            return x + gate_x * info

        elif fusion_mode == 'split':
            num_w, num_d, num_h = self.week_x_FC.in_features // self.in_channels, \
                                  self.day_x_FC.in_features // self.in_channels, \
                                  self.hour_x_FC.in_features // self.in_channels

            week = x[:, :, :num_w, :].reshape(batch_size, node_num, -1)
            day = x[:, :, num_w:num_w+num_d, :].reshape(batch_size, node_num, -1)
            hour = x[:, :, num_w+num_d:num_w+num_d+num_h, :].reshape(batch_size, node_num, -1)

            week_time = x_time[:, :, :num_w, :].reshape(batch_size, node_num, -1)
            day_time = x_time[:, :, num_w:num_w+num_d, :].reshape(batch_size, node_num, -1)
            hour_time = x_time[:, :, num_w+num_d:num_w+num_d+num_h, :].reshape(batch_size, node_num, -1)

            week_out = F.relu(self.week_FC(torch.cat([self.week_x_FC(week), self.week_time_FC(week_time)], dim=-1)))
            day_out = F.relu(self.day_FC(torch.cat([self.day_x_FC(day), self.day_time_FC(day_time)], dim=-1)))
            hour_out = F.relu(self.hour_FC(torch.cat([self.hour_x_FC(hour), self.hour_time_FC(hour_time)], dim=-1)))

            x = self.fusion_x_time2(torch.cat([week_out, day_out, hour_out], dim=-1))
            residual = x
            gate_x = torch.sigmoid(self.gate_FC2(residual))
            info = torch.tanh(self.info_FC2(residual))
            return x + gate_x * info

        return x
