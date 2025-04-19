import torch
import torch.nn as nn

class DummyGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_seq_len):
        super(DummyGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output_seq_len = output_seq_len

    def forward(self, x):
        # x: (batch, num_nodes, seq_len, input_dim)
        batch_size, num_nodes, seq_len, input_dim = x.shape

        # Reshape: (batch*num_nodes, seq_len, input_dim)
        x = x.view(batch_size * num_nodes, seq_len, input_dim)

        _, h = self.gru(x)  # h: (1, batch*num_nodes, hidden_dim)
        h = h.squeeze(0)    # (batch*num_nodes, hidden_dim)

        out = self.linear(h)  # (batch*num_nodes, output_dim)

        # Repeat output for output_seq_len time steps
        out = out.unsqueeze(1).repeat(1, self.output_seq_len, 1)  # (batch*num_nodes, output_seq_len, output_dim)

        # Reshape back: (batch, num_nodes, output_seq_len, output_dim)
        out = out.view(batch_size, num_nodes, self.output_seq_len, -1)
        return out
