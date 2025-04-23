import torch
import torch.nn as nn
from AGCRN_model.AGCRNCell import AGCRNCell


class AGCRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_seq_len, 
                 num_nodes=18, cheb_k=3, embed_dim=10, num_layers=1):
        """
        AGCRN model with core parameters:
        - input_dim: Dimension of input features
        - hidden_dim: Dimension of hidden states
        - output_dim: Dimension of output features
        - output_seq_len: Length of output sequence (horizon)
        
        Other parameters have default values suitable for most cases
        """
        super(AGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = output_seq_len
        self.num_layers = num_layers
        
        # Node embeddings
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        
        # Encoder
        self.encoder = AVWDCRNN(num_nodes, input_dim, hidden_dim, cheb_k, 
                              embed_dim, num_layers)
        
        # Predictor
        self.end_conv = nn.Conv2d(1, output_seq_len * output_dim, 
                                 kernel_size=(1, hidden_dim), bias=True)

    def forward(self, source):
        """
        source: input tensor of shape (B, T_in, N, input_dim)
        returns: output tensor of shape (B, T_out, N, output_dim)
        """
        # Initialize hidden state
        init_state = self.encoder.init_hidden(source.shape[0])
        
        # Encoder forward pass
        output, _ = self.encoder(source, init_state, self.node_embeddings)
        output = output[:, -1:, :, :]  # Take last time step
        
        # CNN-based predictor
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # (B, T, N, C)
        
        return output


class AVWDCRNN(nn.Module):
    """
    Modified AVWDCRNN to work with the simplified interface
    """
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num
        assert x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)