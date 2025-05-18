# Updated STNSCN model with ablation flags from config
import torch
import torch.nn as nn
from model.Decoder import Decoder
from model.Encoder import Encoder
from model.Transform import Transform

class STNSCN(nn.Module):
    def __init__(self, config, static_norm_adjs=None):
        super(STNSCN, self).__init__()

        model_cfg = config["model"]
        ab_cfg = config.get("ablation", {})

        self.in_channels = model_cfg["input_dim"]
        self.time_channels = model_cfg["time_dim"]
        self.hidden_channels = model_cfg["hidden_dim"]
        self.output_channels = model_cfg["output_dim"]

        self.seq_length = model_cfg["num_for_predict"]
        self.device = config["device"]
        self.use_transform = model_cfg.get("use_transform", True)

        # Ablation flags
        self.use_dynamic_graph = ab_cfg.get("use_dynamic_graph", True)
        self.use_graph_gate_rnn = ab_cfg.get("use_graph_gate_rnn", True)
        self.use_counterfactual = ab_cfg.get("use_counterfactual", True)
        self.use_input_gate = ab_cfg.get("use_input_gate", True)

        node_num = static_norm_adjs[0].shape[0]
        self.node_num = node_num

        self.encoder = Encoder(
            self.in_channels,
            self.time_channels,
            self.hidden_channels,
            model_cfg["gcn_depth"],
            model_cfg["alpha"],
            model_cfg["num_of_weeks"],
            model_cfg["num_of_days"],
            model_cfg["num_of_hours"],
            model_cfg["num_for_predict"],
            model_cfg["dropout_prob"],
            model_cfg["dropout_type"],
            model_cfg["fusion_mode"],
            node_num,
            static_norm_adjs,
            model_cfg["dyn_norm"],
            self.device,
            use_dynamic_graph=self.use_dynamic_graph,
            use_graph_gate_rnn=self.use_graph_gate_rnn
        )

        self.decoder = Decoder(
            self.in_channels,
            self.time_channels,
            self.hidden_channels,
            self.output_channels,
            model_cfg["gcn_depth"],
            model_cfg["alpha"],
            model_cfg["num_of_weeks"],
            model_cfg["num_of_days"],
            model_cfg["num_of_hours"],
            model_cfg["num_for_predict"],
            model_cfg["dropout_prob"],
            model_cfg["dropout_type"],
            "mix",
            node_num,
            static_norm_adjs,
            model_cfg["dyn_norm"],
            config["train"].get("use_curriculum_learning", True),
            config["train"].get("cl_decay_steps", 4000)
        )

        if self.use_transform:
            self.transform = Transform(
                self.time_channels,
                self.hidden_channels,
                model_cfg["num_of_weeks"],
                model_cfg["num_of_days"],
                model_cfg["num_of_hours"],
                model_cfg["num_for_predict"],
                model_cfg["num_for_target"],
                model_cfg["num_of_head"],
                model_cfg["dropout_prob"]
            )

    def forward(self, x, x_time, target_time, target_cl=None, task_level=2, global_step=None):
        batch_size, node_num, num_for_predict, dim = x.shape

        if len(x_time.shape) < 4 and len(target_time.shape) < 4:
            x_time = x_time.unsqueeze(dim=1).repeat(1, node_num, 1, 1)
            target_time = target_time.unsqueeze(dim=1).repeat(1, node_num, 1, 1)

        outputs, encoder_hiddens = self.encoder(x, x_time, self.seq_length)

        if self.use_transform:
            encoder_hiddens_last = encoder_hiddens[:, :, :, -1, :]
            encoder_hiddens = self.transform(encoder_hiddens, x_time, target_time)
            encoder_hiddens = encoder_hiddens + encoder_hiddens_last
        else:
            encoder_hiddens = encoder_hiddens[:, :, :, -1, :]

        GO_decoder_input = torch.zeros((batch_size, node_num, 1, self.in_channels), device=self.device)

        outputs_final = self.decoder(
            GO_decoder_input,
            target_time,
            target_cl,
            encoder_hiddens,
            task_level,
            global_step
        )

        del outputs, encoder_hiddens, GO_decoder_input

        return outputs_final
