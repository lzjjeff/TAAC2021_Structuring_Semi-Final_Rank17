import torch
import torch.nn as nn
import torch.nn.functional as F


class SeGate(nn.Module):
    def __init__(self, config):
        super(SeGate, self).__init__()
        self.h = config["h_size"]
        config_v = config["video_model_config"]
        config_t = config["text_model_config"]

        vlad_size = config_v["num_clusters"] * \
                         int((config_v["lamb"] * (config_v["in_size"])) // config_v["groups"])
        self.gate_reduce = config["gate_reduce"]
        self.project = nn.Linear(vlad_size+config_t["in_size"], config["h_size"])
        self.gate_weight1 = nn.Linear(config["h_size"], config["h_size"] // config["gate_reduce"], bias=False)
        self.gate_weight2 = nn.Linear(config["h_size"] // config["gate_reduce"], config["h_size"], bias=False)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x_list):
        concat = torch.cat(x_list, dim=-1)
        activation = self.dropout(self.project(concat))
        gate = F.sigmoid(self.gate_weight2(self.gate_weight1(activation)))
        return gate * activation
