import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import ViTModel
from src.models.attention import CrossAttention
from src.models.nextvlad import NeXtVLAD
from src.models.transformer.transformer import SelfEncoder
from src.models.attention import SoftAttention


class Cos(nn.Module):
    def __init__(self, channel, win_size):
        super(Cos, self).__init__()
        self.conv = nn.Conv1d(in_channels=win_size // 2,
                              out_channels=channel,
                              kernel_size=1)

    def forward(self, x):
        part1, part2 = torch.split(x, x.shape[1] // 2, dim=1)
        part1 = self.conv(part1).squeeze()
        part2 = self.conv(part2).squeeze()
        cos = F.cosine_similarity(part1, part2, dim=2)
        return cos

    
class GlobalBranch(nn.Module):
    def __init__(self, in_size, h_size):
        super(GlobalBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_size,  h_size,   3, padding=1)
        self.conv2 = nn.Conv1d(h_size,   h_size*2, 3, padding=1)
        self.conv3 = nn.Conv1d(h_size*2, h_size*4, 3, padding=1)
        self.conv4 = nn.Conv1d(h_size*6, h_size*2, 3, padding=1)
        self.conv5 = nn.Conv1d(h_size*5, h_size,   3, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        pad_len = (4 - x.size(2) % 4) % 4
        if pad_len < 4:
            zeros = torch.zeros(x.size(0), x.size(1), pad_len).to(x.device)
            x = torch.cat([x, zeros], dim=2)

        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(self.maxpool1(conv1)))
        conv3 = F.relu(self.conv3(self.maxpool2(conv2)))
        conv4 = F.relu(self.conv4(torch.cat([conv2, self.upsample1(conv3)], dim=1)))
        conv5 = F.relu(self.conv5(torch.cat([conv1, self.upsample2(conv2), self.upsample3(conv4)], dim=1)))

        out = conv5[:, :, :-pad_len] if pad_len > 0 else conv5

        return out


class LocalBranch(nn.Module):
    def __init__(self, in_size, h_size):
        super(LocalBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_size,
                               out_channels=h_size,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=h_size,
                               out_channels=h_size,
                               kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        return out2


class SceneSegCNN(nn.Module):
    """ 在frame级别上进行切分的SceneSeg模型
        包含2个一维CNN对frame特征序列进行编码，后接分类层对序列的每个位置(frame)进行二分类， 判断改点是否为一个scene的开头
        只用了video特征，未做模态融合
    """

    def __init__(self, config):
        super(SceneSegCNN, self).__init__()
        config_v = config["video_model_config"]

        self.v_attn = SoftAttention(config_v['in_size'])
        self.cos = Cos(config['cos_channel'], config_v["win_size"])

        self.local_branch = LocalBranch(config_v['in_size'], config_v['h_size'])
        self.global_branch = GlobalBranch(config_v['in_size'], config_v['h_size'])

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config_v['h_size'] + config['cos_channel'], config['h_size'])
        self.classifier = nn.Linear(config['h_size'], config['class_num'])
        
        self.fc2 = nn.Linear(config_v['h_size'], config['h_size'])
        self.classifier2 = nn.Linear(config['h_size'], config['class_num'])

    def forward(self, x, lengths):
        bsz = x.shape[0]
        x = x.view(-1, x.shape[2], x.shape[3])
        sim = self.cos(x)
        sim = sim.view(bsz, -1, sim.shape[-1]).permute(0, 2, 1)
        x, _ = self.v_attn(x)
        x = x.view(bsz, -1, x.shape[-1])
   
        # local
        x = x.permute(0, 2, 1)
        out1 = self.local_branch(x)
        out1 = torch.cat([out1, sim], dim=1)
        out1 = out1.permute(0, 2, 1)
        logits1 = self.classifier(F.relu(self.fc(self.dropout(out1))))
        
        # global
        out2 = self.global_branch(x)
        out2 = out2.permute(0, 2, 1)
        logits2 = self.classifier2(F.relu(self.fc2(self.dropout(out2))))

        return logits1 + logits2

    
class SceneSegMultiCNN(nn.Module):
    """ """

    def __init__(self, config):
        super(SceneSegMultiCNN, self).__init__()
        self.h = config["h_size"]
        config_v = config["video_model_config"]

        self.v_attn = SoftAttention(config_v['in_size'])

        self.cos = Cos(config['cos_channel'], config_v["win_size"])

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=config_v['in_size'], out_channels=config['h_size'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=config['h_size'], out_channels=config['h_size'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=config['h_size'], out_channels=config['h_size'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=config['h_size'], out_channels=config['h_size'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=config['h_size'], out_channels=config['h_size'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=config['h_size'], out_channels=config['h_size'], kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config['h_size'] + config['cos_channel'], config['h_size'])
        self.classifier = nn.Linear(config['h_size'], config['class_num'])

    def forward(self, x, lengths):
        bsz = x.shape[0]
        x = x.view(-1, x.shape[2], x.shape[3])
        sim = self.cos(x)
        sim = sim.view(bsz, -1, sim.shape[-1]).permute(0, 2, 1)
        x, _ = self.v_attn(x)
        x = x.view(bsz, -1, x.shape[-1])
        x = x.permute(0, 2, 1)

        out = self.conv(x)
        out = torch.cat([out, sim], dim=1)
        out = out.permute(0, 2, 1)

        logits = self.classifier(F.relu(self.fc(self.dropout(out))))

        return logits
    

class SceneSegCNNTRM(nn.Module):
    """ """

    def __init__(self, config):
        super(SceneSegCNNTRM, self).__init__()
        self.h = config["h_size"]
        config_v = config["video_model_config"]

        self.v_attn = SoftAttention(config_v['in_size'])

        self.cos = Cos(config['cos_channel'], config_v["win_size"])

        self.conv1 = self.conv1 = nn.Conv1d(in_channels=config_v['in_size'],
                               out_channels=config['h_size'],
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=config['h_size'],
                               out_channels=config['h_size'],
                               kernel_size=3, padding=1)

        self.transformer_encoder = SelfEncoder(hidden_size=config['h_size'] * 2 + config['cos_channel'],
                                               head_size=(config['h_size'] * 2 + config['cos_channel']) // 8,
                                               n_heads=8,
                                               ff_size=128,
                                               n_layers=6)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config['h_size'] * 2 + config['cos_channel'], config['h_size'])
        self.classifier = nn.Linear(config['h_size'], config['class_num'])

    def forward(self, x, lengths):
        bsz = x.shape[0]
        x = x.view(-1, x.shape[2], x.shape[3])
        sim = self.cos(x)
        sim = sim.view(bsz, -1, sim.shape[-1]).permute(0, 2, 1)
        x, _ = self.v_attn(x)
        x = x.view(bsz, -1, x.shape[-1])
        x = x.permute(0, 2, 1)
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out = torch.cat([out1, out2, sim], dim=1)
        out = out.permute(0, 2, 1)

        mask = torch.ones((out.shape[1], out.shape[1])).to(x.device)
        mask = 1 - torch.tril(torch.triu(mask, diagonal=-1), diagonal=1)
        mask = mask.squeeze(0).repeat(bsz, 1, 1).to(torch.bool)
        out, _ = self.transformer_encoder(out, enc_self_attn_mask=mask)

        logits = self.classifier(F.relu(self.fc(self.dropout(out))))

        return logits


class SceneSegTransformer(nn.Module):
    """ """

    def __init__(self, config):
        super(SceneSegTransformer, self).__init__()
        config_v = config["video_model_config"]

        self.win_encoder = SelfEncoder(hidden_size=config_v['in_size'],
                                        head_size=config_v['in_size'] // 8,
                                        n_heads=8,
                                        ff_size=128,
                                        n_layers=6)
        self.v_attn = SoftAttention(config_v['in_size'])
        self.transformer_encoder = SelfEncoder(hidden_size=config_v['in_size'],
                                               head_size=config_v['in_size'] // 8,
                                               n_heads=8,
                                               ff_size=128,
                                               n_layers=6)

        self.cos = Cos(config['cos_channel'], config_v['win_size'])

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config_v['in_size'] + config['cos_channel'], config['h_size'])
        self.classifier = nn.Linear(config['h_size'], 2)

    def forward(self, x, lengths):
        bsz = x.shape[0]
        x = x.view(-1, x.shape[2], x.shape[3])
        sim = self.cos(x)
        sim = sim.view(bsz, -1, sim.shape[-1])
        x, _ = self.win_encoder(x)
        x, _ = self.v_attn(x)
        x = x.view(bsz, -1, x.shape[-1])

        mask = torch.ones((x.shape[1], x.shape[1])).to(x.device)
        mask = 1 - torch.tril(torch.triu(mask, diagonal=-1), diagonal=1)
        mask = mask.squeeze(0).repeat(bsz, 1, 1).to(torch.bool)
        out, _ = self.transformer_encoder(x, enc_self_attn_mask=mask)

        out = torch.cat([out, sim], dim=-1)
        logits = self.classifier(F.relu(self.fc(self.dropout(out))))

        return logits
    

if __name__ == '__main__':
    config = {"v_in_size": 1000, "v_h_size": 1000, "v_dp": 0.2}
    model = SceneSegCNN(config).cuda()
