import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import AutoModel
from src.models.nextvlad import NeXtVLAD
from src.models.attention import CrossAttention, SoftAttention
from src.models.fusion import SeGate


class MultiTagging(nn.Module):
    """
    包含1个NeXtVLAD对单个scene的frame特征序列进行编码，后接一个分类层
    """
    def __init__(self, config):
        super(MultiTagging, self).__init__()
        self.h = config['h_size']
        config_v = config["video_model_config"]
        config_t = config["text_model_config"]

        self.t_attn = SoftAttention(config_t["in_size"])

        vlad_size = config_v["num_clusters"] * \
                         int((config_v["lamb"] * (config_v["in_size"])) // config_v["groups"])

        self.nextvlad = NeXtVLAD(in_size=config_v['in_size'],
                                 lamb=config_v['lamb'],
                                 num_clusters=config_v['num_clusters'],
                                 groups=config_v['groups'],
                                 max_len=config_v['win_size']*config_v['max_len'])

        self.dropout = nn.Dropout(config["dropout"])

        self.fc = nn.Linear(vlad_size + config_t["in_size"], config['h_size'])
        self.classifier = nn.Linear(config['h_size'], 82)

    def _attention(self, src, tgt, src_lenghts):
        batch_size, src_len, src_dim = src.size()
        _, tgt_len, tgt_dim = tgt.size()
        src_reps = src.contiguous().view(batch_size, 1, src_len, src_dim).expand(batch_size, tgt_len, src_len, src_dim)
        tgt_reps = tgt.contiguous().view(batch_size, tgt_len, 1, tgt_dim).expand(batch_size, tgt_len, src_len, tgt_dim)

        seq_range = torch.arange(0, src_len).long().to(src.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, src_len)
        seq_length_expand = src_lenghts.unsqueeze(1).expand_as(seq_range_expand).long().to(src.device)
        mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(batch_size, tgt_len, src_len).float()

        _, attn_weights = self.cross_attn(src_reps, tgt_reps, mask)
        attn_out = attn_weights.unsqueeze(3).expand_as(src_reps) * src_reps
        attn_out = torch.sum(attn_out, dim=2)

        return attn_out

    def forward(self, x_v, x_t, lengths):
        x_v = x_v.view(x_v.shape[0], -1, x_v.shape[-1])
        vlad = self.nextvlad(x_v)
        text, _ = self.t_attn(x_t)
        out = torch.cat([vlad, text], dim=-1)
        out = self.dropout(out)
        out = F.relu(self.fc(out))
        logits = self.classifier(out)

        return logits


class MultiTaggingWin(nn.Module):
    """
    对窗口内的frame进行加权求和，后接1个NeXtVLAD对单个scene的frame特征序列进行编码
    对单个scene的文本序列进行加权求和
    后接一个分类层
    """
    def __init__(self, config):
        super(MultiTaggingWin, self).__init__()
        self.h = config['h_size']
        config_v = config["video_model_config"]
#         config_t = config["text_model_config"]

        self.v_attn = SoftAttention(config_v['in_size'])
#         self.t_attn = SoftAttention(config_t["in_size"])

        vlad_size = config_v["num_clusters"] * \
                         int((config_v["lamb"] * (config_v["in_size"])) // config_v["groups"])

        self.nextvlad = NeXtVLAD(in_size=config_v['in_size'],
                                  lamb=config_v['lamb'],
                                  num_clusters=config_v['num_clusters'],
                                  groups=config_v['groups'],
                                  max_len=config_v['max_len'])

        self.dropout = nn.Dropout(config["dropout"])

        self.fc = nn.Linear(vlad_size, config['h_size'])
        self.classifier = nn.Linear(config['h_size'], 82)

    def _attention(self, src, tgt, src_lenghts):
        batch_size, src_len, src_dim = src.size()
        _, tgt_len, tgt_dim = tgt.size()
        src_reps = src.contiguous().view(batch_size, 1, src_len, src_dim).expand(batch_size, tgt_len, src_len, src_dim)
        tgt_reps = tgt.contiguous().view(batch_size, tgt_len, 1, tgt_dim).expand(batch_size, tgt_len, src_len, tgt_dim)

        seq_range = torch.arange(0, src_len).long().to(src.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, src_len)
        seq_length_expand = src_lenghts.unsqueeze(1).expand_as(seq_range_expand).long().to(src.device)
        mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(batch_size, tgt_len, src_len).float()

        _, attn_weights = self.cross_attn(src_reps, tgt_reps, mask)
        attn_out = attn_weights.unsqueeze(3).expand_as(src_reps) * src_reps
        attn_out = torch.sum(attn_out, dim=2)

        return attn_out

    def forward(self, x_v, lengths):
        bsz = x_v.shape[0]
        x_v = x_v.view(-1, x_v.shape[2], x_v.shape[3])
        video, _ = self.v_attn(x_v)
        video = video.view(bsz, -1, video.shape[-1])
        vlad = self.nextvlad(video)

#         text, _ = self.t_attn(x_t)
#         out = torch.cat([vlad, text], dim=-1)
        out = self.dropout(vlad)
        out = F.relu(self.fc(out))
        logits = self.classifier(out)

        return logits


if __name__ == '__main__':
    config = {"v_in_size": 768, "v_h_size": 768, "v_dp": 0.2}
    model = MultiTagging(config).cuda()
