import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, src_size, tgt_size, hidden_size):
        super(CrossAttention, self).__init__()
        self.attn_1 = nn.Linear(src_size+tgt_size, hidden_size)
        self.attn_2 = nn.Linear(hidden_size, 1, bias=False)

    def get_attn(self, src_reps, tgt_reps, mask):
        cat_reps = torch.cat([src_reps, tgt_reps], dim=-1)
        attn_scores = self.attn_2(F.tanh(self.attn_1(cat_reps))).squeeze(3)  # (batch_size, tgt_len, src_len)
        attn_scores = mask * attn_scores
        attn_weights = torch.softmax(attn_scores, dim=2)     # (batch_size, tgt_len, src_len)

        attn_out = attn_weights.unsqueeze(3).expand_as(src_reps) * src_reps # (batch_size, tgt_len, src_len, hidden_dim)
        attn_out = torch.sum(attn_out, dim=2)   # (batch_size, tgt_len, hidden_dim)

        return attn_out, attn_weights

    def forward(self, src_reps, tgt_reps, mask):
        attn_out, attn_weights = self.get_attn(src_reps, tgt_reps, mask)

        return attn_out, attn_weights   # (batch_size, tgt_len, hidden_dim), (batch_size, tgt_len, src_len)


class SoftAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def get_attn(self, reps, mask=None):
        attn_scores = self.attn(reps).squeeze(2)
        if mask is not None:
            attn_scores = mask * attn_scores
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (batch_size, len, 1)
        attn_out = torch.sum(reps * attn_weights, dim=1)  # (batch_size, hidden_dim)

        return attn_out, attn_weights

    def forward(self, reps, mask=None):
        attn_out, attn_weights = self.get_attn(reps, mask)

        return attn_out, attn_weights  # (batch_size, hidden_dim), (batch_size, len, 1)
