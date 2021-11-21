import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None, key_mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            mask = mask == 0
            mask = mask[:, None, :, None]   
            scores = scores.masked_fill(mask, -1e9)
            if key_mask is not None:
                key_mask = key_mask == 0
                key_mask = key_mask[:,None,None,:]
                scores = scores.masked_fill(key_mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        # p_attn = scores/(scores.sum(dim=-1)[:,:,:,None]+0.0001)

        # if dropout is not None:
        #     p_attn = dropout(p_attn)

        # p_attn = torch.sigmoid(scores)
        # scores = scores*(1.0-mask.float())
        # p_attn = scores/(scores.sum(dim=-1)[:,:,:,None]+0.0001)
        # p_attn = p_attn*(1.0-mask.float())
        return torch.matmul(p_attn, value), p_attn