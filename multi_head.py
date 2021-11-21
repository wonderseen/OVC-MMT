import torch.nn as nn
from attention import Attention
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V
import torch.nn.functional as F
import math
import torch as th
import torch.nn as nn



class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self,
            d_q : int,
            d_k : int,
            d_model : int = 768,
            dropout : float = 0.1,
            h : int = 1, 
            transform_Q : bool = False,
            transform_K : bool = False,
            transform_V : bool = False,
        ):
        super().__init__()
        assert d_model % h == 0

        self.d_qkv = [d_q, d_k, d_k]
        self.d_model_h = d_model // h
        self.h = h
        
        self.transform_Q = transform_Q
        self.transform_K = transform_K
        self.transform_V = transform_V
        
        if self.transform_Q:
            self.linear_layers_Q = nn.Linear(self.d_qkv[0], d_model)
        if self.transform_K:
            self.linear_layers_K = nn.Linear(self.d_qkv[1], d_model)
        if self.transform_V:
            self.linear_layers_V = nn.Linear(self.d_qkv[2], d_model)

        self.transform_QKV = self.transform_Q or self.transform_K or self.transform_V

        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.layernorm = nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, ctx_mask=None, key_mask=None):
        value = key
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if self.transform_QKV:
            if self.transform_Q:
                query = self.linear_layers_Q(query)
            if self.transform_K:
                key   = self.linear_layers_K(key)
            if self.transform_V:
                value = self.linear_layers_V(value)
        
        query, key, value = [x.view(batch_size, -1, self.h, self.d_model_h).transpose(1, 2) for x in (query, key, value)]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=ctx_mask, dropout=self.dropout, key_mask=key_mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_model_h)

        x = self.output_linear(x)

        x = self.dropout(x)
        
        # norm_x = x
        norm_x = self.layernorm(x)

        attn = attn.mean(dim=1).max(dim=-1)[0]

        # attn = attn.max(dim=1)[0].max(dim=-1)[0]

        # attn *= ctx_mask
        # attn /= (attn.sum(dim=-1)[:, None] + 1e-5)

        return x, norm_x, attn[:, None, :], attn.max(dim=1)[0]


class MultiHeadAttention_raw(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_q, d_k, d_model=768, dropout=0.1, h=1):
        super().__init__()
        assert d_model % h == 0

        self.d_model_h = d_model // h
        self.h = h
        self.d_qkv = [d_q, d_k, d_k]
        self.linear_layers = nn.ModuleList([nn.Linear(self.d_qkv[i], d_model) for i in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.layernorm = nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, ctx_mask=None, key_mask=None):
        value = key
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_model_h).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # key = key.view(batch_size, -1, self.h, self.d_model_h).transpose(1, 2)
        # value = key
        # query = self.linear_layers[0](query).view(batch_size, -1, self.h, self.d_model_h).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=ctx_mask, dropout=self.dropout, key_mask=key_mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_model_h)

        x = self.output_linear(x)

        x = self.dropout(x)
        
        norm_x = x


        # attn = attn.mean(dim=1).max(dim=-1)[0]
        attn_maxpool = attn.max(dim=1)[0].max(dim=-1)[0]

        # attn *= ctx_mask
        # attn /= (attn.sum(dim=-1)[:,None]+1e-5)

        return x, norm_x, attn_maxpool[:,None,:]