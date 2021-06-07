import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def makeMask(input_features):
    bs, seq, feats = input_features.shape
    return torch.ones(bs, seq, dtype=torch.long).cuda()  # hardcode to cuda

class LowRank(nn.Module):
    def __init__(self, embed_dim, att_heads, att_mid_drop = 0.1, att_drop = 0.5, bifeat_drop = 0.3, ff_dropout = 0.5):
        super(LowRank, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = embed_dim

        mid_dim = int(embed_dim/att_heads)
        half_mid_dim = int(mid_dim/2)
        att_mid_dim = [mid_dim, half_mid_dim, mid_dim]

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = nn.CELU(1.3)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        in_proj = nn.Sequential(*sequential)
        self.in_proj = clones(in_proj, 4)  # q, k, v1, v2

        self.attn_net = SCAtt(att_mid_dim, att_mid_drop)

        self.dropout = nn.Dropout(att_drop)

        self.bifeat_emb = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(bifeat_drop)
        )

        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.ff_layer = PositionwiseFeedForward(embed_dim, embed_dim, ff_dropout)

        self.clear_buffer()
        # self.in_decocder = decoder


    def apply_to_states(self, fn):
        self.buffer_keys = fn(self.buffer_keys)
        self.buffer_value2 = fn(self.buffer_value2)

    def init_buffer(self, batch_size):
        self.buffer_keys = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()
        self.buffer_value2 = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()

    def clear_buffer(self):
        self.buffer_keys = None
        self.buffer_value2 = None

    # query -- batch_size * qdim
    # value -- batch_size * att_num * vdim

    def forward(self, query, key, value2, mask, precompute=False):
        # print(mask.shape)
        mask = mask.squeeze(1)
        # mask = makeMask(query)
        # print(mask.shape)
        # query = (torch.sum(query * mask.squeeze(1).unsqueeze(-1), 1) / torch.sum(mask.squeeze(1).unsqueeze(-1), 1))
        global_query = (torch.sum(query * mask.unsqueeze(-1), 1) / torch.sum(mask.unsqueeze(-1), 1))
        # cnn_gx = (torch.sum(cnn_feats * cnn_mask.unsqueeze(-1), 1) / torch.sum(cnn_mask.unsqueeze(-1), 1))

        # if self.in_decocder:
        #     attn =  self.forward_decoder(query, key, value2, mask, precompute)
        # else:
        attn =  self.forward_encoder(global_query, key, value2, mask, precompute)
        attn = self.dropout(attn)
        attn_ = torch.cat([attn.unsqueeze(1).expand_as(query), query], dim=-1)
        attn = self.ff_layer(self.layer_norm(self.bifeat_emb(attn_) + query))
        return attn

    def forward_encoder(self, query, key, value2, mask, precompute=False):


        # q should be the same as v1, but using different projection layer

        batch_size = query.size()[0]
        q = self.in_proj[0](query)
        v1 = self.in_proj[2](query)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj[1](key)
            v2 = self.in_proj[3](value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2

        attn_map = q.unsqueeze(-2) * k
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn

    # query -- batch_size * seq_num * qdim
    # value -- batch_size * att_num * vdim
    def forward_decoder(self, query, key, value2, mask, precompute=False):
        batch_size = query.size()[0]
        query_tmp = query.view(-1, query.size()[-1])
        value1 = query.view(-1, query.size()[-1])
        query = query_tmp

        q = self.in_proj[0](query)
        v1 = self.in_proj[2](value1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj[1](key)
            v2 = self.in_proj[3](value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if self.buffer_keys is not None and self.buffer_value2 is not None:
                self.buffer_keys = torch.cat([self.buffer_keys, k], dim=2)
                self.buffer_value2 = torch.cat([self.buffer_value2, v2], dim=2)
                k = self.buffer_keys
                v2 = self.buffer_value2
        else:
            k = key
            v2 = value2

        attn_map = q.unsqueeze(-2) * k.unsqueeze(-3)
        attn = self.attn_net.forward(attn_map, mask, v1, v2).transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2):
        batch_size = value2.size()[0]
        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2


class BasicAtt(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(BasicAtt, self).__init__()

        sequential = []
        for i in range(1, len(mid_dims) - 1):
            sequential.append(nn.Linear(mid_dims[i - 1], mid_dims[i]))
            sequential.append(nn.ReLU())
            if mid_dropout > 0:
                sequential.append(nn.Dropout(mid_dropout))
        self.attention_basic = nn.Sequential(*sequential) if len(sequential) > 0 else None
        self.attention_last = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)
        attn_weights = self.attention_last(att_map)
        attn_weights = attn_weights.squeeze(-1)
        if att_mask is not None:
            attn_weights = attn_weights.masked_fill(att_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn = torch.matmul(attn_weights.unsqueeze(-2), value2).squeeze(-2)
        return attn

class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)

        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        attn = value1 * value2 * alpha_channel
        return attn