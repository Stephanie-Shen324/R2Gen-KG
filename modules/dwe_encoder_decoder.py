from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel
from .low_rank import LowRank


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class DWETransformer(nn.Module):
    def __init__(self, encoder, decoder, cnn_embed, gcn_embed, tgt_embed, rm):
        # TODO check the init param
        super(DWETransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cnn_embed = cnn_embed
        self.gcn_embed = gcn_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, cnn_feats, gcn_feats, seq, cnn_masks, gcn_masks, seq_mask):
        # TODO: form a decoder mask?
        """
        cnn_masks.shape torch.Size([16, 1, 49])
        gcn_masks.shape torch.Size([16, 1, 21])
        """

        src_mask = torch.cat([cnn_masks, gcn_masks], dim = -1) # TODO: cat dim?
        return self.decode(self.encode(cnn_feats, gcn_feats, cnn_masks, gcn_masks), src_mask, seq, seq_mask)

    def encode(self, cnn_feats, gcn_feats, cnn_masks, gcn_masks):
        return self.encoder(self.cnn_embed(cnn_feats),self.gcn_embed(gcn_feats), cnn_masks, gcn_masks )

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)

        # out = self.model(cnn_feats, gcn_feats, seq, cnn_masks, gcn_masks, seq_mask) # Transformer


class DualWayEncoder(nn.Module):
    def __init__(self, layer, N):
        super(DualWayEncoder, self).__init__()
        self.layers = clones(layer, N)
        # separately do layer norm
        self.cnn_norm = LayerNorm(layer.cnn_d_model)
        # issue: duplicate attribute
        self.gcn_norm = LayerNorm(layer.gcn_d_model)

    def forward(self, cnn_feats, gcn_feats, cnn_masks, gcn_masks):
        for layer in self.layers:
            cnn_feats, gcn_feats = layer(cnn_feats, gcn_feats, cnn_masks, gcn_masks)

        cnn_feats, gcn_feats = self.cnn_norm(cnn_feats), self.gcn_norm(gcn_feats)
        all_feats = torch.cat([cnn_feats, gcn_feats], dim = 1)
        return all_feats

class DualWayEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(DualWayEncoderLayer, self).__init__()
        self.self_attn = clones(self_attn, 4) # either self attention or Xattention
        self.feed_forward = clones(feed_forward, 4)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 8)
        self.d_model = d_model
        self.cnn_d_model = d_model
        self.gcn_d_model = d_model

    def forward(self, cnn_feats, gcn_feats, cnn_masks, gcn_masks):
        cnn_feats = self.sublayer[0](cnn_feats, lambda cnn_feats: self.self_attn[0](cnn_feats, cnn_feats, cnn_feats, cnn_masks)) # self_attention, layernorm, dropout, residualconnection,
        cnn_feats = self.sublayer[1](cnn_feats, self.feed_forward[0]) # feed_forward


        gcn_feats = self.sublayer[2](gcn_feats, lambda gcn_feats: self.self_attn[1](gcn_feats, gcn_feats, gcn_feats, gcn_masks)) # self_attention, layernorm, dropout, residualconnection,
        gcn_feats = self.sublayer[3](gcn_feats, self.feed_forward[1]) # feed_forward
        # print('cnn_feats.shape',cnn_feats.shape)
        # print('gcn_feats.shape',gcn_feats.shape)
        all_feats = torch.cat([cnn_feats, gcn_feats], dim = 1)
        src_mask = torch.cat([cnn_masks, gcn_masks], dim = -1)

        cnn_feats = self.sublayer[4](cnn_feats, lambda cnn_feats: self.self_attn[2](cnn_feats, all_feats, all_feats, src_mask)) # self_attention, layernorm, dropout, residualconnection,
        cnn_feats = self.sublayer[5](cnn_feats, self.feed_forward[2]) # feed_forward

        gcn_feats = self.sublayer[6](gcn_feats, lambda gcn_feats: self.self_attn[3](gcn_feats, all_feats, all_feats, src_mask)) # self_attention, layernorm, dropout, residualconnection,
        gcn_feats = self.sublayer[7](gcn_feats, self.feed_forward[3]) # feed_forward
        # print('cnn_feats.shape',cnn_feats.shape)
        # print('gcn_feats.shape',gcn_feats.shape)
        return cnn_feats, gcn_feats



class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory)
        delta_beta = self.mlp_beta(memory)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class DWEEncoderDecoder(AttModel):

    def make_model(self, tgt_vocab, lowrank = False):
        c = copy.deepcopy
        if lowrank:
            attn = LowRank(embed_dim = self.d_model, att_heads = self.num_heads)
        else:
            attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = DWETransformer(
            DualWayEncoder(DualWayEncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers),
            lambda x: x,
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer, lowrank = False):
        super(DWEEncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model
        self.lowrank = lowrank

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab, lowrank)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        cnn_feats, gcn_feats = att_feats[0], att_feats[1]
        cnn_feats, seq, cnn_masks, seq_mask = self._prepare_feature_forward(cnn_feats, att_masks)
        gcn_feats, seq, gcn_masks, seq_mask = self._prepare_feature_forward(gcn_feats, att_masks)
        memory = self.model.encode(cnn_feats, gcn_feats, cnn_masks, gcn_masks)

        att_feats = torch.cat([cnn_feats, gcn_feats], dim = 1)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        # main forward?
        assert len(att_feats) == 2

        cnn_feats, gcn_feats = att_feats[0], att_feats[1]

        # seq and seq_mask should be the same for all input
        cnn_feats, seq, cnn_masks, seq_mask = self._prepare_feature_forward(cnn_feats, att_masks, seq)
        gcn_feats, seq, gcn_masks, seq_mask = self._prepare_feature_forward(gcn_feats, att_masks, seq)

        out = self.model(cnn_feats, gcn_feats, seq, cnn_masks, gcn_masks, seq_mask) # Transformer
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
