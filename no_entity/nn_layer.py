
from torch import nn
import torch
from collections import OrderedDict
import helper
import numpy as np
import torch.nn.functional as F


# class EmbeddingLayer(nn.Module):
#     """Embedding class which includes only an embedding layer."""
#
#     def __init__(self, input_size, config):
#         """"Constructor of the class"""
#         super(EmbeddingLayer, self).__init__()
#
#         if config.emtraining:
#             self.embedding = nn.Sequential(OrderedDict([
#                 ('embedding', nn.Embedding(input_size, config.emsize)),
#                 ('dropout', nn.Dropout(config.dropout))
#             ]))
#         else:
#             self.embedding = nn.Embedding(input_size, config.emsize)
#             self.embedding.weight.requires_grad = False
#
#     def forward(self, input_variable):
#         """"Defines the forward computation of the embedding layer."""
#         return self.embedding(input_variable)
#
#     def init_embedding_weights(self, dictionary, embeddings_index, embedding_dim):
#         """Initialize weight parameters for the embedding layer."""
#         pretrained_weight = np.empty([len(dictionary), embedding_dim], dtype=float)
#         for i in range(len(dictionary)):
#             if dictionary.idx2word[i] in embeddings_index:
#                 pretrained_weight[i] = embeddings_index[dictionary.idx2word[i]]
#             else:
#                 pretrained_weight[i] = helper.initialize_out_of_vocab_words(embedding_dim)
#         # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
#         if isinstance(self.embedding, nn.Sequential):
#             self.embedding[0].weight.data.copy_(torch.from_numpy(pretrained_weight))
#         else:
#             self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
#

class Lstm_Embedding(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, hidden_size, bidirection, config):
        """"Constructor of the class"""
        super(Lstm_Embedding, self).__init__()

        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirection = bidirection

        #if self.config.model in ['LSTM', 'GRU']:
        #    self.embed_rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, self.config.nlayer_enc,
        #                                              batch_first=True, dropout=self.config.dropout,
        #                                              bidirectional=self.bidirection)
        #else:
        #    try:
        #        nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
        #    except KeyError:
        #        raise ValueError("""An invalid option for `--model` was supplied,
        #                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        #    self.rnn = nn.RNN(self.input_size, self.hidden_size, self.config.nlayers, nonlinearity=nonlinearity,
        #                      batch_first=True, dropout=self.config.dropout, bidirectional=self.bidirection)


        self.rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, self.config.nlayer_enc,
                                                  batch_first=True, dropout=self.config.dropout,
                                                  bidirectional=self.bidirection)
        # for param in self.embed_rnn.parameters():
        #    param.requires_grad = False
    def forward(self, sent_variable, sent_len):
        """"Defines the forward computation of the encoder"""
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.config.cuda else torch.from_numpy(idx_sort)
        sent_variable = sent_variable.index_select(0, idx_sort)

        # Sort by length (keep idx)

        # Handling padding in Recurrent Networks

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_variable, sent_len.copy(), batch_first=True)
        sent_output, hidden = self.rnn(sent_packed)
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.config.cuda else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(0, idx_unsort)

        return sent_output, hidden


class Encoder(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, hidden_size, bidirection, config):
        """"Constructor of the class"""
        super(Encoder, self).__init__()

        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirection = bidirection


        self.rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, self.config.nlayer_enc,
                                                  batch_first=True, dropout=self.config.dropout,
                                                  bidirectional=self.bidirection)


    def forward(self, h_n, batch_sequneces_length):
        batch_sequneces_length = np.asarray(batch_sequneces_length)
        h_n = torch.squeeze(h_n, 0)
        sent_variable = helper.pad_sequence(h_n, batch_sequneces_length)

        """"Defines the forward computation of the encoder"""
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(batch_sequneces_length)[::-1], np.argsort(-batch_sequneces_length)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.config.cuda else torch.from_numpy(idx_sort)
        sent_variable = sent_variable.index_select(0, idx_sort)

        # Sort by length (keep idx)

        # Handling padding in Recurrent Networks

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_variable, sent_len.copy(), batch_first=True)
        sent_output, hidden = self.rnn(sent_packed)
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.config.cuda else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(0, idx_unsort)

        return sent_output, hidden

import math
class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """

    def __init__(self, batch_size, hidden_size, method="bahdanau"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
            # stdv = 1. / math.sqrt(self.va.size(0))
            # self.va.data.normal_(mean = 0, std=stdv)
        else:
            raise NotImplementedError


    def forward(self, last_hidden, encoder_outputs, seq_len=None):

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        if seq_len is not None:
            attention_energies = helper.mask_3d(attention_energies, seq_len, -float('inf'))

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`, `bahdanau`)
        :return: a score (batch_size, max_time)
        """

        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(torch.cat((encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)
        # elif method == "concat":
        #     x = last_hidden.unsqueeze(1)
        #     x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
        #     return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            # x = last_hidden.unsqueeze(1)
            out = F.tanh(self.Ua(encoder_outputs))
            # out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        # max_len = encoder_outputs.size(0)
        # this_batch_size = encoder_outputs.size(1)
        # H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        # encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        # attn_energies = self.score(H, encoder_outputs)  # compute attention score
        attn_energies = self.score(encoder_outputs)  # compute attention score (encoder_outputs ---> [B*T*H])

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = (torch.ByteTensor(mask).unsqueeze(1)).cuda()  # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies,1).unsqueeze(1)  # normalize with softmax

    def score(self, encoder_outputs):
        energy = F.tanh(self.attn(encoder_outputs))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

# Luong attention layer
class brsoff_Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(brsoff_Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        x = torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
        energy = self.attn(x).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
