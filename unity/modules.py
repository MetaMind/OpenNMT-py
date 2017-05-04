import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from tc import nn as tnn, softmax as softmax

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Encoder(nn.Module):

    def __init__(self, emb, d_hid, detach_after=6000):
        super().__init__()
        self.emb = emb
        self.proj = tnn.Linear(emb.embedding_dim, emb.embedding_dim)
        self.rnn = tnn.LSTM(emb.embedding_dim, d_hid)
        self.detach_after = 6000
        self.i = 0

    def forward(self, inputs, lengths):
        # make (seq_len, batch_size, d_feat)
        inputs = inputs.transpose(1, 0)
        emb = self.emb(inputs)
        if self.i < self.detach_after:
            emb = emb.detach()
        proj = emb
        h, _ = self.rnn(proj)

        h_last = []
        for i, l in enumerate(lengths):
            h_last.append(h[l-1, i])
        # make (batch_size, seq_len, d_feat)
        h = h.transpose(1, 0)
        return h, torch.stack(h_last, 0)


class BiEncoderCuDNN(nn.Module):

    def __init__(self, d_in, d_out, dropout=0.2, layers=1):
        super().__init__()
        self.layers = layers
        self.rnn = nn.LSTM(d_in, d_out,
                        num_layers=layers,
                        dropout=dropout,
                        bidirectional=True)

    def forward(self, inputs, lens, hidden=None):
        outputs, (h_t, c_t) = self.rnn(pack(inputs, lens), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        return outputs, (h_t, c_t)

    def from_mt(self, corpus):
        x = torch.load(corpus + '.model')['model']
        state_dict = {'.'.join(k.split('.')[-2:]): v for k, v in x.items() if 'rnn' in k and 'encoder' in k}
        if self.layers == 1:
            state_dict = {k: v for k, v in state_dict.items() if 'l1' not in k}
        self.load_state_dict(state_dict)
        return self


class BiEncoder(nn.Module):

    def __init__(self, d_in, d_out, layers=1):
        super().__init__()
        self.rnn = tnn.LSTM(d_in, d_out)

    def forward(self, inputs, lens, dropout=0.2):
        # make (seq_len, batch_size, d_feat)
#        inputs = inputs.transpose(1, 0)
        fw_input = F.dropout(inputs, dropout, self.training)
        fw, _ = self.rnn(fw_input)
        bw_input = F.dropout(self.reverse_sequence(fw_input, lens), dropout, self.training)
        bw_rev, _ = self.rnn(bw_input)
        bw = self.reverse_sequence(bw_rev, lens)
        fw_last = []
        for i, l in enumerate(lens):
            fw_last.append(fw[l-1, i])
        bw_last = []
        for i, l in enumerate(lens):
            bw_last.append(bw[l-1, i])
        h = torch.cat([fw, bw], 2)
        # make (batch_size, seq_len, d_feat)
#        h = h.transpose(1, 0)
        return h, torch.stack(h_last, 0)

    @staticmethod
    def reverse_sequence(seq, lens):
        max_len = max(lens)
        rev = []
        for i, l in enumerate(lens):
            x = seq[:l, i]
            x_rev = torch.stack(list(reversed(list(x))), 0)
            if l < max_len:
                pad = Variable(x_rev.data.new(max_len - l, x_rev.size(-1)).zero_())
                x_rev = torch.cat([x_rev, pad], 0)
            rev.append(x_rev)
        return torch.stack(rev, 1)


class AttentiveMixin:

    @staticmethod
    def attn_context(weights, candidates):
        return weights.unsqueeze(2).expand_as(candidates).mul(candidates).sum(1).squeeze(1)

    @staticmethod
    def coattn_context(weights, candidates):
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(1).squeeze(1)

    @staticmethod
    def normalize(raw_scores, lengths):
        backup = raw_scores.data.clone()
        max_len = max(lengths)
        for i, l in enumerate(lengths):
            if l == max_len:
                continue
            raw_scores.data[i, l:] = -np.inf
        normalized_scores = softmax(raw_scores)
        raw_scores.data.copy_(backup)
        return normalized_scores


class BiAttention(nn.Module, AttentiveMixin):

    def __init__(self, d_in, d_out, proj=False):
        super().__init__()
        if proj:
            self.proj = tnn.Linear(d_in, d_in)

    def forward(self, premise, premise_lens, hypothesis, hypothesis_lens, dropout=0.2):
        # (batch, time_d, h_feat)
        premise = F.dropout(premise, dropout, self.training)
        hypothesis = F.dropout(hypothesis, dropout, self.training)
        if hasattr(self, 'proj'):
            premise_transformed = self.proj(premise)
        else:
            premise_transformed = premise

        raw_attention_scores = premise_transformed.bmm(hypothesis.transpose(2, 1))
        attn_p_given_h = self.normalize(raw_attention_scores, premise_lens)
        attn_h_given_p = self.normalize(raw_attention_scores.transpose(2, 1), hypothesis_lens)

        context_h_given_p = self.coattn_context(attn_h_given_p, hypothesis)
        context_p_given_h = self.coattn_context(attn_p_given_h, premise)

        r1 = torch.cat([premise-context_h_given_p, context_h_given_p*premise], 2)
        r2 = torch.cat([hypothesis-context_p_given_h, context_p_given_h*hypothesis], 2)
        return r1.transpose(0, 1), r2


class Coattention(BiAttention, AttentiveMixin):

    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out)
        self.proj = tnn.Linear(d_in, d_in)
        self.sentinel = tnn.Embedding(2, d_out)

    def forward(self, question, question_lengths, document, document_lengths, proj_dropout=0.2):
        # (batch, time_d, h_feat)
        q_null = self.sentinel(Variable(question.data.new(question.size(0)).long().fill_(0)))
        d_null = self.sentinel(Variable(document.data.new(document.size(0)).long().fill_(1)))
        q = torch.cat([q_null.unsqueeze(1), F.dropout(question, proj_dropout, self.training)], 1)
        d = torch.cat([d_null.unsqueeze(1), F.dropout(document, proj_dropout, self.training)], 1)
        q_len = [i+1 for i in question_lengths]
        d_len = [i+1 for i in document_lengths]

        qproj = self.proj(q).tanh()

        affinity = qproj.bmm(d.transpose(2, 1))
        attn_q_given_d = self.normalize(affinity, q_len)
        attn_d_given_q = self.normalize(affinity.transpose(2, 1), d_len)

        cont_d_given_q = self.coattn_context(attn_d_given_q, d)
        cont_q_given_d = self.coattn_context(attn_q_given_d, qproj)
        coattn = self.coattn_context(attn_q_given_d, cont_d_given_q)
        return torch.cat([cont_q_given_d, coattn], 2)[:, 1:]


class BiCoattention(Coattention, AttentiveMixin):

    def forward(self, question, question_lengths, document, document_lengths, proj_dropout=0.2):
        # (batch, time_d, h_feat)
        document = F.dropout(document, proj_dropout, self.training)

        q_null = self.sentinel(Variable(question.data.new(question.size(0)).long().fill_(0)))
        d_null = self.sentinel(Variable(document.data.new(document.size(0)).long().fill_(1)))
        q = torch.cat([q_null.unsqueeze(1), question], 1)
        d = torch.cat([d_null.unsqueeze(1), document], 1)
        q_len = [i+1 for i in question_lengths]
        d_len = [i+1 for i in document_lengths]

        qproj = self.proj(F.dropout(q, proj_dropout, self.training)).tanh()

        affinity = qproj.bmm(d.transpose(2, 1))
        attn_q_given_d = self.normalize(affinity, q_len)
        attn_d_given_q = self.normalize(affinity.transpose(2, 1), d_len)

        cont_d_given_q = self.coattn_context(attn_d_given_q, d)
        cont_q_given_d = self.coattn_context(attn_q_given_d, qproj)

        coattn_d_given_q = self.coattn_context(attn_q_given_d, cont_d_given_q)
        coattn_q_given_d = self.coattn_context(attn_d_given_q, cont_q_given_d)
        return coattn_q_given_d[:, 1:], coattn_d_given_q[:, 1:]


class SelfAttention(nn.Module, AttentiveMixin):

    def __init__(self, d_in):
        super().__init__()
        self.scorer = tnn.Linear(d_in, 1)

    def __call__(self, inputs, lens, dropout=0.1):
        scores = self.scorer(F.dropout(inputs, dropout, self.training)).squeeze(2)
        weight = self.normalize(scores, lens)
        return self.attn_context(weight, inputs)


class FancySelfAttention(nn.Module, AttentiveMixin):

    def __init__(self, d_in):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Dropout(0.2),
            tnn.Linear(d_in, d_in),
            nn.Dropout(0.2),
            nn.Tanh(),
            tnn.Linear(d_in, 1),
        )

    def __call__(self, inputs, lens, dropout=0.1):
        scores = self.scorer(F.dropout(inputs, dropout, self.training)).squeeze(2)
        weight = self.normalize(scores, lens)
        return self.attn_context(weight, inputs)
