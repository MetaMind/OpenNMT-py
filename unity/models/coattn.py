from models.model import Model as BaseModel
from modules import BiEncoder, BiCoattention, SelfAttention
import torch
from tc import nn as tnn
from torch import nn
from torch.nn import functional as F


class Model(BaseModel):

    def __init__(self, args, vocab, label_vocab):
        super().__init__()
        self.args = args
        self.vocab, self.label_vocab = vocab, label_vocab
        self.labels = [i for i, l in enumerate(self.label_vocab._index2word) if l != '-']
        self.iteration = 1
        self.emb = tnn.Embedding(len(vocab), args.d_emb)
        self.bienc = BiEncoder(args.d_emb, args.d_hid)
        self.attn = BiCoattention(args.d_hid*2, args.d_hid*2)
        self.bienc_premise = BiEncoder(args.d_hid*2+args.d_hid*2+args.d_emb, args.d_hid*2)
        self.bienc_hypothesis = BiEncoder(args.d_hid*2+args.d_hid*2+args.d_emb, args.d_hid*2)
        self.self_attn_premise = SelfAttention(args.d_hid*4)
        self.self_attn_hypothesis = SelfAttention(args.d_hid*4)

        d_feat = args.d_hid * 4

        self.scorer = nn.Sequential(
            tnn.Maxout(d_feat * 2, args.d_hid*2*2, args.d_pool),
            nn.Dropout(0.2),
            nn.BatchNorm1d(args.d_hid*2*2),
            tnn.Maxout(args.d_hid*2*2, args.d_hid*2, args.d_pool),
            nn.Dropout(0.25),
            nn.BatchNorm1d(args.d_hid*2),
            tnn.Maxout(args.d_hid*2, len(label_vocab), args.d_pool)
        )

    def forward(self, batch):
        if self.training:
            self.iteration += 1

        emb_premise = self.emb(batch.premise)
        emb_hypothesis = self.emb(batch.hypothesis)

        if self.iteration < 6000:
            emb_premise = emb_premise.detach()
            emb_hypothesis = emb_hypothesis.detach()

        enc_premise, _ = self.bienc(emb_premise, batch.premise_lens)
        enc_hypothesis, _ = self.bienc(emb_hypothesis, batch.hypothesis_lens)

        coattn_premise, coattn_hypothesis = self.attn(enc_premise, batch.premise_lens, enc_hypothesis, batch.hypothesis_lens)

        bienc_premise, _ = self.bienc_premise(torch.cat([coattn_premise, enc_premise, emb_premise], 2), batch.premise_lens)
        bienc_hypothesis, _ = self.bienc_hypothesis(torch.cat([coattn_hypothesis, enc_hypothesis, emb_hypothesis], 2), batch.hypothesis_lens)

        self_premise = self.self_attn_premise(bienc_premise, batch.premise_lens)
        self_hypothesis = self.self_attn_hypothesis(bienc_hypothesis, batch.hypothesis_lens)

        rep = F.dropout(torch.cat([self_premise * self_hypothesis, self_premise - self_hypothesis], 1), 0.3, self.training)
        return self.scorer(rep)

    def predict(self, out, batch):
        _, argmax = out.max(1)
        return self.label_vocab.index2word(list(argmax.squeeze(1).data))

    def compute_loss(self, out, batch):
        return F.cross_entropy(out, batch.labels)

    def measure(self, out, batch, loss=None):
        _, argmax = out.max(1)
        preds = argmax.squeeze(1)
        correct = preds == batch.labels
        m = {'acc': correct.float().mean().data[0]}
        if loss is not None:
            m['loss'] = loss.data[0]
        return m
