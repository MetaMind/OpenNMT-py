from models.factored import Model as BaseModel
import torch
from tc import nn as tnn
from torch import nn
import math
from torch.nn import functional as F


class Model(BaseModel):

    def __init__(self, args, vocab, label_vocab):
        super().__init__(args, vocab, label_vocab)
        self.neutral_emb = nn.Parameter(torch.Tensor(2, args.d_hid))
        stdv = 1. / math.sqrt(self.neutral_emb.size(1))
        self.neutral_emb.data.uniform_(-stdv, stdv)

        self.entailment_scorer = nn.Sequential(
            tnn.Maxout(args.d_hid*8*4 + (args.d_hid if self.feed_p_neutral else 0), args.d_hid*2*2, args.d_pool),
            nn.BatchNorm1d(args.d_hid*2*2),
            nn.Dropout(0.2),
            tnn.Maxout(args.d_hid*2*2, args.d_hid*2, args.d_pool),
            nn.BatchNorm1d(args.d_hid*2),
            nn.Dropout(0.25),
            tnn.Maxout(args.d_hid*2, len(self.labels), args.d_pool),
        )

    def forward(self, batch):
        if self.training:
            self.iteration += 1

        rep = self.encode(batch)
        neutral_scores = self.neutral_scorer(rep)

        if self.feed_p_neutral:
            weights = F.softmax(neutral_scores)
            context = weights.mm(self.neutral_emb)
            rep = torch.cat([context, rep], 1)
        entailment_scores = self.entailment_scorer(rep)

        sampled_neutral = sampled_entailment = None
        if self.reinforce:
            sampled_neutral = F.softmax(neutral_scores).multinomial()
            sampled_entailment = F.softmax(entailment_scores).multinomial()
        return neutral_scores, entailment_scores, [sampled_neutral, sampled_entailment]
