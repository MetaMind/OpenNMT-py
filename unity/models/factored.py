from models.model import Model as BaseModel
from modules import BiEncoder, BiAttention, SelfAttention
import torch
from tc import nn as tnn
from torch import nn, autograd
from torch.nn import functional as F


class Model(BaseModel):

    def __init__(self, args, vocab, label_vocab):
        super().__init__()
        self.args = args
        self.vocab, self.label_vocab = vocab, label_vocab
        self.labels = [i for i, l in enumerate(self.label_vocab._index2word) if l != '-']
        self.neutral_class, self.entailment_class, self.contradiction_class = self.label_vocab.word2index(['neutral', 'entailment', 'contradiction'])
        self.iteration = 1
        self.emb = tnn.Embedding(len(vocab), args.d_emb)
        self.bienc = BiEncoder(args.d_emb, args.d_hid)
        self.attn = BiAttention(args.d_hid*2, args.d_hid*2)
        self.bienc_premise = BiEncoder(args.d_hid*4+args.d_emb, args.d_hid*2)
        self.bienc_hypothesis = BiEncoder(args.d_hid*4+args.d_emb, args.d_hid*2)
        self.self_attn_premise = SelfAttention(args.d_hid*4)
        self.self_attn_hypothesis = SelfAttention(args.d_hid*4)

        self.entailment_weight = args.entailment_weight
        self.ce_weight = args.ce_weight
        self.feed_p_neutral = args.feed_p_neutral
        self.reinforce = args.reinforce
        self.reward = None

        self.neutral_scorer = nn.Sequential(
            tnn.Maxout(args.d_hid*8*4, args.d_hid*2*2, args.d_pool),
            nn.BatchNorm1d(args.d_hid*2*2),
            nn.Dropout(0.2),
            tnn.Maxout(args.d_hid*2*2, args.d_hid*2, args.d_pool),
            nn.BatchNorm1d(args.d_hid*2),
            nn.Dropout(0.25),
            tnn.Maxout(args.d_hid*2, 2, args.d_pool),
        )

        self.entailment_scorer = nn.Sequential(
            tnn.Maxout(args.d_hid*8*4 + (2 if self.feed_p_neutral else 0), args.d_hid*2*2, args.d_pool),
            nn.BatchNorm1d(args.d_hid*2*2),
            nn.Dropout(0.2),
            tnn.Maxout(args.d_hid*2*2, args.d_hid*2, args.d_pool),
            nn.BatchNorm1d(args.d_hid*2),
            nn.Dropout(0.25),
            tnn.Maxout(args.d_hid*2, len(self.labels), args.d_pool),
        )

    def encode(self, batch):
        emb_premise = self.emb(batch.premise)
        emb_hypothesis = self.emb(batch.hypothesis)

        if self.iteration < 6000:
            emb_premise = emb_premise.detach()
            emb_hypothesis = emb_hypothesis.detach()

        enc_premise, _ = self.bienc(emb_premise, batch.premise_lens)
        enc_hypothesis, _ = self.bienc(emb_hypothesis, batch.hypothesis_lens)

        attn_premise, attn_hypothesis = self.attn(enc_premise, batch.premise_lens, enc_hypothesis, batch.hypothesis_lens)

        bienc_premise, _ = self.bienc_premise(torch.cat([attn_premise, emb_premise], 2), batch.premise_lens)
        self_premise = self.self_attn_premise(bienc_premise, batch.premise_lens)

        bienc_hypothesis, _ = self.bienc_hypothesis(torch.cat([attn_hypothesis, emb_hypothesis], 2), batch.hypothesis_lens)
        self_hypothesis = self.self_attn_hypothesis(bienc_hypothesis, batch.hypothesis_lens)

        pool_premise = self.extract_pool_features(bienc_premise, batch.premise_lens)
        pool_hypothesis = self.extract_pool_features(bienc_hypothesis, batch.hypothesis_lens)

        rep = F.dropout(torch.cat([pool_premise, self_premise, pool_hypothesis, self_hypothesis], 1), 0.2, self.training)
        return rep

    def forward(self, batch):
        if self.training:
            self.iteration += 1

        rep = self.encode(batch)
        neutral_scores = self.neutral_scorer(rep)

        if self.feed_p_neutral:
            rep = torch.cat([F.softmax(neutral_scores), rep], 1)
        entailment_scores = self.entailment_scorer(rep)

        sampled_neutral = sampled_entailment = None
        if self.reinforce and self.iteration > 8000:
            sampled_neutral = F.softmax(neutral_scores).multinomial()
            sampled_entailment = F.softmax(entailment_scores).multinomial()
        return neutral_scores, entailment_scores, [sampled_neutral, sampled_entailment]

    def predict(self, out, batch):
        neutral_scores, entailment_scores, stochastic_vars = out
        pred_neutral = list(neutral_scores.max(1)[1].squeeze(1).data)
        pred_entailment = list(entailment_scores.max(1)[1].squeeze(1).data)
        predictions = ['neutral' if n > 0.5 else self.label_vocab.index2word(e) for n, e in zip(pred_neutral, pred_entailment)]
        return predictions

    def compute_loss(self, out, batch):
        neutral_scores, entailment_scores, stochastic_vars = out
        if self.reinforce and self.iteration > 8000:
            sampled_neutral, sampled_entailment = stochastic_vars
        else:
            stochastic_vars = []

        neutral_labels = (batch.labels == self.neutral_class).long()
        neutral_loss = F.cross_entropy(neutral_scores, neutral_labels)
        entailment_loss = F.cross_entropy(entailment_scores, batch.labels)
        ce_loss = neutral_loss + self.entailment_weight * entailment_loss

        if self.reinforce and self.iteration > 8000:
            baseline_correct = self.compute_correctness(neutral_scores.max(1)[1].squeeze(1), entailment_scores.max(1)[1].squeeze(1), batch.labels)
            sampled_correct = self.compute_correctness(sampled_neutral.squeeze(1), sampled_entailment.squeeze(1), batch.labels)
            reward = sampled_correct - baseline_correct
            if self.reward is None:
                self.reward = reward.mean()
            else:
                self.reward = 0.9 * self.reward + 0.1 * reward.mean()
            sampled_neutral.reinforce(reward.unsqueeze(1))
            sampled_entailment.reinforce(reward.unsqueeze(1))

        return ce_loss, stochastic_vars

    def backward(self, loss):
        ce_loss, stochastic_vars = loss
        ce_loss *= self.ce_weight
        autograd.backward([ce_loss] + stochastic_vars, [ce_loss.data.new(1).fill_(1)] + [None] * len(stochastic_vars))

    def compute_correctness(self, neutral, entailment, labels):
        pred_neutral = list(neutral.data)
        pred_entailment = list(entailment.data)
        for i, p in enumerate(pred_neutral):
            if p > 0.5:
                pred_entailment[i] = self.neutral_class
        correct = labels.data.new(pred_entailment) == labels.data
        return correct.float()

    def measure(self, out, batch, loss=None):
        neutral_scores, entailment_scores, stochastic_vars = out
        correct = self.compute_correctness(neutral_scores.max(1)[1].squeeze(1), entailment_scores.max(1)[1].squeeze(1), batch.labels)
        m = {'acc': correct.float().mean()}
        if loss is not None:
            ce_loss, stochastic_vars = loss
            m['loss'] = ce_loss.data[0]
            if self.reinforce and self.iteration > 8000:
                m['reward'] = self.reward
        return m
