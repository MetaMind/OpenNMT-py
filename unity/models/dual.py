from models.model import Model as BaseModel
from modules import BiEncoderCuDNN, BiEncoder, BiAttention, SelfAttention
import torch
from tc import nn as tnn
from torch import nn
from torch.nn import functional as F

class Model(BaseModel):

    def __init__(self, args, vocab, label_vocab=None):
        super().__init__()
        self.args = args
        self.cudnn = args.cudnn or args.mt
        self.dual = self.args.task in ['snli', 'sick', 'sim']
        self.metric = 'mse' if self.args.task == 'sim' else 'accuracy'
        self.vocab, self.label_vocab = vocab, label_vocab
        self.labels = [i for i, l in enumerate(self.label_vocab._index2word) if l != '-']
        self.iteration = 1
        self.emb = tnn.Embedding(len(vocab), args.d_emb)
        self.bienc = BiEncoder(args.d_emb, args.d_hid) if not self.cudnn else BiEncoderCuDNN(args.d_emb, args.d_hid, layers=args.layers)
        if args.mt:
            self.bienc.from_mt(args.mt)
        self.attn = BiAttention(args.d_hid*2, args.d_hid*2, proj= not args.dot)
        self.bienc_premise = BiEncoder(args.d_hid*4+args.d_emb, args.d_hid*2)
        self.bienc_hypothesis = BiEncoder(args.d_hid*4+args.d_emb, args.d_hid*2)

        start_size = args.d_hid*8*4
        if not self.dual:
            start_size /= 2
        if self.args.self_attn:
            self.self_attn_premise = SelfAttention(args.d_hid*4)
            self.self_attn_hypothesis = SelfAttention(args.d_hid*4)
        else:
            start_size -= args.d_hid*8 if self.dual else args.d_hid*4

        self.out1 = tnn.Maxout(int(start_size), args.d_hid*2*2, args.d_pool)
        self.bn1 = nn.BatchNorm1d(args.d_hid*2*2)
        self.out2 = tnn.Maxout(args.d_hid*2*2, args.d_hid*2, args.d_pool)
        self.bn2 = nn.BatchNorm1d(args.d_hid*2)
        self.out3 = tnn.Maxout(args.d_hid*2, len(label_vocab), args.d_pool)

    def forward(self, batch):
        if self.dual:
            return self.dual_forward(batch)
        else:
            return self.single_forward(batch)

    def single_forward(self, batch):
        if self.training:
            self.iteration += 1

        emb_premise = self.emb(batch.premise)

        if self.iteration < self.args.detach_embed:
            emb_premise = emb_premise.detach()

        enc_premise, _ = self.bienc(emb_premise, batch.premise_lens)

        if self.iteration < self.args.detach_encoder:
            enc_premise = enc_premise.detach()

        attn_premise, _ = self.attn(enc_premise, batch.premise_lens, enc_premise, batch.premise_lens)

        bienc_premise, _ = self.bienc_premise(torch.cat([attn_premise, emb_premise], 2), batch.premise_lens)
        pool_premise = self.extract_pool_features(bienc_premise, batch.premise_lens)
        if self.args.self_attn:
            self_premise = self.self_attn_premise(bienc_premise, batch.premise_lens)
            rep = F.dropout(torch.cat([pool_premise, self_premise], 1), 0.2, self.training)
        else:
            rep = F.dropout(torch.cat([pool_premise], 1), 0.2, self.training)

        o1 = F.dropout(self.bn1(self.out1(rep)), 0.2, self.training)
        o2 = F.dropout(self.bn2(self.out2(o1)), 0.25, self.training)
        o3 = self.out3(o2)
        return o3

    def dual_forward(self, batch):
        if self.training:
            self.iteration += 1

        emb_premise = self.emb(batch.premise)
        emb_hypothesis = self.emb(batch.hypothesis)

        if self.iteration < self.args.detach_embed:
            emb_premise = emb_premise.detach()
            emb_hypothesis = emb_hypothesis.detach()

        enc_premise, _ = self.bienc(emb_premise, batch.premise_lens)
        enc_hypothesis, _ = self.bienc(emb_hypothesis, batch.hypothesis_lens)

        attn_premise, attn_hypothesis = self.attn(enc_premise, batch.premise_lens, enc_hypothesis, batch.hypothesis_lens)

        bienc_premise, _ = self.bienc_premise(torch.cat([attn_premise, emb_premise], 2), batch.premise_lens)
        bienc_hypothesis, _ = self.bienc_hypothesis(torch.cat([attn_hypothesis, emb_hypothesis], 2), batch.hypothesis_lens)

        pool_premise = self.extract_pool_features(bienc_premise, batch.premise_lens)
        pool_hypothesis = self.extract_pool_features(bienc_hypothesis, batch.hypothesis_lens)

        if self.args.self_attn:
            self_premise = self.self_attn_premise(bienc_premise, batch.premise_lens)
            self_hypothesis = self.self_attn_hypothesis(bienc_hypothesis, batch.hypothesis_lens)
            rep = F.dropout(torch.cat([pool_premise, self_premise, pool_hypothesis, self_hypothesis], 1), 0.2, self.training)
        else:
            rep = F.dropout(torch.cat([pool_premise, pool_hypothesis], 1), 0.2, self.training)

        o1 = F.dropout(self.bn1(self.out1(rep)), 0.2, self.training)
        o2 = F.dropout(self.bn2(self.out2(o1)), 0.25, self.training)
        o3 = self.out3(o2)
        return o3

    def predict(self, out, batch):
        _, argmax = out.max(1)
        return self.label_vocab.index2word(list(argmax.squeeze(1).data))

    def compute_loss(self, out, batch):
        return F.cross_entropy(out, batch.labels) if self.args.task != 'sim' else self.kl_loss(out, batch.labels)

    def measure(self, out, batch, loss=None):
        return self.accuracy(out, batch.labels, loss) if self.args.task != 'sim' else self.mse(out, batch.labels, loss)

    def accuracy(self, out, labels, loss=None):
        _, argmax = out.max(1)
        preds = argmax.squeeze(1)
        correct = preds == labels
        m = {'acc': correct.float().mean().data[0]}
        if loss is not None:
            m['loss'] = loss.data[0]
        return m

    def kl_loss(self, out, labels):
        dist = torch.nn.functional.softmax(out)
        gold_high_val = torch.ceil(labels) - 1
        gold_low_val = torch.floor(labels) - 1
        gold_high_prob = labels - gold_high_val
        gold_low_prob = 1 - gold_high_prob
        pred_low_prob = torch.gather(dist,  1, gold_low_val.long().unsqueeze(1)) + 1e-12
        pred_high_prob = torch.gather(dist, 1, gold_high_val.long().unsqueeze(1)) + 1e-12
        low_contribution = gold_low_prob *  - torch.log(pred_low_prob)
        high_contribution = gold_high_prob *  - torch.log(pred_high_prob)
        loss = low_contribution + high_contribution
        loss = loss.contiguous().sum() / labels.size(0)
        return loss

    def mse(self, out, labels, loss=None):
        dist = torch.nn.functional.softmax(out)
        prediction = dist.cpu().data.mm(torch.range(1, 5).view(5, 1))
        mse = torch.nn.MSELoss()(torch.autograd.Variable(prediction), labels.cpu())
        m = {'mse': mse.data[0]}
        if loss is not None:
            m['loss'] = loss.data[0]
        return  m


