from .model import Model as BaseModel
from ..modules import BiEncoderCuDNN, BiAttention, SelfAttention
import torch
from tc import nn as tnn
from torch import nn
from torch.nn import functional as F


class Model(BaseModel):

    def __init__(self, args, dicts):
        super().__init__()
        self.args = args
        self.detach_embedding = args.detach_embedding if hasattr(args, 'detach_embedding') else 0
        self.iteration = 1
        self.num_directions = 2
        self.word_lut = tnn.Embedding(dicts.size(), args.word_vec_size)
        self.bienc = BiEncoderCuDNN(args.word_vec_size, args.rnn_size)
        self.attn = BiAttention(args.rnn_size*2, args.rnn_size*2)
        self.bienc_premise = BiEncoderCuDNN(args.rnn_size*4+args.word_vec_size, args.rnn_size*2)
#        self.bienc_hypothesis = BiEncoder(args.rnn_size*4+args.d_emb, args.rnn_size*2)
#        self.self_attn_premise = SelfAttention(args.rnn_size*4)
#        self.self_attn_hypothesis = SelfAttention(args.rnn_size*4)

#        self.out1 = tnn.Maxout(args.rnn_size*8*4, args.rnn_size*2*2, args.d_pool)
#        self.bn1 = nn.BatchNorm1d(args.rnn_size*2*2)
#        self.out2 = tnn.Maxout(args.rnn_size*2*2, args.rnn_size*2, args.d_pool)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input):
        if self.training:
            self.iteration += 1

        emb_premise = self.word_lut(input[0])

        if self.iteration < self.args.detach_embedding:
            emb_premise = emb_premise.detach()

        enc_premise, _ = self.bienc(emb_premise, input[1])
        attn_premise, _ = self.attn(enc_premise, input[1], enc_premise, input[1])

        bienc_premise, last = self.bienc_premise(torch.cat([attn_premise, emb_premise], 2), input[1])
#        pool_premise = self.extract_pool_features(bienc_premise, batch.premise_lens)
#        self_premise = self.self_attn_premise(bienc_premise, batch.premise_lens)
#        rep = F.dropout(torch.cat([pool_premise, self_premise], 1), 0.2, self.training)

        return last, bienc_premise.transpose(0, 1)
