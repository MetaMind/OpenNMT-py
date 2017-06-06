import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('path')
parser.add_argument('-no-glove', action='store_false', dest='glove')
parser.add_argument('-small-glove', action='store_true', dest='small_glove')
parser.add_argument('-chargram', action='store_true', dest='chargram')
parser.add_argument('-d_hid', default=400, type=int)

args = parser.parse_args()


def ngrams(sentence, n):
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]


def charemb(w):
    chars = ['#BEGIN#'] + list(w) + ['#END#']
    match = {}
    for i in [2, 3, 4]:
        grams = ngrams(chars, i)
        for g in grams:
            g = '{}gram-{}'.format(i, ''.join(g))
            e = None
            if g in kazuma['stoi']:
                e = kazuma['vectors'][kazuma['stoi'][g]]
            if e is not None:
                match[g] = e
    if match:
        emb = sum(match.values()) / len(match)
    else: 
         emb = torch.FloatTensor(100).uniform_(-0.1, 0.1)
    return emb


with open(args.path) as f:
    vocab = [l.split()[0] for l in f] 

import pdb; pdb.set_trace()
if args.small_glove:
    glove = torch.load('glove.6B.100d.pt')
else:
    glove = torch.load('glove.840B.300d.pt')
if args.chargram:
    kazuma = torch.load('kazuma.100d.pt')

vectors = []
for word in vocab:
    vector = torch.FloatTensor(args.d_hid).uniform_(-0.1, 0.1)
    glove_dim = args.d_hid - 100 if args.chargram else args.d_hid
    if args.glove and word in glove['stoi']:
        vector[:glove_dim] = glove['vectors'][glove['stoi'][word]] 
    if args.chargram:
        vector[glove_dim:] = charemb(word)
    vectors.append(vector)

ext = '.glove' if args.glove else ''
ext = '.small-glove' if args.small_glove else ''
ext += '.chargram' if args.chargram else ''
torch.save(torch.stack(vectors), args.path + ext)
