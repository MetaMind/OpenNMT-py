import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

with open(args.path) as f:
    vocab = [l.split()[0] for l in f] 

glove = torch.load('glove.840B.300d.pt')

vectors = []
for word in vocab:
    if word in glove['stoi']:
        vectors.append(glove['vectors'][glove['stoi'][word]]) 
    else:
        vectors.append(torch.FloatTensor(300).uniform_(-0.1, 0.1)) 

torch.save(torch.stack(vectors), args.path + '.glove')
