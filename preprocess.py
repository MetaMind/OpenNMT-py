import argparse
import os
from collections import Counter, defaultdict, OrderedDict

import torch
from torchtext import datasets
from torchtext.data import Dataset, Field, Pipeline
from torchtext import vocab

import onmt


parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")
parser.add_argument('-max_length', default=50, type=int)

parser.add_argument('-train', required=True,
                    help="Path to the training data")
parser.add_argument('-valid', required=True,
                    help="Path to the validation data")
parser.add_argument('-src', required=True,
                     help="Extension for source data")
parser.add_argument('-tgt', required=True,
                     help="Extension for target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

parser.add_argument('-wv_dir', type=str, default='data/glove')
parser.add_argument('-wv_type', type=str, default='')
parser.add_argument('-wv_dim', type=int, default=300)

opt = parser.parse_args()

def filter_by_length(x):
    return len(x.src) < opt.max_length and len(x.tgt) < opt.max_length

def main():

    src = Field(lower=opt.lower, include_lengths=True)
    tgt = Field(init_token=onmt.Constants.BOS_WORD, eos_token=onmt.Constants.EOS_WORD, lower=opt.lower, include_lengths=True)
    train = datasets.TranslationDataset(
        path=opt.train, exts=(opt.src, opt.tgt),
        fields=(src, tgt), filter_pred=filter_by_length)
    valid = datasets.TranslationDataset(
        path=opt.valid, exts=(opt.src, opt.tgt),
        fields=(src, tgt), filter_pred=filter_by_length)
    src.build_vocab(train, max_size=opt.src_vocab_size, wv_dir=opt.wv_dir, wv_type=opt.wv_type, wv_dim=opt.wv_dim)
    tgt.build_vocab(train, max_size=opt.tgt_vocab_size)

    ofile = opt.save_data + '.'
    if opt.lower:
        ofile += 'low.'
    if opt.wv_type:
        ofile += opt.wv_type + '.' + str(opt.wv_dim) + '.'
    ofile += 'pt'

    print('Saving data to \'' + ofile + '\'')
    save_data = {'src': src,
                 'tgt': tgt,
                 'train': train,
                 'valid': valid
                }
    torch.save(save_data, ofile)


if __name__ == "__main__":
    main()
