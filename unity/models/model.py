import torch

import numpy as np
import random

import os
import re
import copy
from dashboard import reporter
import pendulum
from tqdm import tqdm
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from collections import Counter


class EarlyStopper:

    def __init__(self, metric='loss', better='lower'):
        self.best = None
        self.metric = metric
        self._better = better
        self.time_since_best = 0

    def better(self, scores):
        if self.best is None:
            return True
        else:
            best = self.best[self.metric]
            score = scores[self.metric]
            return score < best if self._better == 'lower' else score > best

    def stop(self, scores):
        if self.metric not in scores:
            self.time_since_best += 1
            return False
        if self.better(scores) or self.best is None:
            self.time_since_best = 0
            return True
        else:
            self.time_since_best += 1
            return False


def apply_lr(optimizer, f_apply):
    before, after = [], []
    for group in optimizer.param_groups:
        if 'lr' in group:
            before.append(group['lr'])
            group['lr'] = f_apply(group['lr'])
            after.append(group['lr'])
    return (sum(before) / len(before)) if before else None, (sum(after) / len(after)) if after else None


class SavePruner:

    def __init__(self, metric='loss', better='lower'):
        self.metric, self.better = metric, better

    def prune(self, d, n_keep=5):
        score_re = re.compile('{}:([0-9\.]+)'.format(self.metric))
        files = []
        for f in os.listdir(d):
            if f.endswith('.t7'):
                scores = [float(s.strip('.')) for s in score_re.findall(f)]
                if scores:
                    files.append([f, scores[0]])
        sorted_files = sorted(files, key=lambda t: t[1], reverse=self.better == 'higher')
        to_delete = sorted_files[n_keep:]
        for f, s in to_delete:
            os.remove(os.path.join(d, f))


class Model(nn.Module):

    def initialize_embeddings(self, E):
        self.emb.weight.data = torch.from_numpy(E)

    def printout_sample(self, s1, s2, pred, gt):
        xx = pred == gt
        wr_sample = [i for i, x in enumerate(xx) if not x]
        if len(wr_sample) > 0:
            ind = wr_sample[random.randrange(1, len(wr_sample))]
            str1 = []
            str1.append(self.label_vocab._index2word[int(gt[ind])])
            str1.append(self.label_vocab._index2word[int(pred[ind])])
            str1.append("----")
            for inp in s1:
                if int(inp.data[ind]) != 1:
                    str1.append(self.vocab._index2word[int(inp.data[ind])])
            for inp in s2:
                if int(inp.data[ind]) != 1:
                    str1.append(self.vocab._index2word[int(inp.data[ind])])
            print(" ".join(str1))

    def get_mask(self, inputs, lens, valid_fill=1, invalid_fill=0):
        mask = Variable(inputs.data.new(*inputs.size()).fill_(valid_fill))
        max_len = max(lens)
        for i, l in enumerate(lens):
            # iterate over batch
            if l < max_len:
                mask.data[i, l:].fill_(invalid_fill)
        return mask

    def extract_pool_features(self, inputs, lens):
        def mask(x, y):
            return self.get_mask(inputs, lens, x, y)

        max_pool = torch.max(inputs.mul(mask(1, 0)) + mask(0, -np.inf), 1)[0].squeeze(1)
        s = torch.sum(inputs.mul(mask(1, 0)), 1).squeeze(1)
        mean_pool = s / mask(1, 0).sum(1).squeeze(1)
        min_pool = torch.min(inputs.mul(mask(1, 0)) + mask(0, np.inf), 1)[0].squeeze(1)
        return torch.cat([max_pool, mean_pool, min_pool], 1)

    def forward(self, batch):
        raise NotImplementedError()

    def predict(self, out, batch):
        raise NotImplementedError()

    def compute_loss(self, out, batch):
        raise NotImplementedError()

    def measure(self, out, batch, loss=None):
        raise NotImplementedError()

    def save(self, fname):
        if self.args.gpu is not None and self.args.gpu > -1:
            with torch.cuda.device(self.args.gpu):
                torch.save(self.state_dict(), fname)
        else:
            torch.save(self.state_dict(), fname)
        return self

    def load(self, fname):
        # loads to CPU
        state = torch.load(fname, map_location=lambda storage, loc: storage.cuda(self.args.gpu))
        self.load_state_dict(state)
        return self

    def run_dataset(self, dataset, Batch, batch_size, gpu, train=True, verbose=False):
        iters = range(0, len(dataset), batch_size)
        if verbose:
            iters = tqdm(iters)
        for i in iters:
            b = Batch(dataset[i:i+batch_size], pad_index=self.vocab['<blank>'])
            b.is_train = train
            b.to_gpu(gpu)
            yield b

    def pred_dataset(self, args, Batch, dataset, verbose=False):
        preds = []
        for bb in self.run_dataset(dataset, Batch, args.batch_size, args.gpu, train=False, verbose=verbose):
            self.eval()
            out = self(bb)
            preds += self.predict(out, bb)
        return preds

    @staticmethod
    def test_vote_ensemble(models, args, Batch, dataset, verbose=False):
        log = reporter.Reporter()
        first_model = models[0]
        preds = []
        ground_truths = []

        for bb in first_model.run_dataset(dataset, Batch, args.batch_size, args.gpu, train=False, verbose=verbose):
            outs = []
            for m in models:
                m.eval()
                p = m.predict(m(bb), bb)
                outs.append(p)
            for i in range(len(bb)):
                votes = Counter([p[i] for p in outs])
                preds.append(votes.most_common(1)[0][0])
                l = bb.labels.data[i]
                ground_truths.append(first_model.label_vocab.index2word(l))
        equal = [1 if p == g else 0 for p, g in zip(preds, ground_truths)]
        return sum(equal) / len(equal)

    @staticmethod
    def test_average_ensemble(models, args, Batch, dataset, verbose=False):
        log = reporter.Reporter()
        first_model = models[0]
        preds = []
        ground_truths = []

        for bb in first_model.run_dataset(dataset, Batch, args.batch_size, args.gpu, train=False, verbose=verbose):
            outs = []
            for m in models:
                m.eval()
                outs.append(F.softmax(m(bb)))
            out = sum(outs) / len(outs)
            _, argmax = out.max(1)
            preds += first_model.label_vocab.index2word(list(argmax.squeeze(1).data))
            ground_truths += first_model.label_vocab.index2word(list(bb.labels.data))
        equal = [1 if p == g else 0 for p, g in zip(preds, ground_truths)]
        return sum(equal) / len(equal)

    @staticmethod
    def test_geometric_mean_ensemble(models, args, Batch, dataset, verbose=False):
        log = reporter.Reporter()
        first_model = models[0]
        preds = []
        ground_truths = []

        for bb in first_model.run_dataset(dataset, Batch, args.batch_size, args.gpu, train=False, verbose=verbose):
            outs = None
            for m in models:
                m.eval()
                out = F.softmax(m(bb))
                outs = out if outs is None else outs.mul(out)
            out = outs.pow(1/len(models))
            _, argmax = out.max(1)
            preds += first_model.label_vocab.index2word(list(argmax.squeeze(1).data))
            ground_truths += first_model.label_vocab.index2word(list(bb.labels.data))
        equal = [1 if p == g else 0 for p, g in zip(preds, ground_truths)]
        return sum(equal) / len(equal)

    def test_dataset(self, args, Batch, dataset, verbose=False):
        log = reporter.Reporter()
        for bb in self.run_dataset(dataset, Batch, args.batch_size, args.gpu, train=False, verbose=verbose):
            self.eval()
            out = self(bb)
            log.add({'dev_' + k: 100*v for k, v in self.measure(out, bb).items()})
        return log.summary()

    def backward(self, loss):
        loss.backward()

    def train_dataset(self, args, Batch, train, dev=None, writers=(),
                      early_stop_on='train_loss', metric_better='lower',
                      save_keys=('train_loss', ),
                      ):
        assert metric_better in {'lower', 'higher'}
        o = optim.Adam(self.parameters(), lr=args.lr)

        log = reporter.Reporter()
        iters = 0
        early_stopper = EarlyStopper(metric=early_stop_on, better=metric_better)
        pruner = SavePruner(metric=early_stop_on, better=metric_better)
        s = ''
        for k in save_keys:
            s += k + ':{' + k + '}.'
        save_format = 'model.epoch:{epoch}.iter:{iteration}.' + s + 't7'

        dev_model = copy.deepcopy(self) if args.running_average else self

        start_time = pendulum.now()
        for epoch in range(args.epoch):
            np.random.shuffle(train)
            for b in self.run_dataset(train, Batch, args.batch_size, args.gpu, train=True):
                self.iteration += 1
                o.zero_grad()
                self.train()
                out = self(b)
                loss = self.compute_loss(out, b)
                self.backward(loss)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm(self.parameters(), args.grad_clip)
                o.step()
                iters += 1

                log.add({'train_' + k: v for k, v in self.measure(out, b, loss=loss).items()})

                new_lr = args.lr
                if args.running_average:
                    for pr, po in zip(dev_model.parameters(), self.parameters()):
                        pr.data = pr.data * 0.999 + po.data * 0.001
                elif args.decay:
                    # scale learning rate by 1 / sqrt(iter)
                    if self.iteration > 10:
                        old_lr, new_lr = apply_lr(o, lambda lr: args.lr / np.sqrt(self.iteration / 100))

                dev_summary = {}
                if dev is not None and iters % args.eval_every == 0:
                    dev_summary = dev_model.test_dataset(args, Batch, dev)

                if iters % args.log_every == 0:
                    train_summary = log.summary()
                    log.clear()

                    elapsed = pendulum.now() - start_time
                    summary = {'epoch': epoch, 'iteration': iters, 'lr': new_lr, 'elapsed': elapsed.in_words(locale='en')}
                    summary.update(train_summary)
                    summary.update(dev_summary)

                    # early stopping
                    if early_stopper.stop(summary):
                        dev_model.save(os.path.join(args.out, save_format.format(**summary)))
                        pruner.prune(args.out)

                    # write logs
                    for w in writers:
                        w.add(summary)
