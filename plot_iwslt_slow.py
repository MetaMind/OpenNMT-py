def split_experiments(f):
    experiments = []
    epoch = 30
    for l in f:
        if l:
            split = l.split()
            if len(split) < 3:
                continue
            first, second, rest = split[0], split[1], split[:1]
            if first == 'Epoch':
                curr_epoch = int(second[:-1])
                if curr_epoch < epoch:
                    experiments.append([])
                epoch = curr_epoch
            if first == 'Epoch' or first == 'Validation':
                experiments[-1].append(l)
    return experiments

import visdom
import numpy as np

viz = visdom.Visdom()
train_ppls = []
val_ppls = []
train_accs = []
val_accs = []
val_iters = []
train_iters = []

with open('iwslt_slow_20_log') as f:
    exps = split_experiments(f)
    for exp in exps:
        training = [x for x in exp if x.split()[0] == 'Epoch']
        train_ppls.append([float(x.split()[7][:-1]) for x in training])
        train_accs.append([float(x.split()[5][:-1]) for x in training])
        iters_per_epoch = int(training[0].split()[3][:-1])
        train_iters.append([int(x.split()[2][:-1]) + (int(x.split()[1][:-1])-1)*iters_per_epoch for x in training])
        epoch = list(set([int(x.split()[1][:-1]) for x in training]))
        val_iters.append([x*iters_per_epoch for x in epoch])
        validation = [x for x in exp if x.split()[0] == 'Validation']
        val_ppls.append([float(x.split()[-1].strip()) for x in validation if 'erplex' in x])
        val_accs.append([float(x.split()[-1].strip()) for x in validation if 'ccur' in x])

from scipy.signal import savgol_filter
train_x = np.column_stack(train_iters)
val_x = np.column_stack(val_iters)
viz.line(Y=savgol_filter(np.column_stack(train_ppls), 51, 3, axis=0), X=train_x, opts=dict(title='Train Perplexity', legend=['beta=5', 'beta=0', 'beta=10', 'beta=15', 'beta=20'], ytickmin=0.01, ytickmax=10.0, xtickmin=20000, xtickmax=100000))
viz.line(Y=np.column_stack(val_ppls), X=val_x, opts=dict(title='Validation Perplexity', legend=['beta=5', 'beta=0', 'beta=10', 'beta=15', 'beta=20'], ytickmin=4, ytickmax=8))
viz.line(Y=savgol_filter(np.column_stack(train_accs), 51, 3, axis=0), X=train_x, opts=dict(title='Train Accuracy', legend=['beta=5', 'beta=0', 'beta=10', 'beta=15', 'beta=20'], ytickmin=50, ytickmax=100))
viz.line(Y=np.column_stack(val_accs), X=val_x, opts=dict(title='Validation Accuracy', legend=['beta=5', 'beta=0', 'beta=10', 'beta=15', 'beta=20'], ytickmin=50, ytickmax=80))
