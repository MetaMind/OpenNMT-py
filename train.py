import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import os


from torchtext.data import BucketIterator

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")
parser.add_argument('-max_length', default=100, type=int,
                    help="""Maximum length of a generated sentence""")

## Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=600,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=300,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=100,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')

# learning rate
parser.add_argument('-lr', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-lr_decay', type=float, default=0.5,
                    help="""If update_lr, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=50,
                    help="""Start decaying every epoch after and including this
                    epoch""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize, pad_token):
    weight = torch.ones(vocabSize)
    weight[pad_token] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def memoryEfficientLoss(tgt, model, outputs, tgts, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    tgts_split = torch.split(tgts, opt.max_generator_batches)
    predictions = []
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, tgts_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = model.generator(out_t)
        loss_t = crit(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        if eval:
            predictions.append(pred_t.view(*tgts.size()).t().data)
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(model.tgt_pad).data).sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    predictions = torch.cat(predictions, 1).tolist() if eval else predictions
    return loss, grad_output, num_correct 


def eval(src, model, criterion, valid_iter, tgt):
    total_loss = 0
    total_tgt_words = 0
    total_num_correct = 0

    model.eval()
    for i, batch in enumerate(valid_iter):
        inputs = batch.src[0]
        tgts = batch.tgt[0][1:] # do not include BOS as target
        batch_size = tgts.size(1)
        outputs = model(batch)
        loss, _, num_correct = memoryEfficientLoss(tgt, model,
                outputs, tgts, criterion, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        num_tgt_words = tgts.data.ne(model.tgt_pad).sum()
        total_tgt_words += num_tgt_words

    model.train()
    return total_loss / total_tgt_words, total_num_correct / total_tgt_words

def sort_key(ex):
    return len(ex.src)

def trainModel(src, tgt, train_iter, valid_iter, model, optim):
    print(model)
    model.train()

    criterion = NMTCriterion(len(tgt.vocab), model.tgt_pad)

    start_time = time.time()
    def train_epoch(epoch):

        total_loss, total_tgt_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        for i, batch in enumerate(train_iter):
            inputs = batch.src[0]
            tgts = batch.tgt[0][1:] # do not include BOS as target
            batch_size = tgts.size(1)

            model.zero_grad()
            outputs = model(batch)
            loss, gradOutput, num_correct = memoryEfficientLoss(tgt, model,
                    outputs, tgts, criterion)

            outputs.backward(gradOutput)

            # update the parameters
            optim.step()

            num_tgt_words = tgts.data.ne(model.tgt_pad).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_tgt_words += num_tgt_words
            report_src_words += sum(batch.src[1])
            total_loss += loss
            total_num_correct += num_correct
            total_tgt_words += num_tgt_words
            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                      (epoch, i+1, len(train_iter),
                      report_num_correct / report_tgt_words * 100,
                      math.exp(report_loss / report_tgt_words),
                      report_src_words/(time.time()-start),
                      report_tgt_words/(time.time()-start),
                      time.time()-start_time))

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()

        return total_loss / total_tgt_words, total_num_correct / total_tgt_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc = train_epoch(epoch)

        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % train_acc)

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(src, model, criterion, valid_iter, tgt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc*100))

        #  (3) update the learning rate
        optim.updateLearningRate(valid_loss, epoch)

        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'src': src,
            'tgt': tgt,
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }

        save_model, ext = os.path.splitext(os.path.basename(opt.data))
        details = [save_model]
        details.append(str(opt.layers) + 'l')
        if opt.brnn:
            details.append('brnn')
        details.append(str(opt.word_vec_size) + 'wv')
        details.append(str(opt.rnn_size) + 'h')
        details.append(str(opt.batch_size) + 'bs')
        details.append(str(opt.dropout) + 'dp')
        details.append(str(opt.optim))
        details.append(str(opt.lr) + 'lr')
        save_model = '.'.join(details)
        
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (save_model, 100*valid_acc, valid_ppl, epoch))

def main():

    print("Loading data from '%s'" % opt.data)

    data = torch.load(opt.data)
    src = data['src']
    tgt = data['tgt']
    train = data['train']
    valid = data['valid']

    load_from = opt.train_from or opt.train_from_state_dict
    if load_from:
        print('Loading fields from checkpoint at %s' % load_from)
        checkpoint = torch.load(load_from)
        src = checkpoint['src']
        tgt = checkpoint['tgt']

    train_iter = BucketIterator(
        train, opt.batch_size,
        device=opt.gpus[0], repeat=False, scale=len(train), sort_key=sort_key)

    valid_iter = BucketIterator(
        valid, opt.batch_size,
        train=False, device=opt.gpus[0], repeat=False, sort_key=sort_key)

    print(' * vocabulary size. source = {}; target = {}'.format(len(src.vocab), len(tgt.vocab)))
    print(' * number of training sentences. {}'.format(len(train)))
    print(' * number of validation sentences. {}'.format(len(valid)))
    print(' * maximum batch size. {}'.format(opt.batch_size))

    print('Building model...')

    encoder = onmt.Models.Encoder(opt, src)
    decoder = onmt.Models.Decoder(opt, tgt)

    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, len(tgt.vocab)),
        nn.LogSoftmax())

    model = onmt.Models.NMTModel(encoder, decoder)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items() if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s' % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        optim = onmt.Optim(
            opt.optim, opt.lr, opt.max_grad_norm,
            lr_decay=opt.lr_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    model.src_pad = src.vocab.stoi[onmt.Constants.PAD_WORD]
    model.tgt_pad = tgt.vocab.stoi[onmt.Constants.PAD_WORD]
    model.tgt_eos = tgt.vocab.stoi[onmt.Constants.EOS_WORD]
    model.tgt_bos = tgt.vocab.stoi[onmt.Constants.BOS_WORD]

    trainModel(src, tgt, train_iter, valid_iter, model, optim)


if __name__ == "__main__":
    main()
