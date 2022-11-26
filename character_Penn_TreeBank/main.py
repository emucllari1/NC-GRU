import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from utils import *

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/pennchar',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='NCGRU',
                    help='type of recurrent net (LSTM, QRNN, GRU, scoRNN, NCGRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1000,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=150,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=220,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=0,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--max_length', type = int, default = '20',
                    help='k-steps for layernorm')
# new/updated hyperparameters                    
parser.add_argument('--A_lr', type=float, default=1e-3,
                    help='initial A learning rate')
parser.add_argument('--optimizer_A', type=str,  default='rmsprop', choices=['adam','rmsprop','sgd'],
                    help='optimizer_A to use (rmsprop, adam)')    
parser.add_argument('--A_clip', type=float, default=0,
                    help='gradient clipping for A parameters')
parser.add_argument('--W_clip', type=float, default=0,
                    help='gradient clipping for W parameters')
parser.add_argument('--other_clip', type=float, default=0,
                    help='gradient clipping for other (not A and not W) parameters')                    
parser.add_argument('--nono', type = int, default = '5', nargs='+',
                    help='number of negative ones for the D matrix')
parser.add_argument('--neumann', action='store_true',
                    help='enable/disable neumann series approximation for the inverse of I+A')
parser.add_argument('--neumann_order', type = int, default = '2', choices=range(0, 4),
                    help='order of neumann series approximation for the inverse of I+A; if --neumann_order=0 then proceed with torch.inverse instead')    
parser.add_argument('--train_loss_file', type=str,  default='train_loss',
                    help='Name of the training loss file')
parser.add_argument('--valid_loss_file', type=str,  default='valid_loss',
                    help='Name of the validation loss file')
parser.add_argument('--output_file', type=str,  default='output',
                    help='Name of the output everything file')

args = parser.parse_args()
args.tied = True

assert len(args.nono) == args.nlayers

output_file = args.output_file + '.txt'
val_loss_file = args.valid_loss_file + '.txt'
train_loss_file = args.train_loss_file + '.txt'


# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        
###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
        
def print_info_to_file(file1, path):
    f = open(path, "a+")
    f.write('{:}\n'.format(file1))
    f.close()

def loss_to_file(file1, path):
    f = open(path, "a+")
    f.write('{:}\n'.format(file1))
    f.close()

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.nono, \
                        args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, \
                        args.tied, args.max_length)

if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
if args.model in ['scoRNN', 'NCGRU']:
    A_params = [w for name, w in model.named_parameters() if 'A' in name]
    W_params = [w for name, w in model.named_parameters() if 'weight_hn' in name]
    other_params = [w for name, w in model.named_parameters() if ('A' not in name) and ('weight_hn' not in name)]

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)
for name, param in model.named_parameters():
    msg = 'Name: {:};   Param. Dim: {:};    Total # of params: {:}'.format(name, param.size(), param.nelement())
    print_info_to_file(msg, output_file)
    print(msg)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def train(AplusIinv, D, I):
    v_hist = []
    norm_hist = []
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    cur_loss = 0
    if args.model in ['scoRNN', 'NCGRU']:
        if AplusIinv is None:
            AplusIinv = [torch.inverse(I[l]+model.rnns[l].cell_0.A.data) for l in range(args.nlayers)]
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        if args.model in ['scoRNN', 'NCGRU']:
            optimizer_A.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs, emb = model(data, hidden, return_h=True)
        
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

        loss.backward(retain_graph = True)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.W_clip: torch.nn.utils.clip_grad_norm_(W_params, args.W_clip)
        if args.other_clip: torch.nn.utils.clip_grad_norm_(other_params, args.other_clip)
        optimizer.step() # update all except the A variables

        if args.model in ['scoRNN', 'NCGRU']:
            # updating A variable
            for l in range(args.nlayers):
                model.rnns[l].cell_0.A.grad = Cayley_Transform_Deriv(model.rnns[l].cell_0.weight_hn.grad.data, \
                                                                model.rnns[l].cell_0.weight_hn.data, D[l], AplusIinv[l]) 
            if args.A_clip: torch.nn.utils.clip_grad_value_(A_params, args.A_clip)
            optimizer_A.step()
       
            # updating the new W variable
            if (batch % 100 == 0):
                exact_value = True
            else:
                exact_value = False

            for l in range(args.nlayers):
                model.rnns[l].cell_0.weight_hn.data, AplusIinv[l] = makeW(model.rnns[l].cell_0.A.data, D[l], I[l], \
                                                                        AplusIinv[l], model.rnns[l].cell_0.A.grad, \
                                                                        args.A_lr, order=args.neumann_order, exact=exact_value)

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            train_msg= '| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2))
            print(train_msg)
            print_info_to_file(train_msg, output_file)
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    loss_to_file(loss, train_loss_file)
    return cur_loss, AplusIinv
    
# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
dev_msg = '-' * 100
val_hist = []
train_hist = []

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    if args.model in ['scoRNN', 'NCGRU']:
        optimizer_A = None
        AplusIinv = None
    
        D = [torch.diag(torch.cat((torch.ones(args.nhid - args.nono[l]), -torch.ones(args.nono[l])))).cuda() if l != args.nlayers - 1 else \
            (torch.diag(torch.cat((torch.ones(args.emsize - args.nono[l]), -torch.ones(args.nono[l])))).cuda() if args.tied else \
            torch.diag(torch.cat((torch.ones(args.nhid - args.nono[l]), -torch.ones(args.nono[l])))).cuda()) for l in range(args.nlayers)]
        I = [torch.eye(args.nhid).cuda() if l != args.nlayers - 1 else \
            (torch.eye(args.emsize).cuda() if args.tied else \
            torch.eye(args.nhid).cuda()) for l in range(args.nlayers)]            
            
    # Ensure the optimizer is optimizing params, which includes both the model's
    # weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(other_params+W_params, lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(other_params+W_params, lr=args.lr, weight_decay=args.wdecay)
    if args.model in ['scoRNN', 'NCGRU']:        
        if args.optimizer_A == 'sgd':
            optimizer_A = torch.optim.SGD(A_params, lr=args.A_lr)
        elif args.optimizer_A == 'adam':
            optimizer_A = torch.optim.Adam(A_params, lr=args.A_lr)
        elif args.optimizer_A == 'rmsprop':
            optimizer_A = torch.optim.RMSprop(A_params, lr=args.A_lr)
        
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        t_hist, AplusIinv = train(AplusIinv, D, I)
        train_hist.append(t_hist)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            loss_to_file(val_loss2, val_loss_file)
            val_hist.append(val_loss2)
            
            print(dev_msg)
            print_info_to_file(dev_msg, output_file)
            
            val_msg = '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2))
            print_info_to_file(val_msg, output_file)
            print(val_msg)
            
            print(dev_msg)
            print_info_to_file(dev_msg, output_file)

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            loss_to_file(val_loss, val_loss_file)
            val_hist.append(val_loss)
            
            print_info_to_file(dev_msg, output_file)
            print(dev_msg)
            
            val_msg = '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2))
            print_info_to_file(val_msg, output_file)
            print(val_msg)
            
            print(dev_msg)
            print_info_to_file(dev_msg, output_file)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                print_info_to_file('Switching to ASGD', output_file)
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)
# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print_info_to_file('=' * 89, output_file)
test_msg = '| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2))
print_info_to_file(test_msg, output_file)
print(test_msg)
print('=' * 89)
