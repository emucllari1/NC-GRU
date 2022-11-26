import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
import math

##################################################
#   scoRNN Model
##################################################
class scoRNN(nn.Module):

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, D=None, **kwargs):
        super(scoRNN, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.D = D

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size, D=D,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next = cell(input_=input_[time], hx=hx)
            hx_next = h_next
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = torch.zeros((input_.size(1), self.hidden_size), device='cuda')
        h_n = []
        
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            hx_layer = hx
            if layer == 0:
                layer_output, layer_h_n = scoRNN._forward_rnn(
                    cell=cell, input_=input_, length=length, hx=hx_layer)
            else:
                layer_output, layer_h_n = scoRNN._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)

        return output, h_n

class scoRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True, D = None):

        super(scoRNNCell, self).__init__()
        print(input_size, hidden_size, D.size())
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        self.weight_in = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        
        self.weight_hn = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.A = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        
        self.bias_C = nn.Parameter(torch.FloatTensor(hidden_size))
        
        # Diagonal matrix D initialization
        if D is None:
            self.D = torch.eye(hidden_size, dtype=torch.float)
        else:
            self.D = D.type(torch.float)
        
        # Initialization of skew-symmetric matrix
        s_size = int(math.floor(hidden_size/2.0))
        s = torch.FloatTensor(s_size).uniform_(0, math.pi/2.0)
        s = - torch.sqrt((1.0 - torch.cos(s))/(1.0 + torch.cos(s)))
        z = torch.zeros(s.size(0))
        if hidden_size % 2 == 0:
            diag = torch.stack((s,z),dim=1).view(2*s.size(0))[:-1]
        else:
            diag = torch.stack((s,z),dim=1).view(2*s.size(0))
        A_init = torch.diag(diag, diagonal=1)
        A_init = A_init - A_init.transpose(0,1)
        self.A_init = A_init.type(torch.float)
        I = torch.eye(hidden_size)
        Z_init = torch.mm(torch.inverse(I + self.A_init), (I - self.A_init))
        self.W_init = torch.mm(Z_init, self.D)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            std = 1.0 / math.sqrt(self.hidden_size)
            for name, w in self.named_parameters():
                if ('weight_ih' in name) or ('weight_hn' in name):
                    w.data.uniform_(-std, std)
            
            init.uniform_(self.weight_in, -0.01, 0.01)

            self.weight_hn.data = self.W_init.data
            self.A.data = self.A_init.data
            
            init.constant(self.bias_C.data, val=0)
            

    def forward(self, input_, hx):        
        i_n = torch.mm(input_, self.weight_in)
        h_n = torch.mm(hx, self.weight_hn) 
        
        candidate = i_n + h_n
        newgate = torch.relu(torch.abs(candidate) + self.bias_C) * torch.sign(candidate)
        
        return newgate



###################################################
#   NC-GRU Model
###################################################        
class NCGRU(nn.Module):

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, D=None, **kwargs):
        super(NCGRU, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.D = D

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size, D=D,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next = cell(input_=input_[time], hx=hx)
            hx_next = h_next
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = torch.zeros((input_.size(1), self.hidden_size), device='cuda')
        h_n = []
        
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            hx_layer = hx
            if layer == 0:
                layer_output, layer_h_n = NCGRU._forward_rnn(
                    cell=cell, input_=input_, length=length, hx=hx_layer)
            else:
                layer_output, layer_h_n = NCGRU._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)

        return output, h_n

class NCGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True, D = None):

        super(NCGRUCell, self).__init__()
        print(input_size, hidden_size, D.size())
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Linear(input_size, 2 * hidden_size, bias=use_bias)
        self.weight_hh = nn.Linear(hidden_size, 2 * hidden_size, bias=use_bias)
        
        self.weight_in = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        
        self.weight_hn = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.A = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        
        self.bias_C = nn.Parameter(torch.FloatTensor(hidden_size))
        
        # Diagonal matrix D initialization
        if D is None:
            self.D = torch.eye(hidden_size, dtype=torch.float)
        else:
            self.D = D.type(torch.float)
        
        # Initialization of skew-symmetric matrix
        s_size = int(math.floor(hidden_size/2.0))
        s = torch.FloatTensor(s_size).uniform_(0, math.pi/2.0)
        s = - torch.sqrt((1.0 - torch.cos(s))/(1.0 + torch.cos(s)))
        z = torch.zeros(s.size(0))
        
        if hidden_size % 2 == 0:
            diag = torch.stack((s,z),dim=1).view(2*s.size(0))[:-1]
        else:
            diag = torch.stack((s,z),dim=1).view(2*s.size(0))
        A_init = torch.diag(diag, diagonal=1)
        A_init = A_init - A_init.transpose(0,1)
        self.A_init = A_init.type(torch.float)
        I = torch.eye(hidden_size)
        Z_init = torch.mm(torch.inverse(I + self.A_init), (I - self.A_init))
        self.W_init = torch.mm(Z_init, self.D)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            std = 1.0 / math.sqrt(self.hidden_size)
            for name, w in self.named_parameters():
                if ('weight_ih' in name) or ('weight_hn' in name):
                    w.data.uniform_(-std, std)
            
            init.uniform_(self.weight_in, -0.01, 0.01)

            self.weight_hn.data = self.W_init.data
            self.A.data = self.A_init.data
            
            init.constant(self.bias_C.data, val=0)
            

    def forward(self, input_, hx):
        gate_x = self.weight_ih(input_)
        gate_h = self.weight_hh(hx)
        
        i_r, i_i = gate_x.chunk(2,1)
        h_r, h_i = gate_h.chunk(2,1)
        
        i_n = torch.mm(input_, self.weight_in)
        h_n = torch.mm(hx, self.weight_hn)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        
        # Note, might need to switch matrix multiplication and entrywise multiplication
        candidate = i_n + (resetgate * h_n)
        newgate = torch.relu(torch.abs(candidate) + self.bias_C) * torch.sign(candidate)
        
        hy = newgate + inputgate * (hx - newgate)
        
        return hy 

        
        

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, nono, dropout=0, dropouth=0, dropouti=0, dropoute=0, wdrop=0, tie_weights=False, max_length = 1):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.D = [torch.diag(torch.cat((torch.ones(nhid - nono[l]), -torch.ones(nono[l])))) if l != nlayers - 1 else \
            (torch.diag(torch.cat((torch.ones(ninp - nono[l]), -torch.ones(nono[l])))) if tie_weights else \
            torch.diag(torch.cat((torch.ones(nhid - nono[l]), -torch.ones(nono[l]))))) for l in range(nlayers)]

        assert rnn_type in ['LSTM', 'QRNN', 'GRU', 'scoRNN', 'NCGRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [LSTM(LSTMCell, ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0, max_length = max_length) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop) for i, rnn in enumerate(self.rnns)]
        if rnn_type == 'scoRNN':
            self.rnns = [scoRNN(scoRNNCell, ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0, D=self.D[l]) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]                
        if rnn_type == 'NCGRU':
            self.rnns = [NCGRU(NCGRUCell, ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0, D=self.D[l]) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)
        

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb

        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs, emb
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU' or self.rnn_type == 'scoRNN' or self.rnn_type == 'NCGRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
