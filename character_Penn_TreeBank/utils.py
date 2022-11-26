import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
    
###############################################################################
# scoRNN/NC-GRU parts
###############################################################################
    
def Cayley_Transform_Deriv(grads, W, D, AplusIinv):
    # Calculate Update Matrix
    V = torch.mm(torch.mm(AplusIinv.transpose(0,1), grads), D + W.transpose(0,1)) 
    DFA = V.transpose(0,1) - V
    return DFA    
    
def neuman_series_appx(I, AplusIinv, DFA, lr, order=1):
    Ainv_deltaA = torch.mm(AplusIinv, lr*DFA)
    if order==2:
        return torch.mm(I + Ainv_deltaA + torch.matrix_power(Ainv_deltaA, 2), AplusIinv)
    elif order==3:
        return torch.mm(I + Ainv_deltaA + torch.matrix_power(Ainv_deltaA, 2) + torch.matrix_power(Ainv_deltaA, 3), AplusIinv)        
    else:
        return torch.mm(I + Ainv_deltaA, AplusIinv)
    
def makeW(A, D, I, AplusIinv, DFA, lr, order=1, exact=False):
    # Computing hidden to hidden matrix using the relation 
    # W ~= (I + A)^-1*(I – A)*D
    if exact or order==0:
        Temp = torch.inverse(I+A)
        W = torch.mm(torch.mm(Temp, I - A), D)
        return W, Temp
    else:
        Temporary = neuman_series_appx(I, AplusIinv, DFA, lr, order=order)
        W = torch.mm(torch.mm(Temporary, I - A), D)
        return W, Temporary    
        
        
        
