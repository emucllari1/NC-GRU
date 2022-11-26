# This task tests the ability of the network 
# '' The input data consists of 10 pairs of different types of parenthesis
# combined with some noise data in between and it is given as a on-hot
# encoding vector of length T.
# The output data given as a one-hot encoding vector,
# count the number of the unpaired parenthesis in the input data.''

import time
import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell
import matplotlib.pylab as plt
import sys
import argparse
import os
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
parser = argparse.ArgumentParser(description='Parenthesis task Problem')
parser.add_argument('--model_used', default='LSTM', type=str, choices=['LSTM', 'GRU', 'NCGRU', 'scoRNN', 'goru'], 
                        help='model type')
parser.add_argument('--N_TRAIN_STEPS', type=int, default=20000, 
                        help=' ')
parser.add_argument('--MAX_COUNT', type=int, default=5, help='max_count: level at which outstanding opening parenthesis stop being added')
parser.add_argument('--HIDDEN_DIM', type=int, default=50, help='Hidden layer size')
parser.add_argument('--n_neg_ones', type=int, default=20, help='No. of -1s to put on diagonal of scaling matrix')
#parser.add_argument('--n_clasess', type=int, default=1, help='One output (sum of two numbers)')
parser.add_argument('--T', type=int, default=100, help=' ')
parser.add_argument('--BATCH_SIZE', type=int, default=16, help=' ')
parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-4, help='')
parser.add_argument('--in_out_optimizer', type=str, default='adam', choices=['adam', 'adagrad', 'rmsprop', 'sgd'], 
                        help=' ')
parser.add_argument('--in_out_lr', type=float, default=1e-3, help=' ')
parser.add_argument('--A_optimizer', type=str, default='adam', choices=['adam', 'adagrad', 'rmsprop', 'sgd'],
                        help=' ')
parser.add_argument('--A_lr', type=float, default=1e-3, help=' ')
parser.add_argument('--two_orthogonal', action='store_true', 
                        help='if this argument is called then the NC-GRU(U_c, U_r) is implemented')
args = parser.parse_args()

if args.model_used == 'NCGRU':
    from NCGRU import *
elif args.model_used == 'scoRNN':
    from scoRNN import *
    
from goru import GORUCell
# Network Parameters
model_used = args.model_used
N_TRAIN_STEPS = args.N_TRAIN_STEPS            
MAX_COUNT = args.MAX_COUNT          
HIDDEN_DIM = args.HIDDEN_DIM        
n_neg_ones = args.n_neg_ones        
#n_classes = args.n_classes        
T = args.T
BATCH_SIZE = args.BATCH_SIZE
WEIGHT_DECAY = args.WEIGHT_DECAY 
# Input/Output parameters
in_out_optimizer = args.in_out_optimizer #adam
in_out_lr = args.in_out_lr #1e-3
# Hidden to hidden parameters
A_optimizer = args.A_optimizer #rmsprop
A_lr = args.A_lr #1e-4
#two orthogonal 
two_orthogonal = args.two_orthogonal

#goru hyperparameters
capacity = 4
fft = False

def loss_to_file(file1, path):
    f = open(path, "a+")
    f.write('{:}\n'.format(file1))
    f.close()


# Name of save string/scaling matrix
if model_used == 'LSTM':
    savestring = 'parenthesis_task_{:s}_{:d}_{:s}_{:.1e}_batch_size_{:d}'.format(model_used, HIDDEN_DIM, \
                 in_out_optimizer, in_out_lr, BATCH_SIZE)
if model_used == 'GRU':
    savestring = 'parenthesis_task_{:s}_{:d}_{:s}_{:.1e}_batch_size={:d}'.format(model_used, HIDDEN_DIM, \
                 in_out_optimizer, in_out_lr, BATCH_SIZE)
if model_used == 'scoRNN' or model_used == 'NCGRU':
    savestring = 'parenthesis_task_{:s}_{:d}_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_batch_size_{:d}'.format(model_used, \
                 HIDDEN_DIM, n_neg_ones, in_out_optimizer, in_out_lr, \
                 A_optimizer, A_lr, BATCH_SIZE)
    D = np.diag(np.concatenate([np.ones(HIDDEN_DIM - n_neg_ones), \
        -np.ones(n_neg_ones)]))
if model_used == 'goru':
    savestring = 'parenthesis_task_{:s}_{:d}_{:s}_{:.1e}_batch_size_{:d}'.format(model_used, HIDDEN_DIM, \
                 in_out_optimizer, in_out_lr, BATCH_SIZE)
print('\n' + savestring + '\n')


class rnn_cell_model(object):
    def __init__(self, n_tokens, hidden_dim, target_dim):
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

    def fprop(self, inputs):
        with tf.variable_scope('model', values=[inputs]):
            one_hot_inputs = tf.one_hot(inputs, self.n_tokens, axis=-1)
            with tf.variable_scope('rnn', values=[inputs]):
                if model_used == 'LSTM':
                    states,_=dynamic_rnn(cell=tf.contrib.rnn.BasicLSTMCell(self.hidden_dim,activation=tf.nn.tanh),inputs=one_hot_inputs,      dtype=tf.float32)
                if model_used == 'GRU':
                    states, _ = dynamic_rnn(cell=tf.contrib.rnn.GRUCell(self.hidden_dim), inputs=one_hot_inputs, dtype=tf.float32)
                if model_used == 'scoRNN':
                    states, _ = dynamic_rnn(cell=scoRNNCell(self.hidden_dim, D = D), inputs=one_hot_inputs, dtype=tf.float32)
                if model_used == 'NCGRU':
                    states, _ = dynamic_rnn(cell=NCGRUCell(self.hidden_dim, two_orthogonal, D = D), inputs=one_hot_inputs, dtype=tf.float32)
                if model_used == 'goru':
                    states, _ = dynamic_rnn(cell=GORUCell(self.hidden_dim, capacity,
                        fft), inputs=one_hot_inputs, dtype=tf.float32)

            Wo = tf.get_variable('Wo', shape=[self.hidden_dim, self.target_dim],
                                 initializer=tf.random_normal_initializer(
                                     stddev=1.0 / (self.hidden_dim + self.target_dim) ** 2))
            bo = tf.get_variable('bo', shape=[1, self.target_dim],
                                 initializer=tf.zeros_initializer())

            bs, t = inputs.get_shape().as_list()
            logits = tf.matmul(tf.reshape(states, [t * bs, self.hidden_dim]), Wo) + bo
            logits = tf.reshape(logits, [bs, t, self.target_dim])
        return logits

class ParenthesisTask(object):
    def __init__(self, max_count=5, implied_activation_fn="softmax"):
        """
        Saturating count number of non-closed parenthesis.

        Args:
          max_count: level at which outstanding opening parenthesis stop being added
          implied_activation_fn: how is the output of the network interpreted:
            - softmax: treat it as logits, train via neg-log likelihood minimization
            - identity: treat it as probabilities, train via least-squares
        """
        self.max_count = max_count
        self.implied_activation_fn = implied_activation_fn
        self.parens = "()[]{}"
        self.n_paren_types = len(self.parens) // 2
        self.noises = "a"

        self.id_to_token = self.parens + self.noises
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

        self.n_tokens = len(self.id_to_token)
        self.n_outputs = self.n_paren_types * (self.max_count + 1)

    def sample_batch(self, t, bs):
        inputs = (np.random.rand(bs, t) * len(self.id_to_token)).astype(np.int32)
        counts = np.zeros((bs, self.n_paren_types), dtype=np.int32)
        targets = np.zeros((bs, t, self.n_paren_types), dtype=np.int32)
        opening_parens = (np.arange(0, self.n_paren_types) * 2)[None, :]
        closing_parens = opening_parens + 1
        for i in range(t):
            opened = np.equal(inputs[:, i, None], opening_parens)
            counts = np.minimum(self.max_count, counts + opened)
            closed = np.equal(inputs[:, i, None], closing_parens)
            counts = np.maximum(0, counts - closed)
            targets[:, i, :] = counts
        return inputs, targets

    def loss(self, logits, targets):
        bs, t, _ = logits.get_shape().as_list()
        logits = tf.reshape(logits, (bs, t, self.n_paren_types, self.max_count + 1))

        if self.implied_activation_fn == "softmax":
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
        elif self.implied_activation_fn == "identity":
            loss = tf.square(logits - tf.one_hot(targets, self.max_count + 1))
        else:
            raise Exception()

        return tf.reduce_mean(loss)

    def print_batch(self, inputs, targets, predictions=None, max_to_print=1):
        if predictions is not None:
            predictions = np.reshape(
                predictions,
                predictions.shape[:2] + (self.n_paren_types, self.max_count + 1))
        for i in range(min(max_to_print, inputs.shape[1])):
            print("%3d:" % (i,), " ".join([self.id_to_token[t] for t in inputs[:, i]]))
            for paren_kind in range(self.n_paren_types):
                print("G%s:" % self.parens[2 * paren_kind:2 * paren_kind + 2],
                      " ".join([str(c) for c in targets[:, i, paren_kind]]))
                if predictions is not None:
                    pred = np.argmax(predictions[:, i, paren_kind], axis=1)
                    print("P%s:" % self.parens[2 * paren_kind:2 * paren_kind + 2],
                          " ".join([str(c) for c in pred]))


def main():

    task = ParenthesisTask(max_count=MAX_COUNT,
                           implied_activation_fn="identity")
    task.print_batch(*task.sample_batch(100, 2))

    tf.reset_default_graph()

    with tf.variable_scope("model"):
        model = rnn_cell_model(task.n_tokens, HIDDEN_DIM, task.n_outputs)

        inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, T), name="inputs")
        targets = tf.placeholder(tf.int32, shape=(BATCH_SIZE, T, task.n_paren_types),
                                 name="targets")

        logits = model.fprop(inputs)

        task_loss = task.loss(logits, targets)
        weight_loss = 0.0
        for v in tf.trainable_variables():
            if v.name.split('/')[-1].startswith('W'):
                weight_loss = weight_loss + WEIGHT_DECAY * tf.nn.l2_loss(v)

        loss = task_loss + weight_loss

    learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")


    optimizer_dict = {'adam' : tf.train.AdamOptimizer,
                      'adagrad' : tf.train.AdagradOptimizer,
                      'rmsprop' : tf.train.RMSPropOptimizer,
                      'sgd' : tf.train.GradientDescentOptimizer}
                      
    optimizer = optimizer_dict[in_out_optimizer](learning_rate=learning_rate)


    if model_used == 'LSTM':
        LSTMtrain = optimizer.minimize(loss)
    if model_used == 'GRU':
        GRUtrain = optimizer.minimize(loss)

    if model_used == 'scoRNN':
        opt2 = optimizer_dict[A_optimizer](learning_rate=A_lr)
        Wvar = [v for v in tf.trainable_variables() if 'W:0' in v.name][0]
        Avar = [v for v in tf.trainable_variables() if 'A:0' in v.name][0]
        othervarlist = [v for v in tf.trainable_variables() if v not in \
                     [Wvar, Avar]]
                     
        # Getting gradients
        grads = tf.gradients(loss, othervarlist + [Wvar])
        
        # Applying gradients to input-output weights
        with tf.control_dependencies(grads):
            applygrad1 = optimizer.apply_gradients(zip(grads[:len(othervarlist)], \
                        othervarlist))
                        
        # Updating variables
        newW = tf.placeholder(tf.float32, Wvar.get_shape())
        updateW = tf.assign(Wvar, newW)
        
        # Applying hidden-to-hidden gradients
        gradA = tf.placeholder(tf.float32, Avar.get_shape())
        applygradA = opt2.apply_gradients([(gradA, Avar)])
        
    if model_used == 'NCGRU':
        opt2 = optimizer_dict[A_optimizer](learning_rate=A_lr)
        Wvar = [v for v in tf.trainable_variables() if 'weight_WC:0' in v.name][0]
        Avar = [v for v in tf.trainable_variables() if 'weight_A:0' in v.name][0]
        if two_orthogonal:
            Wvar_one = [v for v in tf.trainable_variables() if 'WC_gate_kernel_one:0' in v.name][0]
            Avar_one = [v for v in tf.trainable_variables() if 'weight_A_w_one:0' in v.name][0]
            othervarlist = [v for v in tf.trainable_variables() if v not in [Wvar, Wvar_one, Avar, Avar_one]]
        else:
            othervarlist = [v for v in tf.trainable_variables() if v not in [Wvar, Avar]]
        
        # Getting gradients
        if two_orthogonal:
            grads = tf.gradients(loss, othervarlist + [Wvar] + [Wvar_one])
        else:
            grads = tf.gradients(loss, othervarlist + [Wvar])
        
        # Applying gradients to input-output weights
        with tf.control_dependencies(grads):
            applygrad1 = optimizer.apply_gradients(zip(grads[:len(othervarlist)], \
                        othervarlist))
                        
        # Updating variables
        newW = tf.placeholder(tf.float32, shape=Wvar.get_shape())
        updateW = tf.assign(Wvar, newW)
        
        if two_orthogonal:
            newW_one = tf.placeholder(tf.float32, shape=Wvar_one.get_shape())
            updateW_one = tf.assign(Wvar_one, newW_one)
        
        # Applying hidden-to-hidden gradients
        gradA = tf.placeholder(tf.float32, Avar.get_shape())
        applygradA = opt2.apply_gradients([(gradA, Avar)])
        
        if two_orthogonal:
            gradA_one = tf.placeholder(tf.float32, Avar_one.get_shape())
            applygradA_one = opt2.apply_gradients([(gradA_one, Avar_one)])

    if model_used == 'goru':
        GORUtrain = optimizer.minimize(loss)
        
    tf.get_variable_scope().reuse_variables()

    print(inputs.get_shape().as_list())

    print("Training the following variables:")
    for v in tf.trainable_variables():
        print("%s: %s (%s)" % (v.name, v.get_shape().as_list(),
                               v.initializer.inputs[1].op.name))
                               
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
        
        
    # Get initial A and W 
    if model_used == 'scoRNN' or model_used == 'NCGRU':
        A, W = sess.run([Avar, Wvar])
        AplusIinv = None
        if two_orthogonal:
            A_one, W_one = sess.run([Avar_one, Wvar_one])
            AplusIinv_one = None

    PRINT_PERIOD = 100

    task_loss_acc = 0
        
    for v in tf.trainable_variables():
        loss_to_file((v.name, v.get_shape(), np.prod(v.get_shape().as_list())), 'structure_{:}.txt'.format(savestring))

    for i in range(N_TRAIN_STEPS):
        start_time = time.time()
        v_inputs, v_targets = task.sample_batch(T, BATCH_SIZE)
            
        if model_used == 'NCGRU':
            if i % 49 == 0:
                exact_value = True
            else:
                exact_value = False
        feed_dict = {inputs: v_inputs,
                    targets: v_targets,
                    learning_rate: in_out_lr
                    }

        if model_used == 'LSTM':
            v_task_loss, v_logits, _  = sess.run([task_loss, logits,LSTMtrain], feed_dict=feed_dict)
                
        if model_used == 'GRU':
            v_task_loss, v_logits, _  = sess.run([task_loss, logits, GRUtrain], feed_dict=feed_dict)
                
        if model_used == 'scoRNN':
            v_task_loss, v_logits, _, hidden_grads =  sess.run([task_loss, logits, applygrad1, grads[-1]], \
                                feed_dict = feed_dict)
                                
            if AplusIinv is None:
                print('(A+I)^-1 is None, initiating np.linalg.lstsq')
                I1 = np.identity(hidden_grads[0].shape[0])
                AplusIinv = np.linalg.lstsq(I1+A, I1, rcond=None)[0]
                    
            DFA = Cayley_Transform_Deriv(hidden_grads, A, W, D, AplusIinv)
            sess.run(applygradA, feed_dict = {gradA: DFA})
            A = sess.run(Avar)
            W, AplusIinv = makeW(A, D)
            sess.run(updateW, feed_dict = {newW: W})
                
        if model_used == 'NCGRU':
            v_task_loss, v_logits, _, hidden_grads =  sess.run([task_loss, logits, applygrad1, grads[-1]], \
                            feed_dict = feed_dict)
                
            if AplusIinv is None:
                print('(A+I)^-1 is None, initiating np.linalg.lstsq')
                I1 = np.identity(hidden_grads[0].shape[0])
                AplusIinv = np.linalg.lstsq(I1+A, I1, rcond=None)[0]
                
            if two_orthogonal:
                if AplusIinv_one is None:
                    print('(A_one+I)^-1 is None, initiating np.linalg.lstsq')
                    I1_one = np.identity(hidden_grads[0].shape[0])
                    AplusIinv_one = np.linalg.lstsq(I1_one+A_one, I1_one, rcond=None)[0]
                    
            DFA = Cayley_Transform_Deriv(hidden_grads, A, W, D, AplusIinv)
            sess.run(applygradA, feed_dict = {gradA: DFA})
            A = sess.run(Avar)
            W, AplusIinv = makeW(A, D, AplusIinv, DFA, A_lr, exact=exact_value)
            sess.run(updateW, feed_dict = {newW: W})
            
            if two_orthogonal:
                DFA_one = Cayley_Transform_Deriv(hidden_grads, A_one, W_one, D, AplusIinv_one)
                sess.run(applygradA_one, feed_dict = {gradA_one: DFA_one})
                A_one = sess.run(Avar_one)
                W_one, AplusIinv_one = makeW(A_one, D, AplusIinv, DFA_one, A_lr, exact=exact_value)
                sess.run(updateW_one, feed_dict = {newW_one: W_one})

        if model_used == 'goru':
            v_task_loss, v_logits, _  = sess.run([task_loss, logits, GORUtrain], feed_dict=feed_dict)
                
        task_loss_acc += v_task_loss
            
        end_time_save = time.time() - start_time

        if ((i + 1) % PRINT_PERIOD) == 0:
            print("Step %d loss %f" % (i, task_loss_acc / PRINT_PERIOD))
            loss_to_file(task_loss_acc / PRINT_PERIOD, 'test_loss_{:}.txt'.format(savestring))
            task.print_batch(v_inputs, v_targets, v_logits)
            print("")
            task_loss_acc = 0
            


if __name__ == '__main__':
    main()
