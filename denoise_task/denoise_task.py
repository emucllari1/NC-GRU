###########################################
# '' The input sequence of length T contains 10 randomly located
# data points and the other T - 10 points are considered noise data.
# These 10 points are selected from a dictionary with i.e. n+2 elements,
# where the first n elements are data points and the last two are 
# 'noise' and 'marker'. 
# The model should output as soon as it receives the 'marker' and 
# the model should filter out the noise part and output the random 10 data points.''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import sys
import argparse
import os
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from goru import GORUCell

def loss_to_file(file1, path):
    f = open(path, "a+")
    f.write('{:}\n'.format(file1))
    f.close()

parser = argparse.ArgumentParser(description='Denoise Problem')
parser.add_argument('--model', default='lstm', type=str, choices=['lstm', 'gru', 'NCGRU', 'scoRNN', 'goru'], 
                        help='The name of the RNN model: goru, lstm')
parser.add_argument('--T', type=int, default=200, 
                        help='Delay step of denoise task')
parser.add_argument('--iters', type=int, default=10000, help='training iteration')
parser.add_argument('--batch_size', type=int, default=128, help='The number of samples in each batch.')
parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size of the RNN model')
parser.add_argument('--capacity', type=int, default=4, help='Capacity of uniary matrix in tunable case')
parser.add_argument('--fft', default=False, help='Whether to use fft version. False means using tunable version')
parser.add_argument('--in_out_optimizer', type=str, default='adam', choices=['adam', 'adagrad', 'rmsprop', 'sgd'], 
                        help=' ')
parser.add_argument('--in_out_lr', type=float, default=1e-3, 
                        help=' ')
parser.add_argument('--A_optimizer', type=str, default='adam', choices=['adam', 'adagrad', 'rmsprop', 'sgd'],
                        help=' ')
parser.add_argument('--A_lr', type=float, default=1e-3, 
                        help=' ')
parser.add_argument('--n_neg_ones', type=int, default=50, 
                        help='number of negative eigenvalues to put on diagonal')
parser.add_argument('--two_orthogonal', action='store_true', 
                        help='if this argument is called then the NC-GRU(U_c, U_r) is implemented')
args = parser.parse_args()
if args.model == 'NCGRU':
    from NCGRU import *
elif args.model == 'scoRNN':
    from scoRNN import *
model = args.model
T = args.T
iters = args.iters
batch_size = args.batch_size
hidden_size = args.hidden_size
capacity = args.capacity
fft = args.fft
# Input/Output parameters
in_out_optimizer = args.in_out_optimizer #adam
in_out_lr = args.in_out_lr #1e-3
# Hidden to hidden parameters
A_optimizer = args.A_optimizer #rmsprop
A_lr = args.A_lr #1e-4
n_neg_ones = args.n_neg_ones
two_orthogonal = args.two_orthogonal
n_neg_ones = int(n_neg_ones)   
fft = False

# Name of save string/scaling matrix
if model == 'lstm':
    savestring = 'denoise_task_{:s}_{:d}_{:s}_{:.1e}_seq_length_{:d}'.format(model, hidden_size, \
                 in_out_optimizer, in_out_lr, batch_size)
if model == 'gru':
    savestring = 'denoise_task_{:s}_{:d}_{:s}_{:.1e}_T={:d}'.format(model, hidden_size, \
                 in_out_optimizer, in_out_lr, batch_size)
if model == 'scoRNN' or model == 'NCGRU':
    savestring = 'denoise_task_{:s}_{:d}_{:d}_{:s}_{:.1e}'.format(model, \
                 int(hidden_size), int(n_neg_ones), in_out_optimizer, in_out_lr)
    D = np.diag(np.concatenate([np.ones(hidden_size - n_neg_ones), \
        -np.ones(n_neg_ones)]))
if model == 'goru':
    savestring = 'denoise_task_{:s}_{:d}_{:s}_{:.1e}_seq_length_{:d}'.format(model, hidden_size, \
                 in_out_optimizer, in_out_lr, batch_size)
print('\n' + savestring + '\n')


def denoise_data(T, n_data, n_sequence):
    seq = np.random.randint(1, high=10, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, T + n_sequence - 1))

    for i in range(n_data):
        ind = np.random.choice(T + n_sequence - 1, n_sequence)
        ind.sort()
        zeros1[i][ind] = seq[i]

    zeros2 = np.zeros((n_data, T + n_sequence))
    marker = 10 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros2, seq), axis=1).astype('int64')

    return x, y


def main(_):

    # --- Set data params ----------------
    n_input = 11
    n_output = 10
    n_sequence = 10
    n_train = iters * batch_size
    n_test = batch_size

    n_input = 10
    n_steps = T + 20
    n_classes = 10

    # --- Create graph and compute gradients ----------------------
    x = tf.placeholder("int32", [None, n_steps])
    y = tf.placeholder("int64", [None, n_steps])

    input_data = tf.one_hot(x, n_input, dtype=tf.float32)

    # --- Input to hidden layer ----------------------
    if model == "lstm":
        cell = tf.nn.rnn_cell.BasicLSTMCell(
            hidden_size, state_is_tuple=True, forget_bias=1)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
        
    if model == 'scoRNN':
        cell = scoRNNCell(hidden_size, D = D)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    if model == 'NCGRU':
        cell = NCGRUCell(hidden_size, two_orthogonal, D = D)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    if model == "gru":
        cell = tf.contrib.rnn.GRUCell(hidden_size)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    if model == "goru":
        cell = GORUCell(hidden_size, capacity,
                        fft)
        hidden_out, _ = tf.nn.dynamic_rnn(
            cell, input_data, dtype=tf.float32)

    # --- Hidden Layer to Output ----------------------
    V_init_val = np.sqrt(6.) / np.sqrt(n_output + n_input)

    V_weights = tf.get_variable("V_weights", shape=[hidden_size, n_classes],
                                dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[n_classes],
                             dtype=tf.float32, initializer=tf.constant_initializer(0.01))

    hidden_out_list = tf.unstack(hidden_out, axis=1)
    temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
    output_data = tf.nn.bias_add(tf.transpose(temp_out, [1, 0, 2]), V_bias)

    # --- evaluate process ----------------------
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=output_data, labels=y))
    correct_pred = tf.equal(tf.argmax(output_data, 2), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # --- Initialization ----------------------

    # Optimizers/Gradients
    optimizer_dict = {'adam' : tf.train.AdamOptimizer,
                      'adagrad' : tf.train.AdagradOptimizer,
                      'rmsprop' : tf.train.RMSPropOptimizer,
                      'sgd' : tf.train.GradientDescentOptimizer}
        
    opt1 = optimizer_dict[in_out_optimizer](learning_rate=in_out_lr)   
    if model == 'lstm':
        LSTMtrain = opt1.minimize(cost)
    if model == 'gru':
        GRUtrain = opt1.minimize(cost)
    if model == 'scoRNN':
        opt2 = optimizer_dict[A_optimizer](learning_rate=A_lr)
        Wvar = [v for v in tf.trainable_variables() if 'W:0' in v.name][0]
        Avar = [v for v in tf.trainable_variables() if 'A:0' in v.name][0]
        othervarlist = [v for v in tf.trainable_variables() if v not in \
                    [Wvar, Avar]]
    
        # Getting gradients
        grads = tf.gradients(cost, othervarlist + [Wvar])

        # Applying gradients to input-output weights
        with tf.control_dependencies(grads):
            applygrad1 = opt1.apply_gradients(zip(grads[:len(othervarlist)], \
                        othervarlist))  
    
        # Updating variables
        newW = tf.placeholder(tf.float32, Wvar.get_shape())
        updateW = tf.assign(Wvar, newW)
    
        # Applying hidden-to-hidden gradients
        gradA = tf.placeholder(tf.float32, Avar.get_shape())
        applygradA = opt2.apply_gradients([(gradA, Avar)])
    if model == 'NCGRU':
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
            grads = tf.gradients(cost, othervarlist + [Wvar] + [Wvar_one])
        else:
            grads = tf.gradients(cost, othervarlist + [Wvar])

        # Applying gradients to input-output weights
        with tf.control_dependencies(grads):
            applygrad1 = opt1.apply_gradients(zip(grads[:len(othervarlist)], \
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
            
    if model == 'goru':
        GORUtrain = opt1.minimize(cost)
    init = tf.global_variables_initializer()

    # --- baseline ----------------------
    baseline = np.log(9) * 10 / (T + 20)
    print("Baseline is " + str(baseline))

    # --- Training Loop ----------------------

    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.log_device_placement = False
    config.allow_soft_placement = False
    with tf.Session(config=config) as sess:

        # --- Create data --------------------

        train_x, train_y = denoise_data(T, n_train, n_sequence)
        test_x, test_y = denoise_data(T, n_test, n_sequence)

        sess.run(init)
        
        # Get initial A and W    
        if model == 'scoRNN' or model == 'NCGRU':    
            A, W = sess.run([Avar, Wvar])
            AplusIinv = None
            if two_orthogonal:
                A_one, W_one = sess.run([Avar_one, Wvar_one])
                AplusIinv_one = None

        step = 0
        for v in tf.trainable_variables():
            loss_to_file((v.name, v.get_shape(), np.prod(v.get_shape().as_list())), 'structure_{:}.txt'.format(savestring))

        while step < iters:
            batch_x = train_x[
                step * batch_size: (step + 1) * batch_size]
            batch_y = train_y[
                step * batch_size: (step + 1) * batch_size]
                
            if model == 'NCGRU':
                if step % 49 == 0:
                    exact_value = True
                else:
                    exact_value = False


            if model == 'lstm':
                sess.run(LSTMtrain, feed_dict={x: batch_x, y: batch_y})
            
            if model == 'gru':
                sess.run(GRUtrain, feed_dict={x: batch_x, y: batch_y})

            if model == 'scoRNN':
                _, hidden_grads = sess.run([applygrad1, grads[-1]], \
                                feed_dict = {x: batch_x, y: batch_y})
            
                if AplusIinv is None:
                    print('(A+I)^-1 is None, initiating np.linalg.lstsq')
                    I1 = np.identity(hidden_grads[0].shape[0])
                    AplusIinv = np.linalg.lstsq(I1+A, I1, rcond=None)[0]
            
                DFA = Cayley_Transform_Deriv(hidden_grads, A, W, D, AplusIinv)
                sess.run(applygradA, feed_dict = {gradA: DFA})
                A = sess.run(Avar)
                W, AplusIinv = makeW(A, D)
                sess.run(updateW, feed_dict = {newW: W})
            if model == 'NCGRU':
                _, hidden_grads = sess.run([applygrad1, grads[-1]], \
                                feed_dict = {x: batch_x, y: batch_y})
            
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
                    
            if model == 'goru':
                sess.run(GORUtrain, feed_dict={x: batch_x, y: batch_y})


            acc, loss = sess.run([accuracy, cost], feed_dict={
                                 x: batch_x, y: batch_y})
                                 
            loss_to_file(loss, 'train_loss_{:}.txt'.format(savestring))
            loss_to_file(acc, 'train_acc_{:}.txt'.format(savestring))

            print(" Iters " + str(step) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                  
            test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
            test_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
            print("Test result: Loss= " +
              "{:.6f}".format(test_loss) + ", Accuracy= " + "{:.5f}".format(test_acc))
              
            loss_to_file(test_loss, 'test_loss_{:}.txt'.format(savestring))
            loss_to_file(test_acc, 'test_acc_{:}.txt'.format(savestring))

            step += 1

        print("Optimization Finished!")


if __name__ == "__main__":
    tf.app.run()
