#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:01:37 2017

Implemented using https://gist.github.com/karpathy/d4dee566867f8291f086
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

@author: rajegupt
"""
import numpy as np

                                # Data Input 
#" some plain simple text file"
data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print (" the data has %d characters, %d unique" % (data_size, vocab_size))

# dict mapping char to index and vice versa
char_to_ix = { ch: i for i, ch  in enumerate(chars)}
ix_to_char = { i:ch for i, ch in enumerate(chars)}


                                # hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1



                                # model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01  # hidden to output
bh = np.zeros((hidden_size,1))  # hidden bias
by = np.zeros ((vocab_size,1))  # output bias

              
def lossFun(inputs, targets, hprev):
    """
   inputs,targets are both list of integers.
   hprev is Hx1 array of initial hidden state
   returns the loss, gradients on model parameters, and last hidden state
    """
  
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss= 0
  
  # forward pass
  
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]]=1
        hs[t] = np.tanh(np.dot(Wxh, xs[t])+ np.dot(Whh, hs[t-1])+ bh) # hidden state
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log prob for next chars
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t])) # ??? probabilites for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
        
        # backward pass: compute gradients going backwards
        
        
        