#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:40:15 2017

Implemented using: https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
@author: rajegupt
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length =15
state_size = 4
num_classes =2
echo_step = 3
batch_size =5

num_batches = total_series_length//batch_size//truncated_backprop_length 
#  //: divide with integral result (discard remainder)


                        ### Generating the training Data (random binary vector)

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5,0.5]))
    # shift the elements by echo_steps and Elements that roll beyond the last 
    #position are re-introduced at the first. 
    y = np.roll(x, echo_step) 
    y[0:echo_step]=0
     
    
    x = x.reshape((batch_size,-1)) ## converting 1*50000 to 5*10000 shape
    y = y.reshape((batch_size,-1))
    
    return (x,y)

                        ## Building the computational graph
                        
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

# State save the output from the previous run
init_state = tf.placeholder(tf.float32, [batch_size, state_size]) 

#  the shape of W? - will get clear in forward pass section
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype = tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype= tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype= tf.float32)


# Unpack columns  (just unpacking the individual batch input from the original list of batches)
input_series = tf.unstack(batchX_placeholder, axis =1)
label_series = tf.unstack(batchY_placeholder, axis =1)


                        ## Forward Pass
                        
current_state = init_state
states_series =[]

for current_input in input_series:
    current_input = tf.reshape(current_input,[batch_size,1])
    input_and_state_concatenated = tf.concat([current_input, current_state],1) # increasing number of columns
    
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated,W)+b) # broadcasted addition
    states_series.append(next_state)
    current_state = next_state
    
                        ## Calculating Loss
                        
logits_series = [tf.matmul(state, W2)+ b2 for state in states_series]
prediction_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series, label_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

                        ## Visualizing the training

def plot(loss_list, prediction_series, batchX, batchY):
    plt.subplot(2,3,1)
    plt.cla() # clear axis
    plt.plot(loss_list)
    
    for batch_series_idx in range(5):
        one_hot_output_series = np.array(prediction_series)[:, batch_series_idx,:]
        # ?? is it > or < 0.5?
        single_output_series = np.array([(1 if out[0] > 0.5 else 0) for out in one_hot_output_series])
        
        plt.subplot(2,3,batch_series_idx+2)
        plt.cla()
        plt.axis([0,truncated_backprop_length,0,2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx,:], width=1, color ='blue')
        plt.bar(left_offset, batchY[batch_series_idx,:]*0.5, width=1, color='red')
        plt.bar(left_offset, single_output_series*0.3, width=1, color='green')
        
    plt.draw()
    plt.pause(0.0001)
    

                            ## Running the training session


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion() # Interactive mode may also be turned on
    plt.figure()
    plt.show()
    loss_list =[]
    
    for epoch_idx in range(num_epochs):
        #Unusual way of generating new data everytime but it is just a an example
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size))
        
        print ("New data, epoch", epoch_idx)
        
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
            
            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]
            
            _total_loss, _train_step, _current_state, _prediction_series=sess.run(
                    [total_loss,train_step,current_state, prediction_series],
                    feed_dict={
                            batchX_placeholder:batchX,
                            batchY_placeholder: batchY,
                            init_state:_current_state
                            })
            loss_list.append(_total_loss)
            
            if batch_idx % 100 ==0:
                print ("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _prediction_series, batchX, batchY)
                
plt.ioff()
plt.show()
    









                                                                                                       

