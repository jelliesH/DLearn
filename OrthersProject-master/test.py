#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:28:26 2017

@author: 390771
"""
from __future__ import print_function
import numpy as np
import random
from sklearn.datasets import load_iris

data=load_iris()
rng = random.sample(range(len(data['data'])),150)
train_x  = data['data'][rng[:130],:]
train_y  = data['target'][rng[:130]]
test_x = data['data'][rng[130:],:]    
test_y = data['target'][rng[130:]]
te_y = np.zeros([len(test_y),3])
ta_y = np.zeros([len(train_y),3])
te_y[test_y == 0,0] = 1
te_y[test_y == 1,1] = 1
te_y[test_y == 2,2] = 1
ta_y[train_y == 0,0] = 1
ta_y[train_y == 1,1] = 1
ta_y[train_y == 2,2] = 1
    
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 25 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_input = 4 # MNIST data input (img shape: 28*28)
n_classes =  3# MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None,n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
#    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result = sess.run([optimizer, cost,pred], feed_dict={x: train_x,
                                                          y: ta_y})
    Out = result
    print(result)
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x:test_x , y:te_y }))