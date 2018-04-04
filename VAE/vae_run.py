#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 18:36:31 2018

@author: rich
"""

import tensorflow as tf
import vae_fun as VAE
import numpy as np
from matplotlib import pyplot as plt
#--------------------------Importing real data--------------------------#
from tensorflow.examples.tutorials.mnist import input_data
min_x = 0
max_x = 1
range_x = 1

def norm_data(xx):
    return (2*xx - min_x - max_x)/range_x

def unnorm_data(xx_norm):
    return (range_x*xx_norm + min_x + max_x)/2

def pad_img(img):#padding images to become a 32x32
    padded_images = []
    for i in range(len(img)):
        img_28 = np.reshape(img[i], [28,28])
        pad = np.pad(img_28, ((2,2),(2,2)), 'constant', constant_values=(-1,-1))
        padded_images.append(np.reshape(pad, [1,1024]))
    return padded_images
        

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

X = mnist.train.images
Y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

#building a training dataset which only contains a single digit of choice
train_digit = 0#'Healthy' digit
train_images = []
new_X = []
new_Y = []
N = len(Y)
for i in range(N):
    label = np.where(Y[i] == 1)
    label = label[0][0]
    if label == train_digit:
        train_images.append(X[i])
train_images = np.array(train_images, np.float32)
#train_images = norm_data(train_images)
pad_train_images = pad_img(train_images)
pad_train_images = np.reshape(pad_train_images, [len(train_images), 1024])
train_images = pad_train_images#images are now a 32x32

batch_size = 128
z_dim = 20
x_dim = 1024
LR = 0.0001

with tf.Graph().as_default():
    real_images = tf.placeholder(dtype=tf.float32, shape=[batch_size,x_dim], name='real-images') 
    
    
    training, training_E, training_D , ecode, dcode, sum_ = VAE.train(real_images, z_dim, batch_size, x_dim, LR)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    write_summary = tf.summary.FileWriter('./tensorboard', sess.graph)
    merged = tf.summary.merge_all()
    
    want2train = True
    step = 1
    while want2train == True:
        image_batch = VAE.random_batch(train_images, batch_size)
        
        #E, D = sess.run([Encode, Decode], feed_dict={real_images:image_batch})
        _e,_d,summary = sess.run([training_E, training_D, merged], feed_dict={real_images:image_batch})
        
        if step % 1 ==0:
            write_summary.add_summary(summary, step)#add to tensorboard
            write_summary.flush()
        if step % 100 == 0:
            print('step:', step)
        step += 1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        