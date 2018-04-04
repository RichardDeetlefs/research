#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 18:35:12 2018

@author: rich
"""

import tensorflow as tf
import numpy as np

#helper functions--------------------------------------------------------------
def random_batch(x, batch_size):
    loc = np.random.randint(len(x), size=batch_size)
    return x[loc,:]
  
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

#encoder-----------------------------------------------------------------------
def encoder_CNN(input_tensor, z, batch, reuse=False):#input_tensor is the 32 x 32 image
    #act_fun_d = tf.nn.relu#change?
    act_fun_d = lrelu#change?
    
    input_tensor_ = tf.reshape(input_tensor, shape=[batch, 32, 32, 1])
    
    cnn_layer1 = tf.layers.conv2d(input_tensor_, 64, [5,5], strides=[2,2],
                                  padding='same',
                                  activation=act_fun_d,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                  reuse=reuse,
                                  name='e_CNN1')#image is now half the size --> 16x16
    
    cnn_layer2 = tf.layers.conv2d(cnn_layer1, 32, [5,5], strides=[2,2],
                                  padding='same',
                                  activation=act_fun_d,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                  reuse=reuse,
                                  name='e_CNN2')#image is now half the size --> 8x8

    cnn_layer3 = tf.layers.conv2d(cnn_layer2, 16, [5,5], strides=[2,2],
                                  padding='same',
                                  activation=act_fun_d,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                  reuse=reuse,
                                  name='e_CNN3')#image is now half the size --> 4x4

    fc = tf.reshape(cnn_layer3,[batch, 4*4*16])
    
    mean_z = tf.layers.dense(fc, 
                             z,
                             #activation=act_fun_d,#change?
                             activation=None,#change?
                             reuse=reuse,
                             name='z_mean')
    sum_hist_mean = tf.summary.histogram('mean_of_z', mean_z)
    
    std_z = tf.layers.dense(fc, 
                            z,
                            #activation=act_fun_d,#change?
                            activation=None,#change?
                            reuse=reuse,
                            name='z_std')
    sum_hist_std = tf.summary.histogram('std_of_z', std_z)
    
    
    samples = tf.random_normal([batch,z],0,1,dtype=tf.float32)
    guessed_z = mean_z + (std_z * samples)
    
    sum_hist_z = tf.summary.histogram('estimated_z', guessed_z)
    summary = tf.summary.merge([sum_hist_mean,sum_hist_std, sum_hist_z])
    
    return guessed_z, mean_z, std_z, summary
    
        
def encoder(data, z_dim, batchsize, reuse=False):
    with tf.variable_scope('encoder'):
        out, mean_z_, std_z_, sum_ = encoder_CNN(data, z_dim, batchsize, reuse=reuse)
    return out, mean_z_, std_z_, sum_

#decoder-----------------------------------------------------------------------
def decoder_CNN(input_tensor, out_dim, batch, reuse=False):#input tensor is  size of Z -  1x100

    #act_fun_g = tf.nn.relu#change?
    act_fun_g = lrelu#added#change?

    fully = tf.layers.dense(input_tensor, 
                               4*4*64,
                               activation=act_fun_g,
                               reuse=reuse,
                               name='g_hid1')
    

    fully_ = tf.reshape(fully, shape=[batch,4,4,64])

    cnn_T_layer1 = tf.layers.conv2d_transpose(fully_, 32, [5,5], strides=[2,2],
                                              padding='same',
                                              activation=act_fun_g,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                              reuse=reuse,
                                              name='d_CNN_T1')#image is now double the size --> 8x8

    cnn_T_layer2 = tf.layers.conv2d_transpose(cnn_T_layer1, 16, [5,5], strides=[2,2],
                                              padding='same',
                                              activation=act_fun_g,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                              reuse=reuse,
                                              name='d_CNN_T2')#image is now double the size --> 16x16

    cnn_T_layer3 = tf.layers.conv2d_transpose(cnn_T_layer2, 1, [5,5], strides=[2,2],
                                              padding='same',
                                              activation=None,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                              reuse=reuse,
                                             name='d_CNN_T3')#image is now doublr the size --> 32x32
    
    output = tf.reshape(cnn_T_layer3,[-1, out_dim])
    
    #output = tf.nn.tanh(output)#change?- for images pixel between (-1,1)
    output = tf.nn.sigmoid(output)#change?- for images pixel between (0,1) #added
    return output

def decoder(Z, x_dim, batchsize, reuse = False):
    sum_z = tf.summary.histogram('latent_space_input2decoder', Z)
    with tf.variable_scope('decoder'):
            out = decoder_CNN(Z, x_dim, batchsize, reuse=reuse)
            
    image_shaped_input = tf.reshape(out, [-1, 32, 32, 1])
    sum_image = tf.summary.image('images', image_shaped_input, 3)# shows images 

    summary = tf.summary.merge([sum_z, sum_image])
    return out, summary

#loss & training---------------------------------------------------------------
def train(real_img_batch, z_dim, batch_size,x_dim, LR):
    with tf.variable_scope('Loss_and_train'):
        with tf.variable_scope('VAE'):
            Encode, mean_Z, std_Z , encode_sum = encoder(real_img_batch, z_dim, batch_size, reuse=False)
            Decode, decode_sum = decoder(Encode, x_dim, batch_size, reuse = False)
            
        with tf.variable_scope('both_losses'):#loss funtion from http://kvfrans.com/variational-autoencoders-explained/ 
            encoder_loss = 0.5 * tf.reduce_sum(tf.square(mean_Z) + tf.square(std_Z) - tf.log(tf.square(std_Z)) - 1,1)
            decoder_loss = -tf.reduce_sum(real_img_batch * tf.log(1e-8 + Decode) + (1-real_img_batch) * tf.log(1e-8 + 1 - Decode),1)
            loss = tf.reduce_mean(encoder_loss+decoder_loss)

        sum_e_loss = tf.summary.scalar('L_encoder', tf.reduce_mean(encoder_loss))
        sum_d_loss = tf.summary.scalar('L_decoder', tf.reduce_mean(decoder_loss))
        sum_total_loss = tf.summary.scalar('L_total', loss)
        summary_ = tf.summary.merge([encode_sum, decode_sum, sum_e_loss, sum_d_loss, sum_total_loss])   
         
        train_both = tf.train.AdamOptimizer(LR).minimize(loss)
        train_encode = tf.train.AdamOptimizer(LR).minimize(encoder_loss)
        train_decode = tf.train.AdamOptimizer(LR).minimize(decoder_loss)

        return train_both, train_encode, train_decode, Encode, Decode, summary_
