#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 08:27:58 2018

@author: rich
"""

import tensorflow as tf
import numpy as np
#--------------------------USEFUL FUNCTIONS--------------------------#
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def OneD_array(ten):#helps get gradents for consensus optimisation into a 1D array
    array = []
    for i in range(len(ten)):
        array.append(tf.reshape(ten[i], [-1]))
    return tf.concat(array, 0)

def instance_noise(x, batch_size, x_dim, var, mean=0):
    x = x + tf.random_normal([batch_size, x_dim], mean=mean, stddev=var)
    return x

def random_batch(x, batch_size):
    loc = np.random.randint(len(x), size=batch_size)
    return x[loc,:]

def dis_test_sum(D):
    with tf.name_scope('discriminator_test'):
        D = tf.reduce_mean(D)
        sum_D = tf.summary.scalar('dis-NO-sigmoid', D)
        
        dis = tf.nn.sigmoid(D)
        sum_sig = tf.summary.scalar('sigmoid-of-dis', dis)
        sum_sig_log = tf.summary.scalar('LOG-sigmoid-of-dis', tf.log(dis))
        
    summary = tf.summary.merge([sum_D, sum_sig, sum_sig_log])
    return summary
        

def variable_summary(v, layer_name):#summaries for graph
    with tf.name_scope(layer_name):
        with tf.name_scope('variable_summaries'):
            norm = tf.reduce_mean(tf.square(v))
            sum_norm = tf.summary.scalar('norm', norm)
            mean = tf.reduce_mean(v)
            sum_mean = tf.summary.scalar('mean', mean)
            with tf.name_scope('std_dev'):
                std_dev = tf.sqrt(tf.reduce_mean(tf.square(v - mean)))
            sum_std = tf.summary.scalar('std_dev', std_dev)
            sum_hist = tf.summary.histogram('histogram', v)
        summary = tf.summary.merge([sum_mean, sum_norm,sum_std,sum_hist])
    return summary

#--------------------------GENERATOR--------------------------#   
def gen_CNN(input_tensor, out_dim, reuse=False):#input tensor is  size of Z -  1x100
    drop_rate = 0.0# this value: x100 = the percentage of nodes that will be droped
    #change? activation function
    act_fun_g = tf.nn.relu
    #act_fun_g = tf.nn.leaky_relu#added - not helpful
    
    fully = tf.layers.dense(input_tensor, 
                               4*4*128,#2048
                               activation=act_fun_g,
                               reuse=reuse,
                               name='g_hid1')
    fully = tf.layers.dropout(fully,rate=drop_rate)#added
    fully_ = tf.reshape(fully, shape=[-1,4,4,128])
    
                                                    #added. was 64
    cnn_T_layer1 = tf.layers.conv2d_transpose(fully_, 64, [5,5], strides=[2,2],
                                              padding='same',#change?
                                              activation=act_fun_g,#change?
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),#change?
                                              reuse=reuse,
                                              name='g_CNN_T1')#image is now double the size --> 8x8
    cnn_T_layer1 = tf.layers.dropout(cnn_T_layer1,rate=drop_rate)#added
    
                                                         #added. was 32
    cnn_T_layer2 = tf.layers.conv2d_transpose(cnn_T_layer1, 32, [5,5], strides=[2,2],
                                              padding='same',#change?
                                              activation=act_fun_g,#change?
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),#change?
                                              reuse=reuse,
                                              name='g_CNN_T2')#image is now double the size --> 16x16
    cnn_T_layer2 = tf.layers.dropout(cnn_T_layer2,rate=drop_rate)#added
        
                                                        #must always be 1
    cnn_T_layer3 = tf.layers.conv2d_transpose(cnn_T_layer2, 1, [5,5], strides=[2,2],
                                              padding='same',#change?
                                              activation=None,#change? added - made none!!!
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),#change?
                                              reuse=reuse,
                                              name='g_CNN_T3')#image is now double the size --> 32x32
    

    output = tf.reshape(cnn_T_layer3,[-1, out_dim])#NEW
    return tf.nn.tanh(output)

def generator(Z, x_dim, reuse = False):
    sum_z = tf.summary.histogram('latent_space', Z)
    with tf.name_scope('The-Generator'):
        with tf.variable_scope('generator'):#allows me to call the variables in here
                out = gen_CNN(Z, x_dim, reuse=reuse)
    sum_output = tf.summary.histogram('gen_output', out)

    summary = tf.summary.merge([sum_z, sum_output])
    return out, summary

def generator_loss_f(D_fake):#only takes the discriminators results of fake data
    dis_fake = tf.nn.sigmoid(D_fake)
    #loss function 1
    g_loss = tf.reduce_mean(-tf.log(tf.divide(dis_fake + 1e-8, 1.0 - dis_fake + 1e-8)))#KL loss. change?
    
    #loss function 2
    #g_loss = tf.reduce_mean(-tf.log(dis_fake))#added - from GAN hacks

    #for consensus optimisation
    g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    derivate_g_wrt_w = tf.gradients(g_loss, g_var)
    g_vec_field = OneD_array(derivate_g_wrt_w)
    
    #summary for tensorboard
    sum_g_loss_grad = variable_summary(g_vec_field, 'der_of_gen_loss')
    sum_g_loss = tf.summary.scalar('generator_loss', g_loss)
    summary = tf.summary.merge([sum_g_loss_grad,sum_g_loss])
    
    return g_loss, g_vec_field, summary

def generator_train(g_loss, LR, g_vec_field, d_vec_field, lam):
    with tf.variable_scope('', reuse=True):
        global_step = tf.get_variable('global_step',dtype=tf.int32)#using it once here so as to update global step each time this is called
    
    optimizer = tf.train.AdamOptimizer(learning_rate=LR,beta1=0.5)#change?
    
    #consensus optimisation
    g_d_vec_field = tf.concat([tf.expand_dims(g_vec_field,1),tf.expand_dims(d_vec_field,1)], axis=0)
    L_regularizer = 0.5*tf.matmul(tf.transpose(g_d_vec_field), g_d_vec_field)
    loss = g_loss + L_regularizer*lam
    
    #we only update the weights of the generator
    g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    g_train = optimizer.minimize(loss, var_list=g_var, global_step=global_step)
    return g_train

#--------------------------DISCRIMINATOR--------------------------#
def dis_CNN(input_tensor, reuse=False):#input_tensor is the 32 x 32 image
    drop_rate = 0.0# this value: x100 = the percentage of nodes that will be droped
    #change? activation function
    #act_fun_d = tf.nn.relu
    act_fun_d = lrelu#added
    
    input_tensor_ = tf.reshape(input_tensor, shape=[-1, 32, 32, 1])#NEW
                                            #added. was 32
    cnn_layer1 = tf.layers.conv2d(input_tensor_, 32, [5,5], strides=[2,2],#change?
                                  padding='same',#change?
                                  activation=act_fun_d,#change?
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),#change?
                                  #kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.2,dtype=tf.float32),#change? added
                                  reuse=reuse,
                                  name='d_CNN1')#image is now half the size --> 16x16
    cnn_layer1 = tf.layers.dropout(cnn_layer1,rate=drop_rate)#added

                                            #added. was 64
    cnn_layer2 = tf.layers.conv2d(cnn_layer1, 64, [5,5], strides=[2,2],#change?
                                  padding='same',#change?
                                  activation=act_fun_d,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),#change?
                                  reuse=reuse,
                                  name='d_CNN2')#image is now half the size --> 8x8
    cnn_layer2 = tf.layers.dropout(cnn_layer2,rate=drop_rate)#added

                                            #added. was 128
    cnn_layer3 = tf.layers.conv2d(cnn_layer2, 128, [5,5], strides=[2,2],#change?
                                  padding='same',#change?
                                  activation=act_fun_d,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),#change?
                                  reuse=reuse,
                                  name='d_CNN3')#image is now half the size --> 4x4
    cnn_layer3 = tf.layers.dropout(cnn_layer3,rate=drop_rate)#added
    cnn_layer3 = tf.reshape(cnn_layer3,[-1, 4*4*128])#this is the manifold
    
    output = tf.layers.dense(cnn_layer3, 
                             1,
                             activation=None,
                             reuse=reuse,
                             name='d_out')

    output = lrelu(output, leak=1.0)#leak = 1.0 --> linear. leak = 0.0 --> relu. 
    return output, cnn_layer3
    
        
def discriminator(data, test=False, reuse=False):
    with tf.name_scope('The-Discriminator'):
        with tf.variable_scope('discriminator'):
            out, second_last_layer = dis_CNN(data, reuse=reuse)
    
    if test == False:
        image_shaped_input = tf.reshape(data, [-1, 32, 32, 1])
        if reuse == True:
            sum_image = tf.summary.image('Fake_images', image_shaped_input, 3)# shows images - these images have noise
            sum_d_output = variable_summary(out, 'Dis-output-Fake')
            sum_d_2nd_last = variable_summary(second_last_layer, 'Dis-2nd-last-Fake')
        else:
            sum_image = tf.summary.image('Real_images', image_shaped_input, 3)# shows images - these images have noise
            sum_d_output = variable_summary(out, 'Dis-output-Real')
            sum_d_2nd_last = variable_summary(second_last_layer, 'Dis-2nd-last-Real')
        
        summary = tf.summary.merge([sum_d_output,sum_d_2nd_last,sum_image])
        return out,summary
    
    else:
        sum_d = dis_test_sum(out)
        sum_d_output = variable_summary(out, 'Dis-output-Test')
        sum_d_2nd_last = variable_summary(second_last_layer, 'Dis-2nd-last-Test')
        summary = tf.summary.merge([sum_d_output,sum_d_2nd_last,sum_d])
        return second_last_layer, summary
        

def discriminator_loss_f(d_fake, d_real):#discriminator fake and real results
#    #loss function 1
#    fake0 = tf.zeros_like(d_fake)
#    Real1 = tf.ones_like(d_real)
#    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = fake0, logits = d_fake))
#    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Real1, logits = d_real))
#    d_loss = tf.divide(tf.add(loss_fake, loss_real),2.0)
    
    #loss function 2 - added
    dis_fake = tf.nn.sigmoid(d_fake)#should be close to 0
    dis_real = tf.nn.sigmoid(d_real)#should be close to 1 
    d_loss = -0.5*tf.reduce_mean((tf.log(dis_real) + tf.log(1.0 - dis_fake)))
   
    #for consensus optimisation
    d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    derivate_d_wrt_w = tf.gradients(d_loss, d_var)
    d_vec_field = OneD_array(derivate_d_wrt_w) 

    sum_d_loss_grad = variable_summary(d_vec_field, 'der_of_dis_loss')
    sum_d_loss_fake = variable_summary(tf.sigmoid(d_fake), 'sigmoid_d_loss_fake')
    sum_d_loss_real = variable_summary(tf.sigmoid(d_real), 'sigmoid_d_loss_real')
    sum_d_loss = tf.summary.scalar('discriminator_loss', d_loss)
    summary = tf.summary.merge([sum_d_loss_grad,sum_d_loss_fake,sum_d_loss_real,sum_d_loss])
    
    return d_loss, d_vec_field, summary

def discriminator_train(d_loss, LR, g_vec_field, d_vec_field, lam):
    
    optimizer = tf.train.AdamOptimizer(learning_rate=LR,beta1=0.5)#change?
    #optimizer = tf.train.GradientDescentOptimizer(LR)#change? added
    
    #consensus optimisation
    g_d_vec_field = tf.concat([tf.expand_dims(g_vec_field,1),tf.expand_dims(d_vec_field,1)], axis=0)
    L_regularizer = 0.5*tf.matmul(tf.transpose(g_d_vec_field), g_d_vec_field)
    loss = d_loss + L_regularizer*lam
    
    #we only update the weights of the discriminator
    d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    d_train = optimizer.minimize(loss, var_list=d_var)
    return d_train