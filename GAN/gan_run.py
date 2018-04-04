#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:27:24 2018

@author: rich
"""

import tensorflow as tf
import gan_functions as GAN
import numpy as np

#--------------------------Importing real data--------------------------#
mu = 0
sigma = 0.1 
data = np.random.normal(mu, sigma, [int(1e6),1])
#check to update x_dim according to data size

#--------------------------Parameters--------------------------#
z_dim = 64#change?
x_dim = 1#dimension of real data, eg; for MNIST --> x_dim = 784
g_LR = 5e-6#change?
d_LR = 5e-4#higher LR in Dis. seems to work well. change?
noise_var = 0.8#change?
batch_size = 1024#change?
lam = 0.01 #constant for consensus optimisation. change?
save_loc = './checkpoints/weights_001'

#--------------------------Training--------------------------#
def training():
    with tf.Graph().as_default(): #helps remove error or reuse each time I rerun the script by reloading my gan functions everytime
        sess = tf.Session()#starting the session - once off
        #global_step = tf.Variable(0, name='global_step', trainable=False)
        global_step = tf.get_variable('global_step', shape=[],dtype=tf.int32)
        
        #Decaying hyperparametrs
        g_lr = tf.train.exponential_decay(g_LR, global_step, 400, 0.9, staircase=False)#decay_rate and decay_step --> change?
        d_lr = tf.train.exponential_decay(d_LR, global_step, 200, 0.9, staircase=False)#decay_rate and decay_step --> change?
        noise = tf.train.polynomial_decay(noise_var, global_step,1000,0.0,power=1.0)
        
        x = tf.placeholder(tf.float32, [None, x_dim], 'data')#real data
        z = tf.random_normal([batch_size, z_dim], mean=0, stddev=0.1)#change?
        gen_x = GAN.generator(z, x_dim, reuse = False)#reuse = False. Generating the fake data
        
        #adding the instance noise to real and fake data
        gen_x = gen_x + tf.random_normal([batch_size, x_dim], mean=0, stddev=noise)#can remove this --> change?
        x = x + tf.random_normal([batch_size, x_dim], mean=0, stddev=noise)#can remove this --> change?
        
        d_real = GAN.discriminator(x, False)
        d_fake = GAN.discriminator(gen_x, True)#reusing the same variables as dis_real
        
        g_loss, g_vec_field, g_summary = GAN.generator_loss_f(d_fake)
        d_loss, d_vec_field, d_summary = GAN.discriminator_loss_f(d_fake, d_real)
    
        g_train = GAN.generator_train(g_loss, g_lr, g_vec_field, d_vec_field, lam)
        d_train = GAN.discriminator_train(d_loss, d_lr, g_vec_field, d_vec_field, lam)
        
        initial_var = tf.global_variables_initializer()
        sess.run(initial_var)
        saver = tf.train.Saver(tf.trainable_variables())
        write_summary = tf.summary.FileWriter('./tensorboard')
        merged = tf.summary.merge_all()
        #Everything above was seeting up the graph
        
        #Actual training
        train = True
        continue_training = False#This is used to import old weights to continue training
        step = 1
        while train == True:
            if step == 1:#either pretrain D or import old weights at start
                pre_train_N = 100
                if continue_training == False:
                    print('Performing pre training on D')
                    print()
                    for i in range(pre_train_N):
                        real_data = GAN.random_batch(data, batch_size)
                        _,loss_d, lr_d = sess.run([d_train, d_loss, d_lr], feed_dict={x: real_data})
                        if i % 50 == 0:
                            print('Dis. Loss: %.5f, with LR: %.10f, at iteration %d of %d' % (loss_d, lr_d, i, pre_train_N))
                else:
                    print('Loading saved weights')
                    print()
                    saver.restore(sess, save_loc)
                    
            #training both D & G
            real_data = GAN.random_batch(data, batch_size)
            _,loss_g,lr_g, _,loss_d,lr_d, gs, summary = sess.run([g_train, g_loss, g_lr, 
                                                                  d_train, d_loss, d_lr,
                                                                  global_step,merged], feed_dict={x: real_data})
            if step % 10 == 0:
                write_summary.add_summary(summary, step)#add to tensorboard
            
            if step % 50 ==0:
                print()
                print('Step # %d & Global step # %d' % (step, gs))
                print('....G...:')
                print('         loss: %.5f, LR: %.10f' % (loss_g, lr_g))
                print('....D...:')
                print('         loss: %.5f, LR: %.10f' % (loss_d, lr_d))
                
            if step % 1000 == 0:#save weights 
                saver.save(sess, save_loc)
                saver.restore(sess, save_loc)
                
            step = step + 1
            

training()   
                
                    

            
    
    
    
    



































