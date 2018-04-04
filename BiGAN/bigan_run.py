#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:17:28 2018

@author: rich
"""

import tensorflow as tf
import bigan_functions as GAN

from helper_functions import mnist_collect, set_up, manifold_images
#--------------------------Importing real data--------------------------#
train_images, test_images = mnist_collect(0, 1)
#--------------------------Parameters--------------------------#
z_dim = 100#change? 
x_dim = 1024#update: dimension of real data, eg; for MNIST --> x_dim = 784
g_LR = 0.0001#change?
d_LR = 0.0001#higher LR in Dis. seems to work well. change?
e_LR = 0.0001
noise_var = 1.5#change?
batch_size = 128#change?
test_batch_size = 64#change
g_lam = 5.0 #constant for consensus optimisation. change? Making larger helped a lot!
d_lam = 5.0 #constant for consensus optimisation. change? Making larger helped a lot!
e_lam = 5.0 #constant for consensus optimisation. change? Not sure how it impacts the encoder
stddev_z = 1.0

save_loc, save_check, img_dir, continue_training, pre_train, pre_train_N = set_up()
#--------------------------Training--------------------------#
def training():
    with tf.Graph().as_default(): #helps remove error or reuse each time I rerun the script by reloading my gan functions everytime
        sess = tf.Session()#starting the session - once off
        global_step = tf.get_variable('global_step', shape=[],dtype=tf.int32)
        
        #Decaying hyperparametrs
        g_lr = tf.train.exponential_decay(g_LR, global_step, 1000, 0.975, staircase=False)#decay_rate and decay_step --> change?
        d_lr = tf.train.exponential_decay(d_LR, global_step, 1000, 0.975, staircase=False)#decay_rate and decay_step --> change?
        e_lr = tf.train.exponential_decay(e_LR, global_step, 1000, 0.975, staircase=False)#decay_rate and decay_step --> change?
        
        g_lam_lr = tf.train.exponential_decay(g_lam, global_step, 1000, 0.95, staircase=True) + 0.01 #decay_rate and decay_step --> change?
        d_lam_lr = tf.train.exponential_decay(d_lam, global_step, 1000, 0.95, staircase=True) + 0.01 #decay_rate and decay_step --> change?
        e_lam_lr = tf.train.exponential_decay(e_lam, global_step, 1000, 0.95, staircase=True) + 0.01 #decay_rate and decay_step --> change?
        
        noise = tf.train.polynomial_decay(noise_var, global_step,5000,0.0,power=1.0)#linear decay
        
        x = tf.placeholder(tf.float32, [None, x_dim], 'data')#real data
        x_test = tf.placeholder(tf.float32, [test_batch_size,x_dim], 'test_data')#single test image - could perhaps to batch testing - change?
        
        z = tf.random_normal([batch_size, z_dim], mean=0, stddev=stddev_z)#change?

        gen_x, sum_gen = GAN.generator(z, x_dim, reuse = False)#reuse = False. Generating the fake data

        gen_z, sum_e = GAN.encoder(x, z_dim, True, reuse = False)#encoder takes real images
        gen_z_test, sum_e_test = GAN.encoder(x_test, z_dim, False, reuse = True)#encoder takes real images
        
        #adding the instance noise to real and fake data
        gen_xx = gen_x + tf.random_normal([batch_size, x_dim], mean=0.0, stddev=noise)#can remove this --> change?#added - changing the factor of noise
        xx = x + tf.random_normal([batch_size, x_dim], mean=0.0, stddev=noise)#can remove this --> change?
        
        #adding noise to the real and generated latent variable
        gen_zz = gen_z + tf.random_normal([batch_size, z_dim], mean=0.0, stddev=noise)
        zz = z + tf.random_normal([batch_size, z_dim], mean=0.0, stddev=noise)
        
        # real images come with generated z. And fake images comes with real latent z
        d_real, image_summary_real = GAN.discriminator(xx, gen_zz, True, False)#added - switched the two around! This was ontop
        d_fake, image_summary_fake = GAN.discriminator(gen_xx, zz, True, True)#reusing the same variables as dis_real
        second_last_layer, d_test_sum = GAN.discriminator(x_test, gen_z_test, False, True)#True as the is for testing only
        
        e_loss, e_vec_field, e_summary = GAN.encoder_loss_f(d_real)
        g_loss, g_vec_field, g_summary = GAN.generator_loss_f(d_fake)
        d_loss, d_vec_field, d_summary = GAN.discriminator_loss_f(d_fake, d_real)
        
        e_train = GAN.encoder_train(e_loss, e_lr, e_vec_field, d_vec_field, e_lam_lr)
        g_train = GAN.generator_train(g_loss, g_lr, g_vec_field, d_vec_field, g_lam_lr)
        d_train = GAN.discriminator_train(d_loss, d_lr, g_vec_field, d_vec_field, d_lam_lr)
        
        initial_var = tf.global_variables_initializer()
        sess.run(initial_var)
        saver = tf.train.Saver(tf.trainable_variables())
        write_summary =  tf.summary.FileWriter(save_loc + '/tensorboard/train', sess.graph)
        write_summary2 = tf.summary.FileWriter(save_loc + '/tensorboard/test', sess.graph)
        #----------------------------------------------------
        merged_summary = tf.summary.merge_all()
        #Everything above was seeting up the graph
        
        train = True
        step = 1
        while train == True:
            #print('step: ', step)
            if step == 1:
                if continue_training == True:
                    print()
                    print('Loading saved weights')
                    print()
                    saver.restore(sess, save_check)
                    step = sess.run(global_step) + 1
                    
                if pre_train == True and continue_training == False:
                    print()
                    print('Pre-training the discriminator')
                    for i in range(pre_train_N):
                        real_data = GAN.random_batch(train_images, batch_size)
                        _,loss_d, lr_d = sess.run([d_train, d_loss, d_lr], feed_dict={x: real_data})
                        if i % 50 == 0:
                            print('Dis. Loss: %.5f, with LR: %.10f, at iteration %d of %d' % (loss_d, lr_d, i, pre_train_N))
            
            #starting real training
            real_data = GAN.random_batch(train_images, batch_size)
            test_img = GAN.random_batch(train_images, test_batch_size)#feeding in healthy images
            _d, _g, _e, gs = sess.run([d_train, g_train, e_train, global_step], feed_dict={x: real_data, x_test: test_img})
            
            if step % 10 == 0:
                summary = sess.run(merged_summary, feed_dict={x: real_data, x_test: test_img})
                write_summary.add_summary(summary, step)#add to tensorboard
                write_summary.flush()
                
                #adding test data to tensorboard
                test_img = GAN.random_batch(test_images, test_batch_size)
                summary = sess.run(d_test_sum, feed_dict={x: real_data, x_test: test_img})
                write_summary2.add_summary(summary, step)#
                write_summary2.flush()
                
            if step % 50 ==0:
                loss_g,lr_g,loss_e,lr_e,loss_d,lr_d, lr_lamG,lr_lamE,lr_lamD = sess.run([g_loss, g_lr,
                                                                                         e_loss, e_lr,
                                                                                         d_loss, d_lr, 
                                                                                         g_lam_lr,e_lam_lr,d_lam_lr], feed_dict={x: real_data, x_test: test_img})
                print()
                print('Step # %d & Global step # %d' % (step, gs))
                print('....G...:')
                print('         loss: %.5f, LR: %.10f' % (loss_g, lr_g))
                print('....E...:')
                print('         loss: %.5f, LR: %.10f' % (loss_e, lr_e))
                print('....D...:')
                print('         loss: %.10f, LR: %.10f' % (loss_d, lr_d))
                print('Con. opt. lambda; G=', lr_lamG, 'E=', lr_lamE, 'D=', lr_lamD)    

            if step % 1000 == 0:#manifold image
                test_img_healthy = GAN.random_batch(train_images, test_batch_size)
                healthy_man = sess.run(second_last_layer, feed_dict={x: real_data, x_test: test_img_healthy})
                
                test_img_unhealthy = GAN.random_batch(test_images, test_batch_size)
                unhealthy_man = sess.run(second_last_layer, feed_dict={x: real_data, x_test: test_img_unhealthy})
                _ = manifold_images(healthy_man, unhealthy_man, img_dir, step)
                
            if step % 1000 == 0:#save weights 
                print()
                print('--------weights saved--------')
                saver.save(sess, save_check)#I dont have to over ride each time. Can save many - check online @ tensorflow
                saver.restore(sess, save_check)
                    
            step += 1
            
training()   