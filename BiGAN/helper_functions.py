# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:15:03 2018

@author: u13026888
"""


import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os
import shutil
#------------------------------------------------------------------------------
from tensorflow.examples.tutorials.mnist import input_data
min_x = 0.0
max_x = 1.0
range_x = 1.0

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
        
def mnist_collect(healthy, unhealthy):
    #healthy is a digit of choice between 0-9. And test can be a single digit OR
    # a bunch of digits in order from 0-9. Test = 10 will give you this
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
    
    x = mnist.train.images
    y = mnist.train.labels
    test_X = mnist.test.images
    test_Y = mnist.test.labels
    
    X = np.concatenate((x, test_X), axis=0)
    Y = np.concatenate((y, test_Y), axis=0)

    test_x = np.concatenate((x, test_X), axis=0)
    test_y = np.concatenate((y, test_Y), axis=0)    
    
    #building a training dataset which only contains a single digit of choice
    train_digit = healthy#'Healthy' digit
    train_images = []
    N = len(Y)
    for i in range(N):
        label = np.where(Y[i] == 1)
        label = label[0][0]
        if label == train_digit:
            train_images.append(X[i])
    train_images = np.array(train_images, np.float32)
    train_images = norm_data(train_images)
    pad_train_images = pad_img(train_images)
    pad_train_images = np.reshape(pad_train_images, [len(train_images), 1024])
    train_images = pad_train_images#images are now a 32x32
    
    if unhealthy == 10:
        test_images = []
        digit = 0
        i = 0
        for i in range(len(test_y)):
            print(i)
            test_digit = digit
            N = len(test_y)
            find = True
            while find == True:
                label = np.where(test_y[i] == 1)
                label = label[0][0]
                if label == test_digit:
                    test_images.append(test_x[i])
                    find = False
            if test_digit != 9:
                digit += 1
            else:
                digit = 0       
        test_images = np.array(test_images, np.float32)
        test_images = norm_data(test_images)
        pad_test_images = pad_img(test_images)
        pad_test_images = np.reshape(pad_test_images, [len(test_images), 1024])
        test_images = pad_test_images#images are now a 32x32
        
    else:
        test_digit = unhealthy#'Healthy' digit
        test_images = []
        N = len(test_y)
        for i in range(N):
            label = np.where(test_y[i] == 1)
            label = label[0][0]
            if label == test_digit:
                test_images.append(X[i])
        test_images = np.array(test_images, np.float32)
        test_images = norm_data(test_images)
        pad_test_images = pad_img(test_images)
        pad_test_images = np.reshape(pad_test_images, [len(test_images), 1024])
        test_images = pad_test_images#images are now a 32x32
        
        
    return train_images, test_images
        
#------------------------------------------------------------------------------

def set_up():
    stats_dir = './run_stats'
    if not os.path.exists(stats_dir):#making a directory
        os.makedirs(stats_dir)
        
    def answer(a):
        if a == 'yes':
            return True
        else:
            return False
    runs_done = len(os.listdir('./run_stats'))
    if runs_done == 0:
        code_run_number = str(runs_done + 1)
    else:
        code_run_number = str(runs_done)
        
    print('Answer as either; yes (True) or no (False)')
    continue_training = answer(input("Continue training? (yes or no) :"))
    if continue_training == True:
        print('Will restore old weights for code run number:', code_run_number)
        pre_train = False
        pre_train_N = 0
    else:
        pre_train = answer(input("Pre-train the discriminator? (yes or no) :"))
        if pre_train == True:
            pre_train_N = int(input("Pre-train for how many iterations? (interger) :"))
        else:
            pre_train_N = 0
            
        new_run = answer(input("Is this a new run? (yes or no) :"))
        if new_run == True:
            if runs_done == 0:
                code_run_number = str(int(code_run_number))
                print('First run! Making directory for code run number:', code_run_number)
            else:
                print('Updating code run number from:', code_run_number)
                code_run_number = str(int(code_run_number) + 1)
                print('                           to:', code_run_number)
                print('A whole new directory has been created for this run')
        else:
            save_loc = './run_stats/run' + code_run_number
            img_dir = save_loc + '/manifold_images'
            print('Overing old file under code run number:', code_run_number)
            files = os.listdir(save_loc)
            exists = 'tensorboard' in files
            if exists == True:#need to delete old file
                shutil.rmtree(save_loc + '/tensorboard')
                print('deteling old tensorboard file')
        
            exists = 'manifold_images' in files
            if exists == True:#need to delete old file
                shutil.rmtree(save_loc + '/manifold_images')
                print('deteling old manifold imaged')
    
    functoin_file = 'dcgan_functions.py'
    this_file = 'dcgan_run.py'
    helper_file = 'helper_functions.py'
    save_loc = './run_stats/run' + code_run_number
    img_dir = save_loc + '/manifold_images'
    if not os.path.exists(save_loc):#making a directory
        os.makedirs(save_loc) 
    if not os.path.exists(img_dir):#making a directory
        os.makedirs(img_dir)
        
    shutil.copy(functoin_file, save_loc)
    shutil.copy(this_file, save_loc)
    shutil.copy(helper_file, save_loc)
    
    save_check = save_loc + '/checkpoints/weights_'
    return save_loc, save_check, img_dir, continue_training, pre_train, pre_train_N

#------------------------------------------------------------------------------
def manifold_images(healthy_man, unhealthy_man, img_dir, step):
    test_batch_size = len(healthy_man)
    d = np.concatenate((healthy_man, unhealthy_man),0) 
            
    pca = PCA(n_components=2)
    pca.fit(d)#d is of shape [2*batch, image_len]
    print()
    print('saving manifold image. explained_variance_ratio_ of PCA is below:')
    print(pca.explained_variance_ratio_, 'sum =', sum(pca.explained_variance_ratio_))
    pca_d = pca.transform(d)
    pca_d0 = pca_d[0:test_batch_size,:]
    pca_d1 = pca_d[test_batch_size:2*test_batch_size,:]
    
    img_log = img_dir + '/manifold_at_step_' + str(step) + '.png'
    plt.figure(figsize=(8,7))
    plt.plot(pca_d0[:,0], pca_d0[:,1], 'r.', label='Healthy')
    plt.plot(pca_d1[:,0], pca_d1[:,1], 'b.', label='Unhealthy')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(img_log)
    plt.clf()
    return 

def two_d_plot(healthy_man, unhealthy_man, img_dir, step):
    test_batch_size = len(healthy_man)
    d = np.concatenate((healthy_man, unhealthy_man),0) 
    
    pca_d0 = d[0:test_batch_size,:]
    pca_d1 = d[test_batch_size:2*test_batch_size,:]
    
    img_log = img_dir + '/manifold_at_step_' + str(step) + '.png'
    plt.figure(figsize=(8,7))
    plt.plot(pca_d0[:,0], pca_d0[:,1], 'r.', label='Healthy')
    plt.plot(pca_d1[:,0], pca_d1[:,1], 'b.', label='Unhealthy')
    plt.xlabel('fc1')
    plt.ylabel('fc2')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(img_log)
    plt.clf()
    return 
    