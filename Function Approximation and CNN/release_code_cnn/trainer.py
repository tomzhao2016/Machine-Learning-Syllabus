import tensorflow as tf
import numpy as np
import datetime
import os
import sys
import argparse


slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):

     
        self.net = net
        self.data = data
       
        self.max_iter = 5000
        self.summary_iter = 200
        


      
        self.learning_rate = 0.1
       
        self.saver = tf.train.Saver()
      
        self.summary_op = tf.summary.merge_all()

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        '''
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        '''
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.net.class_loss)
        

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    


    def optimize(self):

        self.train_losses = []
        self.test_losses = []

        '''
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test loss through out the process
        '''
        #x = tf.placeholder(tf.float32, [None, self.data.image_size, self.data.image_size, 3])
        #y_ = tf.placeholder(tf.float32, [None, self.net.num_class])
        for i in range(self.max_iter):
            feature_batch, label_batch = self.data.get_train_batch()
            feed_dict = {self.net.images: feature_batch, self.net.labels: label_batch}
            self.sess.run(self.train, feed_dict = feed_dict)
            if np.mod(i, self.summary_iter) == 0:
                val_feature_batch, val_label_batch = self.data.get_validation_batch()
                val_feed_dict = {self.net.images: val_feature_batch, self.net.labels: val_label_batch}
                self.test_losses.append(self.sess.run(self.net.accurracy,feed_dict=val_feed_dict))
                self.train_losses.append(self.sess.run(self.net.accurracy,feed_dict=feed_dict))
                print "loss for Train:", self.train_losses[-1]
                print "loss for Validation:", self.test_losses[-1]



            











