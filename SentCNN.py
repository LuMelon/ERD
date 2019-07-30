#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import sonnet as snt

class SentCNN:
    def __init_(self, input_dim, hidden_dim, max_sent_len, class_num):
        self.input_x = tf.placeholder(tf.float32, [None, max_sent_len, input_dim], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, class_num], name="input_y")
        num_filters = 256
        kernel_size = 5
        conv_input = tf.layers.conv1d(inputs, num_filters, kernel_size,strides=1, padding='valid',name='conv2')
        feature_map = tf.relu(conv_input) # [batchsize, conv_feats, filters]
        self.pooled_feat = tf.reduce_max(feature_map, 1)
        classifier = snt.Linear() 
        self.pred_scores = 

