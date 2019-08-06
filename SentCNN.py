#!/usr/bin/env python
# coding: utf-8

# In[42]:


import tensorflow as tf
from tensorflow.losses import Reduction
class SentCNN:
    def __init__(self, input_dim, hidden_dim, max_sent_len, class_num):
        self.input_x = tf.placeholder(tf.float32, [None, max_sent_len, input_dim], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, class_num], name="input_y")
        num_filters = 256
        kernel_size = 5
        conv_input = tf.layers.conv1d(self.input_x, num_filters, kernel_size,strides=1, padding='valid',name='conv2', trainable=True)
        feature_map = tf.nn.relu(conv_input) # [batchsize, conv_feats, filters]
        self.pooled_feat = tf.reduce_max(feature_map, 1)
        classifier = tf.layers.Dense(3, activation= tf.nn.relu, trainable=True)
        self.pred_scores = tf.nn.softmax(classifier(self.pooled_feat), axis=1)
        self.predictions = tf.argmax(self.pred_scores, 1, name="predictions")
        self.loss = tf.losses.softmax_cross_entropy(
                        self.input_y,
                        self.pred_scores,
                        weights=1.0,
                        label_smoothing=0,
                        scope=None,
                        loss_collection=tf.GraphKeys.LOSSES,
                        reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
                    )
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    def pred():
        pass