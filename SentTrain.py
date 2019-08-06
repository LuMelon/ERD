#!/usr/bin/env python
# coding: utf-8

# In[8]:


import dataloader


# In[4]:
from logger import MyLogger

import time
from SentCNN import *

logger = MyLogger("SentTrain")

def get_curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

sent_mm = SentCNN(300, 256, 41, 3)


# In[5]:


import tensorflow as tf
sent_global_step = tf.Variable(0, name="global_step", trainable=False)
sent_train_op = tf.train.AdagradOptimizer(0.01).minimize(sent_mm.loss, sent_global_step)        


# In[1]:


train_batch = 20
test_batch = 20

train_iter = int(len(dataloader.trainlabel)/train_batch) + 1
test_iter = int(len(dataloader.testlabel)/test_batch) + 1

sum_loss = 0.0
sum_acc = 0.0
ret_acc = 0.0


# In[ ]:


sess = tf.Session()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)

with sess.as_default():
    sess.run(tf.global_variables_initializer())

for t_epoch in range(100): 
    for t_iter in range(train_iter):
        data_X, data_Y = dataloader.GetTrainingBatch(t_iter, train_batch, 300)
        feed_dic = {sent_mm.input_x: data_X, sent_mm.input_y: data_Y}
        _, step, loss, acc = sess.run([sent_train_op, sent_global_step, sent_mm.loss, sent_mm.accuracy], feed_dic)
        sum_loss += loss
        sum_acc += acc
        if t_iter % 100 == 99:
            sum_loss = sum_loss / 100
            sum_acc = sum_acc / 100
            ret_acc = sum_acc
            print(get_curtime() + " Step: " + str(step) + " Training loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
            logger.info(get_curtime() + " Step: " + str(step) + " Training loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
#             if sum_acc > 0.9:
#                 break
            sum_acc = 0.0
            sum_loss = 0.0
    # for validation
    sum_acc = 0.0
    sum_loss = 0.0
    for t_iter in range(10):
        data_X, data_Y = dataloader.GetTestData(t_iter, test_batch, 300)
        feed_dic = {sent_mm.input_x: data_X, sent_mm.input_y: data_Y}
        loss, acc = sess.run([sent_mm.loss, sent_mm.accuracy], feed_dic)
        sum_loss += loss
        sum_acc += acc    
    sum_loss = sum_loss / 100
    sum_acc = sum_acc / 100
    ret_acc = sum_acc
    print(get_curtime() + " Step: " + str(step) + " validation loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
    logger.info(get_curtime() + " Step: " + str(step) + " validation loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
    sum_acc = 0.0
    sum_loss = 0.0
    
    saver.save(sess, "df_saved/sent_model")


# In[ ]:




