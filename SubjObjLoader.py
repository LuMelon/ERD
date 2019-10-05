#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import re
import numpy as np
import pandas as pd
import torch as th

# subj_file = "/home/hadoop/rotten_imdb/subj.data"
# obj_file = "/home/hadoop/rotten_imdb/obj.data"
# tr, dev, te = load_data(subj_file, obj_file)

def load_data(subj_file, obj_file): # 5000 obj - 5000 subj | 8500 train - 500 dev - 1000 test
    def TextFile2Dataset(filepath, label):
        sents = open(filepath, encoding="latin")
        instances = [(label, line) for line in sents]
        return instances
    subj_set = TextFile2Dataset(subj_file, 0)
    obj_set = TextFile2Dataset(obj_file, 1)
    total_set = []
    total_set.extend(subj_set)
    total_set.extend(obj_set)
    total_set = random.sample(total_set, len(total_set))
    train_set = total_set[:8500]
    test_set = total_set[8500:9500]
    dev_set = total_set[9500:]
    return train_set, dev_set, test_set

def LabelSmooth(label, epsilon):
    cls_num = len(label[0])
    label = label + (epsilon)/(1.0*(cls_num-1))
    for i in range(len(label)):
        for j in range(len(label[0])):
            if label[i][j]>1:
                label[i][j] -= 2*epsilon
    return label

class SubjObjReader:
    def __init__(self, dataset, batchsize, tokenizer=None):
        length = len(dataset)
        self.length = (length//batchsize)*batchsize
        
        self.label = np.zeros([self.length, 2])
        [self.label.itemset((idx, int(item[0])), 1) for (idx,item) in enumerate(dataset[:self.length])]
        
#         self.label = LabelSmooth(self.label, 0.2)
        
        if tokenizer is not None:
            self.words = np.array([ 
                    tokenizer.encode(item[1], add_special_tokens=True)
                    for item in dataset[:self.length]
            ])
        else:
            self.words = np.array([self.text2words(item[0]) for item in dataset[:self.length]])
        self.words_num = np.array([len(text) for text in self.words])
        self.max_sent_len = max(self.words_num)
        ##################convert into data batchs#################
        self.words = self.words.reshape(-1, batchsize)
        self.label = self.label.reshape(-1, batchsize, 2)
        self.words_num = self.words_num.reshape(-1, batchsize)
    def text2words(self, text):
        rep_dic = {'1':'one ', 
                   '2':'two ', 
                   '3':'three ', 
                   '4':'four ', 
                   '5':'fine ', 
                   '6':'six ', 
                   '7':'seven ', 
                   '8':'eight ', 
                   '9':'nine ', 
                   '0':'zero '}
        for k, v in rep_dic.items():
            text = text.replace(k, v)
        words = re.split('(?:[^a-zA-Z]+)', text.lower().strip() )
        return words
    
    def reset_batchsize(self, new_batch_size):
        self.words = self.words.reshape(-1, new_batch_size)
        self.label = self.label.reshape(-1, new_batch_size, 2)
        self.words_num = self.words_num.reshape(-1, new_batch_size)

    def sample(self):
        batches, _, _ = self.label.shape
        batch_idx = random.randint(0, batches-1)
        return self.words[batch_idx], self.label[batch_idx], self.words_num[batch_idx]
        
    def iter(self):
        for x, y, l in zip(self.words, self.label, self.words_num):
            yield x, y, l 

