#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import re
import numpy as np
import pandas as pd
import torch as th

# train_file = "/home/hadoop/trainingandtestdata/training.1600000.processed.noemoticon.csv"
# test_file = "/home/hadoop/trainingandtestdata/testdata.manual.2009.06.14.csv"
def preproc(item):
    def del_words(sentence):
        sentence = re.sub("http(.?)://(.*)", "",
                        re.sub("#[\w]*", "", 
                            re.sub("@[\w]*", "", 
                                    sentence
                            )
                        )
                    )
        return sentence
    return (del_words(item[0]), item[1])

def load_data(train_file, test_file): # charVocab is a char-to-idx dictionary:{char:idx}
    def CSVFile2Dataset(filepath):
        print(filepath)
        df = pd.read_csv(filepath,encoding='latin-1')
        instances = [preproc((line[-1], line[0])) for line in df.values]
        return instances
    train_set = CSVFile2Dataset(train_file)
    test_set = CSVFile2Dataset(test_file)
    train_set = random.sample(train_set, len(train_set))
    test_set = random.sample(test_set, len(test_set))
    return train_set, test_set

class tSentiReader:
    def __init__(self, dataset, batchsize, tokenizer=None):
        length = len(dataset)
        self.length = (length//batchsize)*batchsize
        
        self.label = np.zeros([self.length, 3])
        [self.label.itemset((idx, int(item[1]/2)), 1) for (idx,item) in enumerate(dataset[:self.length])]
        
        if tokenizer is not None:
            self.words = np.array([ 
                    tokenizer.encode(item[0], add_special_tokens=True)
                    for item in dataset[:self.length]
            ])
        else:
            self.words = np.array([self.text2words(item[0]) for item in dataset[:self.length]])
        self.words_num = np.array([len(text) for text in self.words])
        self.max_sent_len = max(self.words_num)
        ##################convert into data batchs#################
        self.words = self.words.reshape(-1, batchsize)
        self.label = self.label.reshape(-1, batchsize, 3)
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
        
    def iter(self):
        for x, y, l in zip(self.words, self.label, self.words_num):
            yield x, y, l 

class LMReader:
    def __init__(self, dataset, batchsize, tokenizer=None):
        length = len(dataset)
        self.length = (length//batchsize)*batchsize
        
        self.label = np.zeros([self.length, 3])
        [self.label.itemset((idx, int(item[1]/2)), 1) for (idx,item) in enumerate(dataset[:self.length])]
        
        if tokenizer is not None:
            self.words = np.array([ 
                    tokenizer.encode(item[0], add_special_tokens=True)
                    for item in dataset[:self.length]
            ])
        else:
            self.words = np.array([self.text2words(item[0]) for item in dataset[:self.length]])
        self.words_num = np.array([len(text) for text in self.words])
        self.max_sent_len = max(self.words_num)
        
        idxs = {}
        for idx, num in enumerate(self.words_num):
            if idxs.get(num) == None:
                idxs[num] = [idx]
            else:
                idxs[num].append(idx)

        batch_idxs = []
        for i in range(10, self.max_sent_len):
            if idxs.get(i) is not None:
                l_len = len(idxs[i])
                if l_len < batchsize:
                    continue
                else:
                    batch_idxs.extend(idxs[i][:(l_len - l_len%batchsize)])
        self.label = self.label[batch_idxs].reshape([-1, batchsize, 3])
        self.words = self.words[batch_idxs].reshape([-1, batchsize])
        self.words_num = self.words_num[batch_idxs].reshape([-1, batchsize])
        self.max_sent_len = self.words_num.max()

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
        
    def iter(self):
        for x, y, l in zip(self.words, self.label, self.words_num):
            yield x, y, l 
