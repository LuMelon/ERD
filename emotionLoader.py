#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import random


# In[11]:


from pytorch_transformers import *


# In[2]:


def Convey_data_dict(data_dict):
    keys = list(data_dict.keys())
    all_sentences = []
    all_label = np.zeros([1, len(keys)])
    for label, key in enumerate(keys):
        sample_label = np.zeros(len(keys))
        sample_label[label] = 1.0
        sentences = data_dict[key]["sentences"]
        labels = np.zeros([len(sentences), len(keys)]) + sample_label
        all_sentences.extend(sentences)
        all_label = np.append(all_label, labels, axis=0)
    return all_sentences, all_label[1:], keys 


# In[3]:


def load_data_from_file(filename):
    with open(filename) as fr:
        data_dict = json.load(fr)
    sentences, labels, label2emotion = Convey_data_dict(data_dict)
    sampling = random.sample(list(zip(sentences, labels)), len(labels))
    sentences = [item[0] for item in sampling]
    labels = [item[1] for item in sampling]
    return sentences, np.array(labels), label2emotion


# __数据位置：__
# ``` python
# file1 = "/home/hadoop/EmoNet-PyTorch/twitter10.json"
# file2 = "/home/hadoop/EmoNet-PyTorch/twitter30.json"
# file3 = "/home/hadoop/EmoNet-PyTorch/bopang30.json"
# file4 = "/home/hadoop/EmoNet-PyTorch/blogs10.json"
# file5 = "/home/hadoop/EmoNet-PyTorch/blogs30.json"
# ```

# In[26]:


class EmotionReader:
    def __init__(self, sentences, labels, batchsize, tokenizer=None):
        length, label_num = labels.shape
#         print("length:", length, "| num:", label_num)
        self.length = (length//batchsize)*batchsize
        if tokenizer is not None:
            self.words = np.array([ 
                    tokenizer.encode(item, add_special_tokens=True)
                    for item in sentences[:self.length]
            ])
        else:
            self.words = np.array([self.text2words(item[0]) for item in dataset[:self.length]])
        self.label = labels[:self.length]
        self.words_num = np.array([len(text) for text in self.words])
        self.max_sent_len = max(self.words_num)
        ##################convert into data batchs#################
        self.words = self.words.reshape(-1, batchsize)
        self.label = self.label.reshape(-1, batchsize, label_num)
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
        _, _, label_num = self.label.shape
        self.words = self.words.reshape(-1, new_batch_size)
        self.label = self.label.reshape(-1, new_batch_size, label_num)
        self.words_num = self.words_num.reshape(-1, new_batch_size)
        
    def iter(self):
        for x, y, l in zip(self.words, self.label, self.words_num):
            yield x, y, l 

