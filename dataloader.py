#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import re
import pandas as pd
import os
import numpy as np
# In[ ]:

def transIrregularWord(word):
    if not word:
        return ''
    pattern1 = "[^A-Za-z]*$" #punctuation at the end of sentence
    pattern2 = "^[^A-Za-z@#]*" #punctuation at the start of sentence
    word = re.sub(pattern2, "", re.sub(pattern1, "", word))
    pattern3 = '(.*)http(.?)://(.*)' # url
    pattern4 = '^[0-9]+.?[0-9]+$' # number
    if not word:
        return ''
    elif word.__contains__('@'):
        return 'person'
    elif word.__contains__('#'):
        return 'topic'
    elif re.match(r'(.*)http?://(.*)', word, re.M|re.I|re.S):    
        return 'links'
    elif re.match(pattern4, word, re.M|re.I):
        return 'number'
    else:
        return  word.lower()
    
def sentence2words(line):
    words = re.split('([,\n ]+)', line.strip() )
    words = list( filter(lambda s: len(s)>0, [transIrregularWord(word) for word in words]) )
    return words

def CSVFile2Dataset(filepath):
    print(filepath)
    df = pd.read_csv(filepath,encoding='latin-1')
    instances = [(line[-1], line[0]) for line in df.values]
    del df
    texts = [sentence2words(instance[0]) for instance in instances]
    labels = [instance[1] for instance in instances]
    return texts, labels

# dirpath = '/home/hadoop/trainingandtestdata'
# trainfile = 'training.1600000.processed.noemoticon.csv'
# testfile = 'testdata.manual.2009.06.14.csv'
# trainset, trainlabel = CSVFile2Dataset(os.path.join(dirpath, trainfile))
# testset, testlabel = CSVFile2Dataset(os.path.join(dirpath, testfile))
trainset = np.load("trainset.npy").tolist()
testset = np.load("testset.npy").tolist()
trainlabel = np.load("trainlabel.npy").tolist()
testlabel = np.load("testlabel.npy").tolist()
max_sent_len = max(max([len(text) for text in trainset]), max([len(text) for text in testset]))

# import gensim
# word2vec = gensim.models.KeyedVectors.load_word2vec_format('/home/hadoop/word2vec.model')
import chars2vec 
c2vec = chars2vec.load_model('eng_300')


def GetTrainingBatch(batchId, batchsize, embedding_dim):
    data_x = np.zeros([batchsize, max_sent_len, embedding_dim], dtype=np.float32)
    data_y = np.zeros([batchsize, 3], dtype=np.int32)
    startIdx = batchId*batchsize
    # miss_vec = 0
    # hit_vec = 0
    trainVol = len(trainlabel)
    if startIdx >= len(trainset):
        startIdx = startIdx%len(trainset)
    for i in range(batchsize):
        mts = int(np.random.uniform()*trainVol)
        data_y[i][int(trainlabel[mts]/2)] = 1
        # for j in range(len(trainset[mts])):
            # try:
            #     data_x[i][j] = word2vec[trainset[mts][j]]
            # except KeyError:
            #     print("word:", trainset[mts][j])
            #     miss_vec += 1
            # except IndexError:
            #     print("i, j, k:", FLAGS.batch_size, '|',t_data_len[mts] ,'|', len(t_words))
            #     print("word:", trainset[mts][j], "(", i, j, k, ")")
            #     raise
            # else:
            #     hit_vec += 1
        data_x[i][:len(trainset[mts])] = c2vec.vectorize_words(trainset[mts])
    # print("hit_vec | miss_vec:", hit_vec, '|', miss_vec)
    return data_x, data_y

def GetTestData(batchId, batchsize, embedding_dim):
    data_x = np.zeros([batchsize, max_sent_len, embedding_dim], dtype=np.float32)
    data_y = np.zeros([batchsize, 3], dtype=np.int32)
    startIdx = batchId*batchsize
    miss_vec = 0
    hit_vec = 0
    testVol = len(testlabel)
    if startIdx >= len(testset):
        startIdx = startIdx%len(testset)
    for i in range(batchsize):
        mts = int(np.random.uniform()*testVol)
        data_y[i][int(testlabel[mts]/2)] = 1
    #     for j in range(len(testset[mts])):
    #         try:
    #             data_x[i][j] = word2vec[testset[mts][j]]
    #         except KeyError:
    #             print("word:", testset[mts][j])
    #             miss_vec += 1
    #         except IndexError:
    #             print("i, j, k:", batch_size, '|',t_data_len[mts] ,'|', len(t_words))
    #             print("word:", testset[mts][j], "(", i, j, k, ")")
    #             raise
    #         else:
    #             hit_vec += 1
        data_x[i][:len(testset[mts])] = c2vec.vectorize_words(testset[mts])
    # print("hit_vec | miss_vec:", hit_vec, '|', miss_vec)
    return data_x, data_y

