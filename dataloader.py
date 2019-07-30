#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import pandas as pd
import os

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
    print('read out csv file')
    instances = [(line[-1], line[0]) for line in df.values]
    print('df to instances')
    del df
    texts = [sentence2words(instance[0]) for instance in instances]
    print('to texts')
    labels = [instance[1] for instance in instances]
    print('to labels')
    return texts, labels

dirpath = '/Users/lumenglong/Downloads/trainingandtestdata'
trainfile = 'training.1600000.processed.noemoticon.csv'
testfile = 'testdata.manual.2009.06.14'
trainset, trainlabel = CSVFile2Dataset(os.path.join(dirpath, trainfile))
testset, testlabel = CSVFile2Dataset(os.path.join(dirpath, testfile))
max_sent_len = max([(len(sent) for sent in texts) for texts in [trainset, testset]])
print(max_sent_len)


# In[ ]:



def GetTrainingBatch(start, batchsize):
    pass

def GetTestData():
    pass

