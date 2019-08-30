import re
import pandas as pd
import os
import numpy as np
import keras
import pickle

class SentiDataLoader:
    def __init__(self, dirpath, trainfile, testfile, charVocab):
        self.dirpath = dirpath
        self.trainfile = trainfile
        self.testfile = testfile
        self.charVocab = charVocab

    def load_data(self): # charVocab is a char-to-idx dictionary:{char:idx}
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
        #     elif word.__contains__('@'):
        #         return 'person'
            elif word.__contains__('#'):
                return 'topic'
            elif re.match(r'(.*)http?://(.*)', word, re.M|re.I|re.S):    
                return 'links'
#             elif re.match(pattern4, word, re.M|re.I):
#                 print("word:", word)
#                 return 'number'
            else:
                return  word.lower()

        def sentence2words(line):
            words = re.split('([,\n ]+)', line.strip() )
            words = list( filter(lambda s: len(s)>0, [transIrregularWord(word) for word in words]) )
            return words

        def word2chars(word, charVocab):
            rst = [charVocab.get(char) for char in word]
            for i in range(len(rst)):
                if rst[i] is None:
                    print("Unknown char:", word[i])
                    print("word:", word)
                    rst[i] = 0
            return rst

        def CSVFile2Dataset(filepath):
            print(filepath)
            df = pd.read_csv(filepath,encoding='latin-1')
            instances = [(line[-1], line[0]) for line in df.values]
            del df
            texts = [sentence2words(instance[0]) for instance in instances]
            labels = [instance[1] for instance in instances]
            return texts, labels
        
        # charVocab = {c:idx for (idx, c) in enumerate(self.s)}
        texts, self.train_label = CSVFile2Dataset(os.path.join(self.dirpath, self.trainfile))
        self.train_data = [list(word2chars(word, self.charVocab) for word in sentence) for sentence in texts]
        texts, self.test_label = CSVFile2Dataset(os.path.join(self.dirpath, self.testfile))
        self.test_data = [list(word2chars(word, self.charVocab) for word in sentence) for sentence in texts]
        del texts
        self.max_sent_len = max([max(len(sent) for sent in texts) for texts in [self.train_data, self.test_data]])
        self.max_char_num = max(
                                max([max(len(word) for word in sent) for sent in self.test_data]), 
                                max([max(len(word) for word in sent) for sent in self.train_data])
                               )
    def load_data_fast(self, train_data_path, train_label_path, 
                        test_data_path, test_label_path):
        with open(train_label_path, "rb") as handle:
            self.train_label = pickle.load(handle)
        
        with open(train_data_path, "rb") as handle:
            self.train_data = pickle.load(handle)

        with open(test_label_path, "rb") as handle:
            self.test_label = pickle.load(handle)

        with open(test_data_path, "rb") as handle:
            self.test_data = pickle.load(handle)
        self.max_sent_len = max([max(len(sent) for sent in texts) for texts in [self.train_data, self.test_data]])
        self.max_char_num = max(
                                max([max(len(word) for word in sent) for sent in self.test_data]), 
                                max([max(len(word) for word in sent) for sent in self.train_data])
                               )

    def GetTrainingBatch(self, batchId, batchsize, max_word_num = -1, max_char_num = 21):
        if max_word_num == -1:
            max_word_num = self.max_sent_len
            
        def padding_sequence(max_len, sentence):
            placeholder = np.zeros([max_word_num, max_char_num], np.int32)
            rst = keras.preprocessing.sequence.pad_sequences(
                                                        sentence, 
                                                        maxlen=max_len, 
                                                        dtype='int32', 
                                                        padding='post', 
                                                        truncating='post',
                                                        value=0.0
            )
            placeholder[:len(rst)] = rst
            return placeholder

        startIdx = batchId*batchsize
        data_y = np.zeros([batchsize, 3], dtype=np.int32)
        ids = [idx%len(self.train_data) for idx in range(startIdx, startIdx+batchsize, 1)]
        data_x = np.array(
                        [padding_sequence(max_char_num, self.train_data[x]).tolist() 
                        for x in ids]
        )
        for i in range(batchsize):
            data_y[i][ int(self.train_label[ids[i]]/2) ] = 1
        return data_x, data_y
    
    def GetTestData(self, batchId, batchsize, max_char_num=21):
        def padding_sequence(max_len, sentence):
            placeholder = np.zeros([max_word_num, max_char_num], np.int32)
            rst = keras.preprocessing.sequence.pad_sequences(
                                                        sentence, 
                                                        maxlen=max_len, 
                                                        dtype='int32', 
                                                        padding='post', 
                                                        truncating='post',
                                                        value=0.0
            )
            placeholder[:len(rst)] = rst
            return placeholder
        
        startIdx = batchId*batchsize
        data_y = np.zeros([batchsize, 3], dtype=np.int32)
        ids = [idx%len(self.test_data) for idx in range(startIdx, startIdx+batchsize, 1)]
        data_x = np.array(
                        [padding_sequence(max_char_num, self.test_data[x]).tolist() 
                        for x in ids]
        )
        for idx in ids:
            data_y[idx][int(self.test_label[idx]/2)] = 1
        return data_x, data_y