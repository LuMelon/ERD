#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import json
import os
import time
import datetime
import numpy as np
import gensim
import random
import math
import re
import pickle


# In[2]:


import torch.nn.utils.rnn as rnn_utils


# In[3]:


files = []
data = {}
data_ID = []
data_len = []
data_y = []
reward_counter = 0
eval_flag = 0


# In[4]:


def get_data():
    global data
    return data

def get_data_ID():
    global data_ID
    return data_ID

def get_data_len():
    global data_len
    return data_len

def get_curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def get_data_y():
    global data_y
    return data_y

# In[5]:


def list_files(data_path):
    files = []
    fs = os.listdir(data_path)
    for f1 in fs:
        tmp_path = os.path.join(data_path, f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.split('.')[-1] == 'json':
                files.append(tmp_path)
        else:
            files.extend(list_files(tmp_path))
    return files


# In[6]:


def data_process(file_path):
    ret = {}
    ss = file_path.split("/")
    data = json.load(open(file_path, mode="r", encoding="utf-8"))
    # 'Wed Jan 07 11:14:08 +0000 2015'
    # print("SS:", ss)
    ret[ss[6]] = {'label': ss[5], 'text': [data['text'].lower()], 'created_at': [str2timestamp(data['created_at'])]}
    return ret


# In[7]:


def str2timestamp(str_time):
    month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
             'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
             'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    ss = str_time.split(' ')
    m_time = ss[5] + "-" + month[ss[1]] + '-' + ss[2] + ' ' + ss[3]
    d = datetime.datetime.strptime(m_time, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp


# In[8]:


def sortTempList(temp_list):
    time = np.array([item[0] for item in temp_list])
    posts = np.array([item[1] for item in temp_list])
    idxs = time.argsort().tolist()
    rst = [[t, p] for (t, p) in zip(time[idxs], posts[idxs])]
    del time, posts
    return rst


# In[9]:


def load_data(data_path, FLAGS):
    global data, data_ID, data_len, data_y, eval_flag
    files = list_files(data_path) #load all filepath to files
    max_sent = 0
    
    for file in files:
        td = data_process(file) # read out the information from json file, and organized it as {dataID:{'key':val}}
        for key in td.keys(): # use temporary data to organize the final whole data
            if key in data:
                data[key]['text'].append(td[key]['text'][0])
                data[key]['created_at'].append(td[key]['created_at'][0])
            else:
                data[key] = td[key]

    # convert to my data style
    for key, value in data.items():
        temp_list = []
        for i in range(len(data[key]['text'])):
            temp_list.append([data[key]['created_at'][i], data[key]['text'][i]])
        temp_list = sortTempList(temp_list)
        data[key]['text'] = []
        data[key]['created_at'] = []
        ttext = ""
        last = 0
        for i in range(len(temp_list)):
            if temp_list[i][0] - temp_list[0][0] > FLAGS.time_limit * 3600 or len(data[key]['created_at']) >= 100:
                break
            if i % FLAGS.post_fn == 0: # merge the fixed number of texts in a time interval
                if len(ttext) > 0: # if there are data already in ttext, output it as a new instance
                    data[key]['text'].append(ttext)
                    data[key]['created_at'].append(temp_list[i][0])
                ttext = temp_list[i][1]
            else:
                ttext += " " + temp_list[i][1]
            last = i
        # keep the last one
        if len(ttext) > 0:
            data[key]['text'].append(ttext)
            data[key]['created_at'].append(temp_list[last][0])

    for key in data.keys():
        data_ID.append(key)
    data_ID = random.sample(data_ID, len(data_ID)) #shuffle the data id
    for i in range(len(data_ID)): #pre processing the extra informations
        data_len.append(len(data[data_ID[i]]['text']))
        if data[data_ID[i]]['label'] == "rumours":
            data_y.append([1.0, 0.0])
        else:
            data_y.append([0.0, 1.0])
    eval_flag = int(len(data_ID) / 4) * 3
    print("{} data loaded".format(len(data)))


# In[10]:

def load_test_data_fast():
    global data, data_ID, data_len, data_y, eval_flag
    with open("data/data_dict.txt", "rb") as handle:
        data = pickle.load(handle)
    data_ID = np.load("data/test_data_ID.npy").tolist()
    data_len = np.load("data/test_data_len.npy").tolist()
    data_y = np.load("data/test_data_y.npy").tolist()
    max_sent = max( map(lambda value: max(map(lambda txt_list: len(txt_list), value['text']) ), list(data.values()) ) )
    print("max_sent:", max_sent, ",  max_seq_len:", max(data_len))
    eval_flag = int(len(data_ID) / 4) * 3
    print("{} data loaded".format(len(data))) 

def load_data_fast():
    global data, data_ID, data_len, data_y, eval_flag
    with open("data/data_dict.txt", "rb") as handle:
        data = pickle.load(handle)
    data_ID = np.load("data/data_ID.npy").tolist()
    data_len = np.load("data/data_len.npy").tolist()
    data_y = np.load("data/data_y.npy").tolist()
    max_sent = max( map(lambda value: max(map(lambda txt_list: len(txt_list), value['text']) ), list(data.values()) ) )
    print("max_sent:", max_sent, ",  max_seq_len:", max(data_len))
    eval_flag = int(len(data_ID) / 4) * 3
    print("{} data loaded".format(len(data)))    


# In[11]:


import json


# In[12]:


with open("config.json", "r") as cr:
    dic = json.load(cr)

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

FLAGS = adict(dic)


# In[19]:


# save the Twitter data
# with open('data/data_dict.txt', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# idxs = [ idx if item>6 else idx - len(data_len) for (idx, item) in enumerate(data_len)]
# saved_idxs = list(filter(lambda x: x > 0, idxs))
# np.save("data/data_ID.npy", np.array(data_ID)[saved_idxs])
# np.save("data/data_len.npy", np.array(data_len)[saved_idxs])
# np.save("data/data_y.npy", np.array(data_y)[saved_idxs])

    
# # save the PTB data
# with open('data/char_tensors.txt', 'wb') as handle:
#     pickle.dump(char_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/word_tensors.txt', 'wb') as handle:
#     pickle.dump(word_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('data/char_vocab.txt', 'wb') as handle:
#     pickle.dump(char_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/word_vocab.txt', 'wb') as handle:
#     pickle.dump(word_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save the senti data
# with open('data/senti_train_data.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/senti_train_label.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.train_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('data/senti_test_data.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/senti_test_label.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[29]:


def get_df_batch(start, batchsize, new_data_len=[], tokenizer=None):
#     data_x = np.zeros([batchsize, FLAGS.max_seq_len, FLAGS.max_sent_len, FLAGS.bert_embedding], 
#                       dtype=np.int32)
    m_data_y = np.zeros([batchsize, 2], dtype=np.int32)
    m_data_len = np.zeros([batchsize], dtype=np.int32)
    data_x = [] #[batchsize, seq_len, sent_len]
    if len(new_data_len) > 0:
        t_data_len = new_data_len
    else:
        t_data_len = data_len
    mts = start * batchsize
    if mts >= len(data_ID):
        mts = mts % len(data_ID)
    
    for i in range(batchsize):
        m_data_y[i] = data_y[mts]
        m_data_len[i] = t_data_len[mts]
        seq_x = [
            tokenizer.encode(
                data[data_ID[mts]]['text'][j],
                add_special_tokens=True
            )
            for j in range(t_data_len[mts])
        ]
        data_x.append(seq_x)
        mts += 1
        if mts >= len(data_ID): # read data looply
            mts = mts % len(data_ID)
    return data_x, m_data_len, m_data_y


# In[ ]:


def get_rl_batch(ids, seq_states, stop_states, counter_id, start_id, FLAGS, tokenizer=None):
#     input_x = np.zeros([FLAGS.batch_size, FLAGS.max_sent_len, FLAGS.max_char_num], dtype=np.float32)
    input_x = []  # [batch_size, sent_len]
    input_y = np.zeros([FLAGS.batch_size, FLAGS.class_num], dtype=np.float32)
    assert(len(ids)==FLAGS.batch_size)
    miss_vec = 0
    total_data = len(data_len)
    for i in range(FLAGS.batch_size):
        # seq_states:records the id of a sentence in a sequence
        # stop_states: records whether the sentence is judged by the program
        if stop_states[i] == 1 or seq_states[i] >= data_len[ids[i]]: 
            ids[i] = counter_id + start_id
            seq_states[i] = 0
            try:
                input_x.append(
                    tokenizer.encode(
                        data[ data_ID[ids[i]] ]['text'][seq_states[i]], 
                        add_special_tokens=True
                    )
                )
            except:
                print("ids and seq_states:", ids[i], seq_states[i])
                raise
            input_y[i] = data_y[ids[i]]
            counter_id += 1
            counter_id = counter_id % total_data
        else:
            try:
                input_x.append(
                    tokenizer.encode(
                        data[ data_ID[ids[i]] ]['text'][seq_states[i]], 
                        add_special_tokens=True
                    )
                )
            except:
                print("ids and seq_states:", ids[i], seq_states[i])
                raise
            input_y[i] = data_y[ids[i]]
        # point to the next sequence
        seq_states[i] += 1
    return input_x, input_y, ids, seq_states, counter_id


# In[26]:


def get_reward(isStop, ss, pys, ids, seq_ids):
    global reward_counter
    reward = torch.zeros([len(isStop)], dtype=torch.float32)
    for i in range(len(isStop)):
        if isStop[i] == 1:
            if pys[ids[i]][seq_ids[i]-1].argmax() == np.argmax(data_y[ids[i]]):
                reward_counter += 1 # more number of correct prediction, more rewards
                r = 1 + FLAGS.reward_rate * math.log(reward_counter)
                reward[i] = r   
            else:
                reward[i] = -100
        else:
            reward[i] = -0.01 + 0.99 * max(ss[i])
    return reward


def padding_sequence(sequences):
    max_size = sequences[0].size()
    trailing_dims = max_size[2:]
    max_seq_len = max([s.size(0) for s in sequences])
    max_sent_len = max([s.size(1) for s in sequences])
    out_dims = (len(sequences), max_seq_len, max_sent_len) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(0.0)
    for i, tensor in enumerate(sequences):
        seq_len = tensor.size(0)
        sent_len = tensor.size(1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :seq_len, :sent_len, ...] = tensor
    return out_tensor


# In[30]:

def get_RL_Train_batch(D, FLAGS, cuda=False):
    m_batch = random.sample(D, FLAGS.batch_size)
    s_state = torch.zeros([1, FLAGS.batch_size, FLAGS.hidden_dim], dtype=torch.float32)
    s_x = []
    s_isStop = torch.zeros([FLAGS.batch_size, FLAGS.action_num], dtype=torch.float32)
    s_rw = torch.zeros([FLAGS.batch_size], dtype=torch.float32)
    for i in range(FLAGS.batch_size):
        s_state[0][i] = m_batch[i][0]
        s_x.append(m_batch[i][1])
        s_isStop[i][m_batch[i][2]] = 1
        s_rw[i] = m_batch[i][3]
    if cuda:
        return s_state.cuda(), s_x, s_isStop.cuda(), s_rw.cuda()
    else:
        return s_state, s_x, s_isStop, s_rw


def get_new_len(sentence2vec, rdm_model, cm_model, FLAGS, cuda):
    # 因为计算句子向量的方式不一致，所以这个函数在生成句子向量的时候，应当要抽离出来,生成句子向量的部分在各个模型中自己定义
    ids = list(range(500))
    id_len = [0]*500
    stoped = [0]*500
    new_id = 501
    new_x_len = np.zeros([len(data_ID)], dtype=np.int32)
    zero_tensor = torch.zeros([1, 1, 768]) if not cuda else torch.zeros([1, 1, 768]).cuda()
    with torch.no_grad():
        while sum(stoped) != 500:
            init_state = torch.zeros([1, 500, FLAGS.hidden_dim], dtype=torch.float32) if not cuda else torch.zeros([1, 500, FLAGS.hidden_dim], dtype=torch.float32).cuda()
            try:
                sentences = [data[data_ID[idx]]['text'][idx_len] if stoped[t] == 0 else [] for (t, (idx, idx_len)) in enumerate(list(zip(ids, id_len)))]
            except:
                print(t, idx, idx_len)
                raise
            try:
                sentence_emb = torch.cat([sentence2vec(sent) if stoped[t]==0 else zero_tensor for (t,sent) in enumerate(sentences)], axis=0)
            except:
                print("sent:", sent)
                raise
            stopScore, isStop, rl_new_state = cm_model(rdm_model, sentence_emb, init_state)
            init_state = rl_new_state
            for k in range(len(isStop)):
                if stoped[k] == 1:
                    continue
                if isStop[k] == 1: # 如果在这个位置停止了，就要重新调入一个序列
                    new_x_len[ids[k]] = id_len[k]+1
                    if new_id < len(data_ID):
                        ids[k] = new_id
                        id_len[k] = 0
                        init_state[0][k].fill_(0.0)
                        new_id += 1
                    else:
                        stoped[k] = 1
                else:
                    id_len[k] += 1
                    if id_len[k] == data_len[ids[k]]:
                        new_x_len[ids[k]] = id_len[k]
                        if new_id < len(data_ID):
                            ids[k] = new_id
                            id_len[k] = 0
                            init_state[0][k].fill_(0.0)
                            new_id += 1
                        else:
                            stoped[k] = 1
                        
    for i in range(len(data_len)):    
        if new_x_len[i] == 0 or new_x_len[i] > data_len[i]:
            print("Error")
            new_x_len[i] = data_len[i]
    return new_x_len
#　先计算一个批次，这个批次会改变，直到不能再变
# 有两种情况需要改变:
#   第一，某个序列经过ｃｍ模型判定，他可以停止了
#   第二，某个序列一直不能停止，直到走到这个序列的尽头
# 不能再变有两个条件：　
# 改变的时候，要变三个点:
#   第一个是存储的ｉｄ要变，　
#   第二个是存储的len要变, 
#   第三个是要重置init_state
# 不能再变时，要把这个位置设置为停止