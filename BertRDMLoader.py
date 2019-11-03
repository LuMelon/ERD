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
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


# In[3]:


files = []
data = {}
data_ID = []
data_len = []
data_y = []
valid_data_ID = []
valid_data_len = []
valid_data_y = []

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
    global data, data_ID, data_len, data_y, valid_data_ID, valid_data_y, valid_data_len
    with open("data/data_dict.txt", "rb") as handle:
        data = pickle.load(handle)
    data_ID = np.load("data/data_ID.npy").tolist()
    data_len = np.load("data/data_len.npy").tolist()
    data_y = np.load("data/data_y.npy").tolist()
    # valid_data_ID = np.load("data/valid_data_ID.npy").tolist()
    # valid_data_len = np.load("data/valid_data_len.npy").tolist()
    # valid_data_y = np.load("data/valid_data_y.npy").tolist()
    valid_data_ID = np.load("data/test_data_ID.npy").tolist()
    valid_data_len = np.load("data/test_data_len.npy").tolist()
    valid_data_y = np.load("data/test_data_y.npy").tolist()
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


def accuracy_on_valid_data(bert = None, rdm_model = None, rdm_classifier=None, new_data_len=[], tokenizer=None, cuda=True):
    batchsize = 20
    m_data_y = np.zeros([batchsize, 2], dtype=np.int32)
    m_data_len = np.zeros([batchsize], dtype=np.int32)
    data_x = [] #[batchsize, seq_len, sent_len]
    if len(new_data_len) > 0:
        t_data_len = new_data_len
    else:
        t_data_len = valid_data_len
        
    t_steps = int(len(valid_data_ID)/batchsize)
    sum_acc = 0.0
    
    def rdm_data2bert_tensors(data_X, cuda):
        def padding_sent_list(sent_list):
            sent_len = [len(sent) for sent in sent_list]
            max_sent_len = max(sent_len)
            sent_padding = torch.zeros([len(sent_list), max_sent_len], dtype=torch.int64)
            attn_mask = torch.ones_like(sent_padding)
            for i, sent in enumerate(sent_list):
                sent_padding[i][:len(sent)] = torch.tensor(sent, dtype=torch.int32)
                attn_mask[i][len(sent):].fill_(0)
            return sent_padding, attn_mask
        sent_list = []
        [sent_list.extend(seq) for seq in data_X]
        seq_len = [len(seq) for seq in data_X]
        sent_tensors, attn_mask = padding_sent_list(sent_list)
        if cuda:
            sent_tensors = sent_tensors.cuda()
            attn_mask = attn_mask.cuda()
        return sent_tensors, attn_mask, seq_len

    def Count_Acc(ylabel, preds):
        correct_preds = np.array(
            [1 if y1==y2 else 0 
            for (y1, y2) in zip(ylabel, preds)]
        )
        acc = sum(correct_preds) / (1.0 * len(ylabel))
        return acc

    for step in range(t_steps):
        for i in range(batchsize):
            m_data_y[i] = valid_data_y[step*batchsize + i]
            m_data_len[i] = t_data_len[step*batchsize + i]
            seq_x = [
                tokenizer.encode(
                    data[valid_data_ID[step*batchsize+i]]['text'][j],
                    add_special_tokens=True
                )
                for j in range(t_data_len[step*batchsize+i])
            ]
            data_x.append(seq_x)
        acc = 0.0 
        if bert is not None and rdm_model is not None and rdm_classifier is not None:
            with torch.no_grad():
                sent_tensors, attn_mask, seq_len = rdm_data2bert_tensors(data_x, cuda)
                bert_outs = bert(sent_tensors, attention_mask=attn_mask)
                pooled_sents = [bert_outs[1][sum(seq_len[:idx]):sum(seq_len[:idx])+seq_len[idx]] for idx, s_len in enumerate(seq_len)]
                data_tensors = rnn_utils.pad_sequence(pooled_sents, batch_first=True).unsqueeze(-2)
                rdm_hiddens = rdm_model(data_tensors)
                batch_size, _, _ = rdm_hiddens.shape
                rdm_outs = torch.cat(
                    [ rdm_hiddens[i][m_data_len[i]-1] for i in range(batch_size)] 
                    # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
                ).reshape(
                    [-1, rdm_model.hidden_dim]
                )
                rdm_scores = rdm_classifier(
                    rdm_outs
                )
                rdm_preds = rdm_scores.argmax(axis=1)
                y_label = torch.tensor(m_data_y).argmax(axis=1).cuda() if cuda else torch.tensor(m_data_y).argmax(axis=1)
                acc = Count_Acc(y_label, rdm_preds)
        data_x = []
        sum_acc += acc
    mean_acc = sum_acc / (1.0*t_steps)
    return mean_acc

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
            # stop之后, 要换一个新的序列，新序列的下标也要重新进行标记，从头开始计数.
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
    Q_Val = torch.zeros([len(isStop)], dtype= torch.float32)
    for i in range(len(isStop)):
        if isStop[i] == 1:
            if pys[ids[i]][seq_ids[i]-1].argmax() == np.argmax(data_y[ids[i]]):
                reward_counter += 1 # more number of correct prediction, more rewards
                r = 1 + min(FLAGS.reward_rate * math.log(reward_counter), 10)
                reward[i] = r   
            else:
                reward[i] = -100
            Q_Val[i] = reward[i]
        else:
            reward[i] = -0.01 
            Q_Val[i] = reward[i] + 0.99 * max(ss[i])
    return reward, Q_Val


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

def rdm_data2bert_tensors(data_X, cuda):
    def padding_sent_list(sent_list):
        sent_len = [len(sent) for sent in sent_list]
        max_sent_len = max(sent_len)
        sent_padding = torch.zeros([len(sent_list), max_sent_len], dtype=torch.int64)
        attn_mask = torch.ones_like(sent_padding)
        for i, sent in enumerate(sent_list):
            sent_padding[i][:len(sent)] = torch.tensor(sent, dtype=torch.int32)
            attn_mask[i][len(sent):].fill_(0)
        return sent_padding, attn_mask
    sent_list = []
    [sent_list.extend(seq) for seq in data_X]
    seq_len = [len(seq) for seq in data_X]
    sent_tensors, attn_mask = padding_sent_list(sent_list)
    if cuda:
        sent_tensors = sent_tensors.cuda()
        attn_mask = attn_mask.cuda()
    return sent_tensors, attn_mask, seq_len


def get_new_len(tokenizer, bert, rdm_model, cm_model, FLAGS, cuda):
    batch_size = 20
    new_len = []
    if len(data_ID) % batch_size == 0: # the total number of events
        flags = int(len(data_ID) / FLAGS.batch_size)
    else:
        flags = int(len(data_ID) / FLAGS.batch_size) + 1
    for i in range(flags):
        with torch.no_grad():
            x, x_len, y = get_df_batch(i, batch_size, tokenizer=tokenizer)
            sent_tensors, attn_mask, seq_len = rdm_data2bert_tensors(x, cuda)
            bert_outs = bert(sent_tensors, attention_mask=attn_mask)
            pooled_sents = [bert_outs[1][sum(seq_len[:idx]):sum(seq_len[:idx])+seq_len[idx]] for idx, s_len in enumerate(seq_len)]
            x_emb = rnn_utils.pad_sequence(pooled_sents, batch_first=True).unsqueeze(-2)
            batchsize, max_seq_len, max_sent_len, emb_dim = x_emb.shape
            rdm_hiddens = rdm_model(x_emb)
            batchsize, _, _ = rdm_hiddens.shape
            rdm_outs = torch.cat(
                [ rdm_hiddens[i][x_len[i]-1] for i in range(batchsize)] 
                # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
            ).reshape(
                [-1, rdm_model.hidden_dim]
            )
            stopScores = cm_model.Classifier(
                    nn.functional.relu(
                        cm_model.DenseLayer(
                            rdm_hiddens.reshape([-1, rdm_model.hidden_dim])
                    )
                )
            ).reshape(
                [batchsize, -1, 2]
            )
            isStop = stopScores.argmax(axis=-1).cpu().numpy()
            new_len.extend([iS.argmax()+1 if iS.argmax() <= x_len[i] and iS.max() ==1 else x_len[i] for i, iS in enumerate(isStop)])
    print("max_new_len:", max(new_len))
    print("mean_new_len:", sum(new_len)*1.0/len(new_len))
    return new_len[:len(data_len)]
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


# 这个函数太慢了，而且像是一个死循环