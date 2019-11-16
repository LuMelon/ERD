#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


import torch


# In[3]:


import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import pickle
import tqdm
import os


# In[4]:


import torch.nn as nn


# In[5]:


sys.path.append(".")


# In[6]:


from dataUtils import *


# In[7]:


import json


# ### 数据集分析

# In[9]:


with open("../../config.json", "r") as cr:
    dic = json.load(cr)

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

FLAGS = adict(dic)


# In[10]:


load_data("/home/hadoop/pheme-rnr-dataset/", FLAGS)


# In[11]:


del data
del data_ID
del data_y
del data_len
from dataUtils import data
from dataUtils import data_ID
from dataUtils import data_y
from dataUtils import data_len


# In[12]:


import time


# In[13]:


t_hour = [(data[data_ID[i]]['created_at'][-1]-data[data_ID[i]]['created_at'][0])/3600.0 for i in range(len(data_ID))] 


# In[14]:


import seaborn as sns


# In[41]:


def plot_hist(x_tuples, bins, xlabel, ylabel, legends, title):
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0,), (0, 0, 1)]
    def normfun(x,mu,sigma):
        pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        return pdf
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
#     plt.xlim(0, 200)
    for i in range(len(x_tuples)):
        sns.distplot(x_tuples[i], bins=bins, rug=False, kde=True, hist=True, norm_hist=True, label=legends[i], hist_kws={"histtype": "step", "linewidth": 2,
        "alpha": 1}, kde_kws={"color": colors[0], "lw": 0, "label": ""})


# In[16]:


import matplotlib.pyplot as plt


# In[28]:


len(t_hour), max(t_hour), min(t_hour)


# In[38]:


max(data_len), min(data_len)


# In[42]:


plot_hist([data_len], 1000, "tweets", "percentage", [""], "statics of the dataset")


# ### 切割数据集

# In[43]:


len(data_ID), len(data_y), len(data_len)


# In[52]:


new_y =[]
new_ID = []
new_len = []

for l, ID, y in zip(data_len, data_ID, data_y):
    if l > 5:
        new_len.append(l)
        new_ID.append(ID)
        new_y.append(y)


# In[53]:


len(new_ID), len(new_len), len(new_y)


# In[62]:


with open('data/data_dict.txt', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
np.save("data/data_ID.npy", np.array(new_ID)[:-500])
np.save("data/data_len.npy", np.array(new_len)[:-500])
np.save("data/data_y.npy", np.array(new_y)[:-500])

np.save("data/test_data_ID.npy", np.array(new_ID)[-500:])
np.save("data/test_data_len.npy", np.array(new_len)[-500:])
np.save("data/test_data_y.npy", np.array(new_y)[-500:])


# ### 模型训练与测试

# In[8]:


class pooling_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(pooling_layer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, inputs, cuda=True):
        # [batchsize, max_seq_len, max_word_num, input_dim] 
#         batch_size, max_seq_len, max_word_num, input_dim = inputs.shape
#         assert(input_dim == self.input_dim)
#         t_inputs = inputs.reshape([-1, self.input_dim])
#         return self.linear(t_inputs).reshape(
            
#             [-1, max_word_num, self.output_dim]
        
#         ).max(axis=1)[0].reshape(
        
#             [-1, max_seq_len, self.output_dim]
        
#         )
        inputs_sent = [torch.cat([self.linear(sent_tensor.cuda() if cuda else sent_tensor).max(axis=0)[0].unsqueeze(0) for sent_tensor in seq]) for seq in inputs]
        seqs = torch.nn.utils.rnn.pad_sequence(inputs_sent, batch_first=True)
        return seqs

class RDM_Model(nn.Module):
    def __init__(self, word_embedding_dim, sent_embedding_dim, hidden_dim, dropout_prob):
        super(RDM_Model, self).__init__()
        self.embedding_dim = sent_embedding_dim
        self.hidden_dim = hidden_dim
        self.gru_model = nn.GRU(word_embedding_dim, 
                                self.hidden_dim, 
                                batch_first=True, 
                                dropout=dropout_prob
                            )
        self.DropLayer = nn.Dropout(dropout_prob)

    def forward(self, input_x): 
        """
        input_x: [batchsize, max_seq_len, sentence_embedding_dim] 
        x_len: [batchsize]
        init_states: [batchsize, hidden_dim]
        """
        batchsize, max_seq_len, emb_dim = input_x.shape
        init_states = torch.zeros([1, batchsize, self.hidden_dim], dtype=torch.float32).cuda()
        try:
            df_outputs, df_last_state = self.gru_model(input_x, init_states)
        except:
            print("Error:", pool_feature.shape, init_states.shape)
            raise
        # hidden_outs = [df_outputs[i][:x_len[i]] for i in range(batchsize)]
        # final_outs = [df_outputs[i][x_len[i]-1] for i in range(batchsize)]
        # return hidden_outs, final_outs
        return df_outputs


# In[11]:


def Count_Accs(ylabel, preds):
    correct_preds = np.array(
        [1 if y1==y2 else 0 
        for (y1, y2) in zip(ylabel, preds)]
    )
    y_idxs = [idx if yl >0 else idx - len(ylabel) 
            for (idx, yl) in enumerate(ylabel)]
    pos_idxs = list(filter(lambda x: x >= 0, y_idxs))
    neg_idxs = list(filter(lambda x: x < 0, y_idxs))
    acc = sum(correct_preds) / (1.0 * len(ylabel))
    if len(pos_idxs) > 0:
        pos_acc = sum(correct_preds[pos_idxs])/(1.0*len(pos_idxs))
    else:
        pos_acc = 0
    if len(neg_idxs) > 0:
        neg_acc = sum(correct_preds[neg_idxs])/(1.0*len(neg_idxs))
    else:
        neg_acc = 0
    return acc, pos_acc, neg_acc, y_idxs, pos_idxs, neg_idxs, correct_preds


# In[12]:


def TrainRDMModel(rdm_model, sent_pooler, rdm_classifier, 
                    t_steps=100, stage=0, new_data_len=[], valid_new_len=[], logger=None, 
                        log_dir="RDMBertTrain", cuda=True):
    batch_size = 20
    sum_loss = 0.0
    sum_acc = 0.0
    t_acc = 0.9
    ret_acc = 0.0
    init_states = torch.zeros([1, batch_size, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adagrad([
                                {'params': sent_pooler.parameters(), 'lr': 5e-3},
                                {'params': rdm_model.parameters(), 'lr': 5e-3},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-3}
                             ]
    )
    
    writer = SummaryWriter(log_dir, filename_suffix="_ERD_CM_stage_%3d"%stage)
    best_valid_acc = 0.0
    for step in range(t_steps):
        optim.zero_grad()
        try:
            x, x_len, y = get_df_batch(step*batch_size, batch_size)
            seq = sent_pooler(x)
            rdm_hiddens = rdm_model(seq)
            batchsize, _, _ = rdm_hiddens.shape
            rdm_outs = torch.cat(
                [ rdm_hiddens[i][x_len[i]-1].unsqueeze(0) for i in range(batchsize)] 
                # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
            )
            rdm_scores = rdm_classifier(
                rdm_outs
            )
            rdm_preds = rdm_scores.argmax(axis=1)
            y_label = torch.tensor(y).argmax(axis=1).cuda() if cuda else torch.tensor(y).argmax(axis=1)
            acc, _, _, _, _, _, _ = Count_Accs(y_label, rdm_preds)
            loss = loss_fn(rdm_scores, y_label)
            loss.backward()
            torch.cuda.empty_cache()
#                 print("%d, %d | x_len:"%(step, j), x_len)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                print("%d, %d | x_len:"%(step, j), x_len)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
#                     time.sleep(5)
                raise exception
            else:   
                raise exception

        optim.step()        
        writer.add_scalar('Train Loss', loss, step)
        writer.add_scalar('Train Accuracy', acc, step)

        sum_loss += loss
        sum_acc += acc
        
        torch.cuda.empty_cache()
        
        if step % 10 == 9:
            sum_loss = sum_loss / 10
            sum_acc = sum_acc / 10
            print('%3d | %d , train_loss/accuracy = %6.8f/%6.7f'             % (step, t_steps, 
                sum_loss, sum_acc,
                ))
            if step%100 == 99:
                valid_acc = accuracy_on_valid_data(rdm_model, sent_pooler, rdm_classifier)
                if valid_acc > best_valid_acc:
                    print("valid_acc:", valid_acc)
                    writer.add_scalar('Valid Accuracy', valid_acc, step)
                    best_valid_acc = valid_acc
                    rdm_save_as = '%s/ERD_best.pkl'% (log_dir)
                    torch.save(
                        {
                            "rmdModel":rdm_model.state_dict(),
                            "bert":sent_pooler.state_dict(),
                            "rdm_classifier": rdm_classifier.state_dict()
                        },
                        rdm_save_as
                    )
            sum_acc = 0.0
            sum_loss = 0.0
    print(get_curtime() + " Train df Model End.")
    return ret_acc


# In[13]:


load_data_fast()


# In[9]:


rdm_model = RDM_Model(300, 300, 256, 0.2).cuda()
sent_pooler = pooling_layer(300, 300).cuda()


# In[10]:


rdm_classifier = nn.Linear(256, 2).cuda()


# In[14]:


TrainRDMModel(rdm_model, sent_pooler, rdm_classifier, 
                    t_steps=10000, stage=0, new_data_len=[], valid_new_len=[], logger=None, 
                        log_dir="RDMBertTrain", cuda=True)


# ### 原始CM模型

# In[ ]:




