#!/usr/bin/env python
# coding: utf-8
import sys
import torch
import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import pickle
import tqdm
import os
import torch.nn as nn
from collections import deque
sys.path.append(".")
from dataUtilsV0 import *
import json


# ### 模型训练与测试
class pooling_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(pooling_layer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, inputs, cuda=True):
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
        return df_outputs

class CM_Model_V1(nn.Module):
    def __init__(self, hidden_dim, action_num):
        super(CM_Model_V1, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_num = action_num
        self.DenseLayer = nn.Linear(self.hidden_dim, 64)
        self.Classifier = nn.Linear(64, self.action_num)
        
    def forward(self, rdm_state):
        """
        rdm_state: [batchsize, hidden_dim]
        """
        batchsize, hidden_dim = rdm_state.shape
        rl_h1 = nn.functional.relu(
            self.DenseLayer(
                rdm_state
            )
        )
        stopScore = self.Classifier(rl_h1)
        isStop = stopScore.argmax(axis=1)
        return stopScore, isStop

class CM_Model(nn.Module):
    def __init__(self, sentence_embedding_dim, hidden_dim, action_num):
        super(CM_Model, self).__init__()
        self.sentence_embedding_dim = sentence_embedding_dim
        self.hidden_dim = hidden_dim
        self.action_num = action_num
#         self.PoolLayer = pooling_layer(self.embedding_dim, 
#                                             self.hidden_dim)
        self.DenseLayer = nn.Linear(self.hidden_dim, 64)
        self.Classifier = nn.Linear(64, self.action_num)
        
    def forward(self, rdm_model, rl_input, rl_state):
        """
        rl_input: [batchsize, max_word_num, sentence_embedding_dim]
        rl_state: [1, batchsize, hidden_dim]
        """
        assert(rl_input.ndim==3)
        batchsize, max_word_num, embedding_dim = rl_input.shape
        rl_output, rl_new_state = rdm_model.gru_model(
                                            rl_input, 
                                            rl_state
                                        )
        rl_h1 = nn.functional.relu(
            self.DenseLayer(
#                 rl_state.reshape([len(rl_input), self.hidden_dim]) #it is not sure to take rl_state , rather than rl_output, as the feature
                rl_output.reshape(
                    [len(rl_input), self.hidden_dim]
                )
            )
        )
        stopScore = self.Classifier(rl_h1)
        isStop = stopScore.argmax(axis=1)
        return stopScore, isStop, rl_new_state

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
                            "sent_pooler":sent_pooler.state_dict(),
                            "rdm_classifier": rdm_classifier.state_dict()
                        },
                        rdm_save_as
                    )
            sum_acc = 0.0
            sum_loss = 0.0
    print(get_curtime() + " Train df Model End.")
    return ret_acc

def TrainCMModel(sent_pooler, rdm_model, rdm_classifier, cm_model, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=True):
    batch_size = 20
    t_acc = 0.9
    ids = np.array(range(batch_size), dtype=np.int32)
    seq_states = np.zeros([batch_size], dtype=np.int32)
    isStop = np.zeros([batch_size], dtype=np.int32)
    max_id = batch_size
    df_init_states = torch.zeros([1, batch_size, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    writer = SummaryWriter(log_dir, filename_suffix="_ERD_CM_stage_%3d"%stage)
    state = df_init_states
    D = deque()
    ssq = []
    print("in RL the begining")
    rl_optim = torch.optim.Adam([{'params':cm_model.parameters(), 'lr':1e-5}])
    # get_new_len(sess, mm)
    data_ID = get_data_ID()

    if len(data_ID) % batch_size == 0: # the total number of events
            flags = int(len(data_ID) / FLAGS.batch_size)
    else:
        flags = int(len(data_ID) / FLAGS.batch_size) + 1

    rdm_hiddens_seq = []
    for i in range(flags):
        with torch.no_grad():
            x, x_len, y = get_df_batch(i, batch_size)
            seq = sent_pooler(x)
            rdm_hiddens = rdm_model(seq)
            batchsize, _, _ = rdm_hiddens.shape
            tmp_hiddens = [ rdm_hiddens[i][:x_len[i]] for i in range(batchsize)] 

            rdm_hiddens_seq.extend(tmp_hiddens)
            print("batch %d"%i)
            if len(ssq) > 0:
                ssq.extend([rdm_classifier(h) for h in tmp_hiddens])
            else:
                ssq = [rdm_classifier(h) for h in tmp_hiddens]
            torch.cuda.empty_cache()
    del rdm_hiddens, tmp_hiddens

    print(get_curtime() + " Now Start RL training ...")

    counter = 0
    sum_rw = 0.0 # sum of rewards
    data_len = get_data_len()

    while True:
    #         if counter > FLAGS.OBSERVE:
        if counter > 1000:
            if counter > t_steps:
                print("Retch The Target Steps")
                break
            rdm_state, s_ids, s_seq_states = get_RL_Train_batch_V1(D, FLAGS, batch_size, cuda)
            print("s_seq_states:", s_seq_states)
            with torch.no_grad():
                s_stopScore, s_isStop = cm_model(rdm_state)                
                rw, q_val = get_reward_01(s_isStop, s_stopScore, ssq, s_ids, s_seq_states)
            
            s_isStop = s_stopScore.argsort()
            for j in range(batch_size):
                if random.random() < FLAGS.random_rate:
                    s_isStop[j][int(torch.rand(2).argmax())] = 1 # 设置了一个随机的干扰。
                if seq_states[j] == data_len[s_ids[j]]:
                    s_isStop[j] = 1

            t_stopScore, t_isStop = cm_model(rdm_state)
            out_action = (t_stopScore*s_isStop.float()).sum(axis=1)
            rl_cost = torch.mean((q_val.cuda() - out_action)*(q_val.cuda() - out_action))
            rl_cost.backward()
            torch.cuda.empty_cache()
            rl_optim.step()
            print("RL Cost:", rl_cost)
            writer.add_scalar('RL Cost', rl_cost, counter - FLAGS.OBSERVE)

        ids, seq_states, max_id = get_rl_batch(ids, seq_states, max_id, 0, FLAGS)
        if counter > FLAGS.OBSERVE:
            print("step:", counter - FLAGS.OBSERVE, ", reward:", rw.mean())
            print("step:", counter - FLAGS.OBSERVE, ", reward:", q_val.mean())
            print("rw:", rw)
            print("q_val", q_val)
            print("stopScore:", t_stopScore)
            writer.add_scalar('reward', rw.mean(), counter - FLAGS.OBSERVE)
        for j in range(batch_size):
            D.append((rdm_hiddens_seq[ids[j]][seq_states[j]-1], ids[j], seq_states[j]))
            if len(D) > FLAGS.max_memory:
                D.popleft()
        counter += 1

# In[13]:


load_data_fast()

rdm_model = RDM_Model(300, 300, 256, 0.2).cuda()
sent_pooler = pooling_layer(300, 300).cuda()
rdm_classifier = nn.Linear(256, 2).cuda()
cm_model = CM_Model_V1(256, 2).cuda()

log_dir = os.path.join(sys.path[0], "ERD/")

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

# #### 导入模型预训练参数
pretrained_file = "%s/ERD_best.pkl"%log_dir
if os.path.exists(pretrained_file):
    checkpoint = torch.load(pretrained_file)
    sent_pooler.load_state_dict(checkpoint['sent_pooler'])
    rdm_model.load_state_dict(checkpoint["rmdModel"])
    rdm_classifier.load_state_dict(checkpoint["rdm_classifier"])
else:
    TrainRDMModel(rdm_model, sent_pooler, rdm_classifier, 
                    t_steps=5000, stage=0, new_data_len=[], valid_new_len=[], logger=None, 
                        log_dir=log_dir, cuda=True)



#### 标准ERD模型
for i in range(20):
    erd_save_as = '%s/erdModel_epoch%03d.pkl'% (log_dir , i)
    if i==0:
        TrainCMModel(sent_pooler, rdm_model, rdm_classifier, cm_model, 0, 0.5, 20000, log_dir, None, FLAGS, cuda=True)
    else:
        TrainCMModel(sent_pooler, rdm_model, rdm_classifier, cm_model, 0, 0.5, 2000, log_dir, None, FLAGS, cuda=True)
    torch.save(
        {
            "sent_pooler":sent_pooler.state_dict(),
            "rmdModel":rdm_model.state_dict(),
            "rdm_classifier": rdm_classifier.state_dict(),
            "cm_model":cm_model.state_dict()
        },
        erd_save_as
    )
    print("iter:", i, ", train cm model completed!")
    new_len, valid_new_len = get_new_len(sent_pooler, rdm_model, cm_model, FLAGS, cuda=True)
    print("after new len:")
    print("new_data_len:", new_len)
    print("valid_new_len:", valid_new_len)
    TrainRDMModel(rdm_model, sent_pooler, rdm_classifier, 
                    t_steps=1000, stage=0, new_data_len=[], valid_new_len=[], logger=None, 
                        log_dir=log_dir, cuda=True)




