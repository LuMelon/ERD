#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys

sys.path.append(".")

from logger import MyLogger
import time
import SubjObjLoader
import json
from torch import nn
import torch
from pytorch_transformers import *
import importlib
from collections import deque
# import dataloader
from BertRDMLoader import *
import json
from torch import nn
import torch
from pytorch_transformers import *
import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import tsentiLoader
import pickle
import tqdm

import os
assert(len(sys.argv)==2)
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


class pooling_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(pooling_layer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, inputs):
        assert(inputs.ndim == 4 ) # [batchsize, max_seq_len, max_word_num, input_dim] 
        batch_size, max_seq_len, max_word_num, input_dim = inputs.shape
        assert(input_dim == self.input_dim)
        t_inputs = inputs.reshape([-1, self.input_dim])
        return self.linear(t_inputs).reshape(
            
            [-1, max_word_num, self.output_dim]
        
        ).max(axis=1)[0].reshape(
        
            [-1, max_seq_len, self.output_dim]
        
        )


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
#         self.PoolLayer = pooling_layer(word_embedding_dim, sent_embedding_dim) 
        
    def forward(self, x_emb): 
        """
        input_x: [batchsize, max_seq_len, sentence_embedding_dim] 
        x_emb: [batchsize, max_seq_len, 1, embedding_dim]
        x_len: [batchsize]
        init_states: [batchsize, hidden_dim]
        """
        batchsize, max_seq_len, _ , emb_dim = x_emb.shape
        init_states = torch.zeros([1, batchsize, self.hidden_dim], dtype=torch.float32).cuda()
        pool_feature = x_emb.reshape(
                [-1, max_seq_len, emb_dim]
        )
        try:
            df_outputs, df_last_state = self.gru_model(pool_feature, init_states)
        except:
            print("Error:", pool_feature.shape, init_states.shape)
            raise
        # hidden_outs = [df_outputs[i][:x_len[i]] for i in range(batchsize)]
        # final_outs = [df_outputs[i][x_len[i]-1] for i in range(batchsize)]
        # return hidden_outs, final_outs
        return df_outputs


# ##### CM 模型

# In[38]:


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


# #### 工具函数


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

def Loss_Fn(ylabel, pred_scores):
    diff = ((ylabel - pred_scores)*(ylabel - pred_scores)).mean(axis=1)
#     pos_neg = (1.0*sum(ylabel.argmax(axis=1)))/(1.0*(len(ylabel) - sum(ylabel.argmax(axis=1))))
    pos_neg = 0
    if pos_neg > 0:
        print("unbalanced data")
        weight = torch.ones(len(ylabel)).cuda() + (ylabel.argmax(axis=1).to(torch.float32)/(1.0*pos_neg)) - ylabel.argmax(axis=1).to(torch.float32)
        return (weight *diff).mean()
    else:
        print("totally unbalanced data")
        return diff.mean()

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

def sent_list2bert_tensors(sent_list, cuda):
    sent_len = [len(sent) for sent in sent_list]
    max_sent_len = max(sent_len)
    sent_padding = torch.zeros([len(sent_list), max_sent_len], dtype=torch.int64)
    attn_mask = torch.ones_like(sent_padding)
    for i, sent in enumerate(sent_list):
        sent_padding[i][:len(sent)] = torch.tensor(sent, dtype=torch.int32)
        attn_mask[i][len(sent):].fill_(0)
    if cuda:
        sent_padding = sent_padding.cuda()
        attn_mask = attn_mask.cuda()
    return sent_padding, attn_mask


# #### 训练RDM

# In[4]:


def TrainRDMModel(rdm_model, bert, rdm_classifier, 
                    tokenizer, t_steps, stage=0, new_data_len=[], valid_new_len=[], logger=None, 
                        log_dir="RDMBertTrain", cuda=True):
    batch_size = 20
    max_gpu_batch = 5 #cannot load a larger batch into the limited memory, but we could  accumulates grads
    splits = int(batch_size/max_gpu_batch)
    assert(batch_size%max_gpu_batch == 0)
    sum_loss = 0.0
    sum_acc = 0.0
    t_acc = 0.9
    ret_acc = 0.0
    init_states = torch.zeros([1, max_gpu_batch, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':1e-4},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-3},
                                {'params': rdm_model.parameters(), 'lr': 5e-3}
                             ]
    )
    
    writer = SummaryWriter(log_dir, filename_suffix="_ERD_CM_stage_%3d"%stage)
    acc_l = np.zeros(splits)
    loss_l = np.zeros(splits)

    best_valid_acc = 0.0
    for step in range(t_steps):
        optim.zero_grad()
        try:
            for j in range(splits):
                if len(new_data_len) > 0:
                    x, x_len, y = get_df_batch(step*batch_size+j*max_gpu_batch, max_gpu_batch, new_data_len, tokenizer=tokenizer)
                else:
                    x, x_len, y = get_df_batch(step*batch_size+j*max_gpu_batch, max_gpu_batch, tokenizer=tokenizer) 
                sent_tensors, attn_mask, seq_len = rdm_data2bert_tensors(x, cuda)
                bert_outs = bert(sent_tensors, attention_mask=attn_mask)
                pooled_sents = [bert_outs[1][sum(seq_len[:idx]):sum(seq_len[:idx])+seq_len[idx]] for idx, s_len in enumerate(seq_len)]
                data_tensors = rnn_utils.pad_sequence(pooled_sents, batch_first=True).unsqueeze(-2)
                rdm_hiddens = rdm_model(data_tensors)
                batchsize, _, _ = rdm_hiddens.shape
                rdm_outs = torch.cat(
                    [ rdm_hiddens[i][x_len[i]-1] for i in range(batchsize)] 
                    # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
                ).reshape(
                    [-1, rdm_model.hidden_dim]
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
                loss_l[j] = loss
                acc_l[j] = acc
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
        writer.add_scalar('Train Loss', loss_l.mean(), step)
        writer.add_scalar('Train Accuracy', acc_l.mean(), step)

        sum_loss += loss_l.mean()
        sum_acc += acc_l.mean()
        
        torch.cuda.empty_cache()
        
        if step % 10 == 9:
            sum_loss = sum_loss / 10
            sum_acc = sum_acc / 10
            print('%3d | %d , train_loss/accuracy = %6.8f/%6.7f'             % (step, t_steps, 
                sum_loss, sum_acc,
                ))
            if step%100 == 99:
                valid_acc = accuracy_on_valid_data(bert, rdm_model, rdm_classifier, valid_new_len, tokenizer)
                if valid_acc > best_valid_acc:
                    print("valid_acc:", valid_acc)
                    writer.add_scalar('Valid Accuracy', valid_acc, step)
                    best_valid_acc = valid_acc
                    rdm_save_as = '%s/BertRDM_best.pkl'% (log_dir)
                    torch.save(
                        {
                            "bert":bert.state_dict(),
                            "rmdModel":rdm_model.state_dict(),
                            "rdm_classifier": rdm_classifier.state_dict()
                        },
                        rdm_save_as
                    )
            sum_acc = 0.0
            sum_loss = 0.0
    print(get_curtime() + " Train df Model End.")
    return ret_acc


# #### 训练CM

# In[50]:


def TrainCMModel(bert, rdm_model, rdm_classifier, cm_model, tokenizer, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=False):
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
    rdm_optim = torch.optim.Adagrad([
                            {'params': bert.parameters(), 'lr':1e-6},
                            {'params': rdm_classifier.parameters(), 'lr': 5e-5},
                            {'params': rdm_model.parameters(), 'lr': 5e-5}
                         ],
                            weight_decay = 0.2
    )
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
            x, x_len, y = get_df_batch(i, batch_size, tokenizer=tokenizer)
            sent_tensors, attn_mask, seq_len = rdm_data2bert_tensors(x, cuda)
            bert_outs = bert(sent_tensors, attention_mask=attn_mask)
            pooled_sents = [bert_outs[1][sum(seq_len[:idx]):sum(seq_len[:idx])+seq_len[idx]] for idx, s_len in enumerate(seq_len)]
            x_emb = rnn_utils.pad_sequence(pooled_sents, batch_first=True).unsqueeze(-2)
            batchsize, max_seq_len, max_sent_len, emb_dim = x_emb.shape
            rdm_hiddens = rdm_model(x_emb)
            batchsize, _, _ = rdm_hiddens.shape
            tmp_hiddens = [ rdm_hiddens[i][:x_len[i]] for i in range(batchsize)] 

            rdm_hiddens_seq.extend(tmp_hiddens)
            print("batch %d"%i)
            if len(ssq) > 0:
                ssq.extend([rdm_classifier(h) for h in tmp_hiddens])
            else:
                ssq = [rdm_classifier(h) for h in tmp_hiddens]
    del rdm_hiddens, tmp_hiddens
    torch.cuda.empty_cache()

    # with open('cached_ssq.pickle', 'wb') as handle:
    #     pickle.dump(ssq, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(get_curtime() + " Now Start RL training ...")

    counter = 0
    sum_rw = 0.0 # sum of rewards
    data_len = get_data_len()

    while True:
        if counter > FLAGS.OBSERVE:
        # squeif counter > 20:
            sum_rw += rw.mean()
            if counter % 200 == 0:
                sum_rw = float(sum_rw) / 200
                print( get_curtime() + " Step: " + str(counter) 
                       + " REWARD IS " + str(sum_rw) 
                     )
                # if sum_rw > t_rw:
                #     print("Retch The Target Reward")
                #     break
                if counter > t_steps:
                    print("Retch The Target Steps")
                    break
                sum_rw = 0.0
            s_state, s_x, s_isStop, q_val = get_RL_Train_batch(D, FLAGS, cuda)
            sent_tensors, attn_mask = sent_list2bert_tensors(input_x, cuda)
            _, x_emb = bert(sent_tensors, attention_mask=attn_mask)
            x_emb = x_emb.unsqueeze(-2)
            stopScore, isStop, rl_new_state = cm_model(rdm_model, x_emb, s_state)
            out_action = (stopScore*s_isStop).sum(axis=1)
            rl_cost = torch.mean((q_val - out_action)*(q_val - out_action))
            rl_cost.backward()
            torch.cuda.empty_cache()
            rl_optim.step()
            print("RL Cost:", rl_cost)
            writer.add_scalar('RL Cost', rl_cost, counter - FLAGS.OBSERVE)

        input_x, input_y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, FLAGS, tokenizer=tokenizer)

        with torch.no_grad():
            sent_tensors, attn_mask = sent_list2bert_tensors(input_x, cuda)
            _, x_emb = bert(sent_tensors, attention_mask=attn_mask)
            x_emb = x_emb.unsqueeze(-2)
            batchsize, max_sent_len, emb_dim = x_emb.shape
            mss, isStop, mNewState = cm_model(rdm_model, x_emb, state)

        for j in range(FLAGS.batch_size):
            if random.random() < FLAGS.random_rate:
                isStop[j] = int(torch.rand(2).argmax()) # 设置了一个随机的干扰。
            if seq_states[j] == data_len[ids[j]]:
                isStop[j] = 1

        rw, QVal = get_reward_0(isStop, mss, ssq, ids, seq_states)
        if counter > FLAGS.OBSERVE:
            print("step:", counter - FLAGS.OBSERVE, ", reward:", rw.mean())
            print("step:", counter - FLAGS.OBSERVE, ", reward:", QVal.mean())
            writer.add_scalar('reward', rw.mean(), counter - FLAGS.OBSERVE)
        for j in range(FLAGS.batch_size):
            D.append((state[0][j], input_x[j], isStop[j], QVal[j]))
            # if len(D) > FLAGS.max_memory:
            #     D.popleft()

        state = mNewState
        for j in range(FLAGS.batch_size):
            if isStop[j] == 1:
                state[0][j].fill_(0)
        counter += 1



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

load_data_fast()


# #### 创建模型
tokenizer = BertTokenizer.from_pretrained("./bertModel/")
bert = BertModel.from_pretrained("./bertModel/").cuda()
rdm_model = RDM_Model(768, 300, 256, 0.2).cuda()
cm_model = CM_Model(300, 256, 2).cuda()
rdm_classifier = nn.Linear(256, 2).cuda()
cm_log_dir="CMBertTrain"


log_dir = os.path.join(sys.path[0], "BertRDM/")

if torch.cuda.device_count() > 1:
    # device_ids = [int(device_id) for device_id in sys.argv[1].split(",")]
    device_ids = list( range( len( sys.argv[1].split(",") ) ) )
    bert = nn.DataParallel(bert, device_ids=device_ids)
    device_name = "cuda:%d"%device_ids[0]
    device = torch.device(device_name)
    bert.to(device)

# #### 导入模型预训练参数
pretrained_file = "%s/BertRDM_best.pkl"%log_dir
if os.path.exists(pretrained_file):
    checkpoint = torch.load(pretrained_file)
    bert.load_state_dict(checkpoint['bert'])
    rdm_model.load_state_dict(checkpoint["rmdModel"])
    rdm_classifier.load_state_dict(checkpoint["rdm_classifier"])
else:
    TrainRDMModel(rdm_model, bert, rdm_classifier, 
                    tokenizer, 2000, stage=0, new_data_len=[], logger=None, 
                        log_dir=log_dir, cuda=True)


#### 标准ERD模型
for i in range(20):
    erd_save_as = '%s/erdModel_epoch%03d.pkl'% (log_dir , i)
    if i==0:
        if os.path.exists(erd_save_as):
            checkpoint = torch.load(erd_save_as)
            bert.load_state_dict(checkpoint['bert'])
            rdm_model.load_state_dict(checkpoint["rmdModel"])
            rdm_classifier.load_state_dict(checkpoint["rdm_classifier"])
            cm_model.load_state_dict(checkpoint['cm_model'])
        else:
            TrainCMModel(bert, rdm_model, rdm_classifier, cm_model, tokenizer, i, 0.5, 20000, "BertERD/", None, FLAGS, cuda=True)
    else:
        TrainCMModel(bert, rdm_model, rdm_classifier, cm_model, tokenizer, i, 0.5, 2000, "BertERD/", None, FLAGS, cuda=True)
    torch.save(
        {
            "bert":bert.state_dict(),
            "rmdModel":rdm_model.state_dict(),
            "rdm_classifier": rdm_classifier.state_dict(),
            "cm_model":cm_model.state_dict()
        },
        erd_save_as
    )
    print("iter:", i, ", train cm model completed!")
    new_len, valid_new_len = get_new_len(tokenizer, bert, rdm_model, cm_model, FLAGS, cuda=True)
    print("after new len:")
    print("new_data_len:", new_len)
    print("valid_new_len:", valid_new_len)
    TrainRDMModel(rdm_model, bert, rdm_classifier, 
                    tokenizer, 500, i, new_data_len=new_len, valid_new_len=valid_new_len, logger=None, 
                        log_dir=log_dir)





