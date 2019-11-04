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
import transformer_utils

import os
assert(len(sys.argv)==2)
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


# ##### CM 模型

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

# In[23]:


def Sent2Vec_Generater(tokenizer, bert, cuda):
    def fn(sentence):
        input_ids = tokenizer.encode(
                            sentence,
                            add_special_tokens=True
                        )
        input_ids = torch.tensor([input_ids]).cuda() if cuda else torch.tensor([input_ids])
        outs = bert(input_ids)
        sentence_emb = outs[1].reshape([1, 1,-1])
        return sentence_emb
    return fn



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


def TrainRDMModel(bert, rdm_transformer, rdm_classifier, 
                    tokenizer, t_steps, stage=0, new_data_len=[], logger=None, 
                        log_dir="RDMBertTrain", cuda=True):
    batch_size = 20 
    max_gpu_batch = 5 #cannot load a larger batch into the limited memory, but we could  accumulates grads
    splits = int(batch_size/max_gpu_batch)
    assert(batch_size%max_gpu_batch == 0)
    sum_loss = 0.0
    sum_acc = 0.0
    t_acc = 0.9
    ret_acc = 0.0
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adam([
                                {'params': bert.parameters(), 'lr':2e-5},
                                {'params': rdm_transformer.parameters(), 'lr':2e-3},
                                {'params': rdm_classifier.parameters(), 'lr': 2e-3},
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
                data_tensors = rnn_utils.pad_sequence(pooled_sents, batch_first=True)
                seq_mask = torch.ones([len(seq_len), max(seq_len)]).cuda() if cuda else torch.ones([len(seq_len), max(seq_len)])
                for i in range(len(seq_len)):
                    seq_mask[i][seq_len[i]:].fill_(0)
                rdm_outs = rdm_transformer(data_tensors, attention_mask = seq_mask)
                rdm_scores = rdm_classifier(
                    rdm_outs[0][:, 0]
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
                valid_acc = accuracy_on_valid_data(bert, rdm_model, rdm_classifier, [], tokenizer)
                if valid_acc > best_valid_acc:
                    print("valid_acc:", valid_acc)
                    writer.add_scalar('Valid Accuracy', valid_acc, step)
                    best_valid_acc = valid_acc
                    rdm_save_as = '%s/BertRDM_best.pkl'% (log_dir)
                    torch.save(
                        {
                            "bert":bert.state_dict(),
                            "rdm_transformer":rdm_transformer.state_dict(),
                            "rdm_classifier": rdm_classifier.state_dict()
                        },
                        rdm_save_as
                    )
#                 rdm_model, bert, sentiModel, rdm_classifier
            sum_acc = 0.0
            sum_loss = 0.0
    print(get_curtime() + " Train df Model End.")
    return ret_acc



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



trans_conf = adict({
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "num_attention_heads": 8,
  "num_hidden_layers": 4,
  "num_labels": 2,
  "output_attentions": False,
  "output_hidden_states": False,
  "torchscript": False
})
BertEncoder = transformer_utils.BertEncoder
rdm_transformer = BertEncoder(trans_conf).cuda()

cm_model = CM_Model(300, 256, 2).cuda()
rdm_classifier = nn.Linear(768, 2).cuda()
cm_log_dir="CMBertTrain"


log_dir = os.path.join(sys.path[0], "BertRDM/")

if torch.cuda.device_count() > 1:
    # device_ids = [int(device_id) for device_id in sys.argv[1].split(",")]
    device_ids = list( range( len( sys.argv[1].split(",") ) ) )
    bert = nn.DataParallel(bert, device_ids=device_ids)
    rdm_transformer = nn.DataParallel(rdm_transformer, device_ids=device_ids)
    device_name = "cuda:%d"%device_ids[0]
    device = torch.device(device_name)
    bert.to(device)
    rdm_transformer.to(device)

# #### 导入模型预训练参数
pretrained_file = "%s/BertRDM_best.pkl"%log_dir
if os.path.exists(pretrained_file):
    checkpoint = torch.load(pretrained_file)
    bert.load_state_dict(checkpoint['bert'])
    rdm_transformer.load_state_dict(checkpoint["rdm_transformer"])
    rdm_classifier.load_state_dict(checkpoint["rdm_classifier"])
else:
    TrainRDMModel(bert, rdm_transformer, rdm_classifier, 
                    tokenizer, 2000, stage=0, new_data_len=[], logger=None, 
                        log_dir=log_dir, cuda=True)


# #### 标准ERD模型
# for i in range(20):
#     if i==0:
#         TrainCMModel(bert, rdm_model, rdm_classifier, cm_model, tokenizer, i, 0.5, 50000, "BertERD/", None, FLAGS, cuda=True)
#     else:
#         TrainCMModel(bert, rdm_model, rdm_classifier, cm_model, tokenizer, i, 0.5, 5000, "BertERD/", None, FLAGS, cuda=True)
#     erd_save_as = './BertERD/erdModel_epoch%03d.pkl'% (i)
#     torch.save(
#         {
#             "bert":bert.state_dict(),
#             "rmdModel":rdm_model.state_dict(),
#             "rdm_classifier": rdm_classifier.state_dict(),
#             "cm_model":cm_model.state_dict()
#         },
#         erd_save_as
#     )
#     s2vec = Sent2Vec_Generater(tokenizer, bert, cuda=True)
#     new_len = get_new_len(s2vec, rdm_model, cm_model, FLAGS, cuda=True)
#     print("after new len:")
#     TrainRDMModel(rdm_model, bert, rdm_classifier, 
#                     tokenizer, i, 1000, new_data_len=new_len, logger=None, 
#                         log_dir="BertERD_%d"%i)

# In[ ]:




