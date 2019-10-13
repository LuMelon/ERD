#!/usr/bin/env python
# coding: utf-8

# In[1]:


from logger import MyLogger
import SubjObjLoader
import json
from torch import nn
import torch
from pytorch_transformers import *
import importlib
from collections import deque
# import dataloader
from BertRDMLoader import *
import time
import json
import sys
from torch import nn
import torch
from pytorch_transformers import *
import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import tsentiLoader
from emotionLoader import *
from SubjObjLoader import *
import numpy as np


# In[2]:
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


# In[4]:


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
    
def WeightsForUmbalanced(data_label):
    _, _, labels = data_label.shape
    label_cnt = data_label.reshape([-1, labels]).sum(axis=0)
    weights = 1.0/label_cnt
    normalized_weights = weights/sum(weights)
    return normalized_weights


# In[4]:


# x_new = [sent1, sent2, sent3, ...]　
# x_new -> x_old_emb [batchsize, seq_len, sent_emb]：#使用seq_info 将sent组装回去
# ---> [batchsize, max_seq_len, sent_emb] # padding 成一个可以计算的batch, 从而可以切分
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

def senti_data2bert_tensors(sent_list, cuda):
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


# In[5]:


def TrainRDMWithSubj(rdm_model, bert, rdm_classifier,
                     transformer, task_embedding, subj_classifier, 
                     subjReader, 
                    tokenizer, t_steps, new_data_len=[], logger=None, cuda=False, 
                        log_dir="SubjRDM"):
    batch_size = 10
    max_gpu_batch = 2 #cannot load a larger batch into the limited memory, but we could  accumulates grads
    subjReader.reset_batchsize(max_gpu_batch)

    assert(batch_size%max_gpu_batch == 0)

    sum_loss = np.zeros(3)
    sum_acc = np.zeros(2)

    t_acc = 0.9
    ret_acc = 0.0

    #  rdm loss graph    
    #-----------------------------------------
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    #------------------------------------------------    

    #  subj loss graph
    #-----------------------------------------------------------
    subj_weights = torch.tensor(
            WeightsForUmbalanced(
                subjReader.label
            ),
            dtype = torch.float32
    )
    subj_loss_fn = nn.CrossEntropyLoss(weight=subj_weights) if not cuda else nn.CrossEntropyLoss(weight=subj_weights.cuda())
    subj_task_id = torch.tensor([1]) if not cuda else torch.tensor([1]).cuda()
    #-------------------------------------------------------------    

    loss_weight = torch.tensor([0.8, 0.2]) if not cuda else torch.tensor([0.8, 0.2]).cuda()   
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':5e-7},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-5},
                                {'params': rdm_model.parameters(), 'lr': 5e-5},
                                {'params': task_embedding.parameters(), 'lr':1e-6},
                                {'params': transformer.parameters(), 'lr': 1e-6},
                                {'params': subj_classifier.parameters(), 'lr': 1e-6}
                             ]
    )

    writer = SummaryWriter(log_dir)
    acc_tmp = np.zeros([2, int(batch_size/max_gpu_batch)])
    loss_tmp = np.zeros([3, int(batch_size/max_gpu_batch)])

    for step in range(t_steps):
        optim.zero_grad()
        for j in range(int(batch_size/max_gpu_batch)):
            # RDM loss 用的是bert的pooled emb，　其它模型用的是cls emb

            #----------------RDM loss computation--------------------------
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
            loss_back = loss.mean()*loss_weight[0]
            loss_back.backward()
            torch.cuda.empty_cache()
            #----------------------------------------------------------------------

            # ----------------subjective analysis------------------------
            xsj, ysj, lsj = subjReader.sample()
            sent_tensors, sent_mask = senti_data2bert_tensors(xsj, cuda)
            xsj_embs, _ = bert(sent_tensors, attention_mask = sent_mask)
            tensors = xsj_embs + task_embedding(subj_task_id)
            subj_feature = transformer(tensors.transpose(0, 1)).transpose(0, 1)
            cls_feature = subj_feature.max(axis=1)[0]
            subj_scores = subj_cls(cls_feature)
            y_label = torch.tensor(ysj.argmax(axis=1)).cuda() if cuda else torch.tensor(ysj.argmax(axis=1))
            sj_loss = subj_loss_fn(subj_scores, y_label)
            sj_acc, _, _, _, _, _, _ = Count_Accs(y_label, subj_scores.argmax(axis=1))
            sj_loss_back = sj_loss.mean()*loss_weight[1]
            sj_loss_back.backward()
            torch.cuda.empty_cache()
            #-----------------------------------------------------------
            loss_tmp[:, j] = np.array([loss_back+sj_loss_back, loss.mean(), sj_loss.mean()])
            acc_tmp[:, j] = np.array([acc.mean(), sj_acc.mean()])
            torch.cuda.empty_cache()

        optim.step()
        optim.zero_grad()
        writer.add_scalar('Train Loss', loss_tmp[0].mean(), step)
        writer.add_scalar('Train Accuracy', acc_tmp[0].mean(), step)

        sum_acc += acc_tmp.mean(axis=1)
        sum_loss += loss_tmp.mean(axis=1)

        print("%6d %6d|MTL_Loss:%6.8f, rdm_loss/rdm_acc = %6.8f/%6.7f | senti_loss/senti_acc = %6.8f/%6.7f " % (
                                                                                                step, t_steps, loss_tmp[0].mean(),        
                                                                                            loss_tmp[1].mean(), acc_tmp[0].mean(),
                                                                                            loss_tmp[2].mean(), acc_tmp[1].mean()
            )
            )

        if step % 10 == 9:
            sum_loss = sum_loss / 10
            sum_acc = sum_acc / 10
            print("MTL_Loss:%6.8f, rdm_loss/rdm_acc = %6.8f/%6.7f | senti_loss/senti_acc = %6.8f/%6.7f" % (
                                                                                            sum_loss[0],        
                                                                                            sum_loss[1], sum_acc[0],
                                                                                            sum_loss[2], sum_acc[1]
            )
            )
            if step%100 == 99:
                rdm_save_as = './%s/sentiRDMModel_epoch%03d.pkl'% (log_dir, step/100)
                torch.save(
                    {
                        "bert":bert.state_dict(),
                        "transformer":transformer.state_dict(),
                        "task_embedding":task_embedding.state_dict(),
                        "subj_classifier": subj_classifier.state_dict(),
                        "rmdModel":rdm_model.state_dict(),
                        "rdm_classifier": rdm_classifier.state_dict()
                    },
                    rdm_save_as
                )
    #                 rdm_model, bert, sentiModel, rdm_classifier
            sum_acc = 0.0
            sum_loss = 0.0

    print(get_curtime() + " Train df Model End.")
    return ret_acc


# In[6]:


tt = BertTokenizer.from_pretrained("./bertModel/")
bb = BertModel.from_pretrained("./bertModel/")
task_embedding = nn.Embedding(3, 768)
encoder_layer = nn.TransformerEncoderLayer(768, 8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, 1)
subj_cls = nn.Linear(768, 2)
bert = bb.cuda()
transformer = transformer_encoder.cuda()
task_embedding = task_embedding.cuda()
subj_cls = subj_cls.cuda()
rdm_model = RDM_Model(768, 300, 256, 0.2).cuda()
rdm_classifier = nn.Linear(256, 2).cuda()


# In[7]:


subj_file = "./rotten_imdb/subj.data"
obj_file = "./rotten_imdb/obj.data"
tr, dev, te = load_data(subj_file, obj_file)
subj_train_reader = SubjObjReader(tr, 20, tt)
load_data_fast()


# In[8]:


joint_save_as = './MTLTrain/jointModel_epoch015.pkl'
checkpoint = torch.load(joint_save_as)
bert.load_state_dict(checkpoint['bert'])
transformer.load_state_dict(checkpoint['transformer'])
task_embedding.load_state_dict(checkpoint['task_embedding'])
subj_cls.load_state_dict(checkpoint['subj_classifier'])


# In[9]:
if torch.cuda.device_count() > 1:
    # device_ids = [int(device_id) for device_id in sys.argv[1].split(",")]
    device_ids = list( range( len( sys.argv[1].split(",") ) ) )
    bert = nn.DataParallel(bert, device_ids=device_ids)
    transformer = nn.DataParallel(transformer, device_ids=device_ids)

    device_name = "cuda:0"
    device = torch.device(device_name)
    bert.to(device)
    transformer.to(device)


TrainRDMWithSubj(rdm_model, bert, rdm_classifier,
                     transformer, task_embedding, subj_cls, 
                     subj_train_reader, 
                    tt, 20000, new_data_len=[], logger=None, cuda=True, 
                        log_dir="SubjRDM")