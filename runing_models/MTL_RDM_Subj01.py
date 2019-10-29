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
import transformer_utils
from sklearn.metrics import accuracy_score

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

def subj_cls_train(subj_reader, valid_reader, 
                   bert, transformer, task_embedding, subj_classifier, 
                   train_epochs, cuda=False, log_dir="SubjRDM"
                  ):
    subj_weights = torch.tensor(
            WeightsForUmbalanced(
                subj_reader.label
            ),
            dtype = torch.float32
    )
    subj_loss_fn = nn.CrossEntropyLoss(weight=subj_weights) if not cuda else nn.CrossEntropyLoss(weight=subj_weights.cuda())

    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':5e-5},
                                {'params': task_embedding.parameters(), 'lr':5e-5},
                                {'params': transformer.parameters(), 'lr': 5e-5},
                                {'params': subj_classifier.parameters(), 'lr': 5e-5}
                             ]
    )
    losses = np.zeros([10]) 
    accs = np.zeros([10])
    
    writer = SummaryWriter(log_dir)
    
    subj_task_id = torch.tensor([1]).cuda() if cuda else torch.tensor([1])
    best_valid_acc = 0.0
    
    optim.zero_grad()
    for epoch in range(train_epochs):
        step = 0
        for xsj, ysj, lsj in subj_reader.iter():
            sent_tensors, sent_mask = senti_data2bert_tensors(xsj, cuda)
            xsj_embs, _ = bert(sent_tensors, attention_mask = sent_mask)
            tensors = xsj_embs + task_embedding(subj_task_id)
            subj_feature = transformer(tensors, attention_mask = sent_mask)
            cls_feature = subj_feature[0][:, 0]
            subj_scores = subj_cls(cls_feature)
            y_label = torch.tensor(ysj.argmax(axis=1)).cuda() if cuda else torch.tensor(ysj.argmax(axis=1))
            sj_loss = subj_loss_fn(subj_scores, y_label)
            sj_acc = accuracy_score(ysj.argmax(axis=1), subj_scores.cpu().argmax(axis=1))
            optim.zero_grad()
            sj_loss.backward()
            optim.step()
            torch.cuda.empty_cache()
            
            losses[int(step%10)] = sj_loss.cpu()
            accs[int(step%10)] = sj_acc
            
            print("step:%d | loss/acc = %.3f/%.3f"%(step, sj_loss, sj_acc))
            writer.add_scalar('Train Loss', sj_loss.cpu(), step)
            writer.add_scalar('Train Accuracy', sj_acc, step)
            if step %10 == 9:
                print('subjective task: %6d: [%5d/%5d], subj_loss/subj_acc = %6.8f/%6.7f ' % ( step,
                                                                                epoch, train_epochs,
                                                                                losses.mean(), accs.mean(),
                                                                            )
                         )
            step += 1 
        
        with torch.no_grad():
            bs_cnt, bs, l_cnt = valid_reader.label.shape
            preds = []
            losses = np.zeros(bs_cnt)
            it = 0
            for xsj, ysj, lsj in valid_reader.iter():
                sent_tensors, sent_mask = senti_data2bert_tensors(xsj, cuda)
                xsj_embs, _ = bert(sent_tensors, attention_mask = sent_mask)
                tensors = xsj_embs + task_embedding(subj_task_id)
                subj_feature = transformer(tensors, attention_mask = sent_mask)
                cls_feature = subj_feature[0][:, 0]
                subj_scores = subj_cls(cls_feature)
                y_label = torch.tensor(ysj.argmax(axis=1)).cuda() if cuda else torch.tensor(ysj.argmax(axis=1))
                sj_loss = subj_loss_fn(subj_scores, y_label)
                sj_acc = accuracy_score(ysj.argmax(axis=1), subj_scores.cpu().argmax(axis=1))
                losses[it] = sj_loss
                preds.append(subj_scores)
                torch.cuda.empty_cache()
                it += 1
            val_preds = torch.cat(preds).cpu()
            val_acc = accuracy_score(valid_reader.label.reshape(bs_cnt*bs, l_cnt).argmax(axis=1), val_preds.argmax(axis=1))
            val_loss = losses.mean()
            writer.add_scalar('valid Loss', val_loss, epoch)
            writer.add_scalar('valid Accuracy', val_acc, epoch)
            print("valid loss/acc: %.6f/%.6f:"%(val_loss, val_acc))

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            print("best_valid_acc:", best_valid_acc)
            subj_save_as = './%s/subj_best_Model.pkl'% (log_dir)
            torch.save(
                {
                    "bert":bert.state_dict(),
                    "transformer":transformer.state_dict(),
                    "task_embedding":task_embedding.state_dict(),
                    "subj_classifier": subj_classifier.state_dict(),
                },
                subj_save_as
            )


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

    if os.path.exists("./%s/cached_ssq.pkl"%log_dir):
    #load the cached ssq
        with open("./%s/cached_ssq.pkl"%log_dir, 'rb') as handle:
            ssq = pickle.load(handle)    
    else:
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
                print("batch %d"%i)
                if len(ssq) > 0:
                    ssq.extend([rdm_classifier(h) for h in rdm_hiddens])
                else:
                    ssq = [rdm_classifier(h) for h in rdm_hiddens]
        # cache ssq for development
        with open('./%s/cached_ssq.pkl'%log_dir, 'wb') as handle:
            pickle.dump(ssq, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(get_curtime() + " Now Start RL training ...")

    counter = 0
    sum_rw = 0.0 # sum of rewards
    data_len = get_data_len()

    while True:
        if counter > FLAGS.OBSERVE:
            sum_rw += rw.mean()
            if counter % 200 == 0:
                sum_rw = float(sum_rw) / 200
                print( get_curtime() + " Step: " + str(counter) 
                       + " REWARD IS " + str(sum_rw) 
                     )
                if sum_rw > t_rw:
                    print("Retch The Target Reward")
                    break
                if counter > t_steps:
                    print("Retch The Target Steps")
                    break
                sum_rw = 0.0
            s_state, s_x, s_isStop, s_rw = get_RL_Train_batch(D, FLAGS, cuda)
            sent_tensors, attn_mask = senti_data2bert_tensors(input_x, cuda)
            _, x_emb = bert(sent_tensors, attention_mask=attn_mask)
            x_emb = x_emb.unsqueeze(-2)
            stopScore, isStop, rl_new_state = cm_model(rdm_model, x_emb, s_state)
            out_action = (stopScore*s_isStop).sum(axis=1)
            rl_cost = torch.mean((s_rw - out_action)*(s_rw - out_action))
            rl_cost.backward()
            torch.cuda.empty_cache()

            rl_optim.step()
    #             rdm_optim.step() #后期要尝试一下，是否要同时训练整个模型
            writer.add_scalar('RL Cost', rl_cost, counter - FLAGS.OBSERVE)
            writer.add_scalar('RL Reward', rw.mean(), counter - FLAGS.OBSERVE)

        input_x, input_y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, FLAGS, tokenizer=tokenizer)

        with torch.no_grad():
            sent_tensors, attn_mask = senti_data2bert_tensors(input_x, cuda)
            _, x_emb = bert(sent_tensors, attention_mask=attn_mask)
            x_emb = x_emb.unsqueeze(-2)
            batchsize, max_sent_len, emb_dim = x_emb.shape
            mss, isStop, mNewState = cm_model(rdm_model, x_emb, state)

        for j in range(FLAGS.batch_size):
            if random.random() < FLAGS.random_rate:
                isStop[j] = int(torch.rand(2).argmax())
            if seq_states[j] == data_len[ids[j]]:
                isStop[j] = 1

        # eval
        rw = get_reward(isStop, mss, ssq, ids, seq_states)
        for j in range(FLAGS.batch_size):
            D.append((state[0][j], input_x[j], isStop[j], rw[j]))
            if len(D) > FLAGS.max_memory:
                D.popleft()

        state = mNewState
        for j in range(FLAGS.batch_size):
            if isStop[j] == 1:
                state[0][j].fill_(0)
        counter += 1


def TrainRDMWithSubj(rdm_model, bert, rdm_classifier,
                     transformer, task_embedding, subj_classifier, 
                     subjReader, 
                    tokenizer, t_steps, new_data_len=[], logger=None, cuda=False, 
                        log_dir="SubjRDM"):
    batch_size = 20
    max_gpu_batch = 5 #cannot load a larger batch into the limited memory, but we could  accumulates grads
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

    loss_weight = torch.tensor([0.99, 0.01]) if not cuda else torch.tensor([0.99, 0.01]).cuda()   
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':5e-5},
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

    best_valid_acc = 0.0
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
            subj_feature = transformer(tensors, attention_mask = sent_mask)
            cls_feature = subj_feature[0][:, 0]
            subj_scores = subj_cls(cls_feature)
            y_label = torch.tensor(ysj.argmax(axis=1)).cuda() if cuda else torch.tensor(ysj.argmax(axis=1))
            sj_loss = subj_loss_fn(subj_scores, y_label)
            sj_acc = accuracy_score(ysj.argmax(axis=1), subj_scores.cpu().argmax(axis=1))
            sj_loss_back = sj_loss*loss_weight[1]
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
        writer.add_scalar('Subj Loss', loss_tmp[1].mean(), step)
        writer.add_scalar('Subj Accuracy', acc_tmp[1].mean(), step)


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
                valid_acc = accuracy_on_valid_data(bert, rdm_model, rdm_classifier, [], tokenizer)
                writer.add_scalar('rdm valid acc', valid_acc, step/100)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    print("best valid_acc:", best_valid_acc)
                    rdm_save_as = './%s/SubjRDM_best.pkl'% (log_dir)
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


tt = BertTokenizer.from_pretrained("./bertModel/")
bb = BertModel.from_pretrained("./bertModel/")
task_embedding = nn.Embedding(3, 768)

trans_conf = adict({
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 2,
  "num_labels": 2,
  "output_attentions": False,
  "output_hidden_states": False,
  "torchscript": False
})
BertEncoder = transformer_utils.BertEncoder
transformer = BertEncoder(trans_conf)

subj_cls = nn.Linear(768, 2)
bert = bb.cuda()
transformer = transformer.cuda()
task_embedding = task_embedding.cuda()
subj_cls = subj_cls.cuda()
rdm_model = RDM_Model(768, 300, 256, 0.2).cuda()
rdm_classifier = nn.Linear(256, 2).cuda()
cm_model = CM_Model(300, 256, 2).cuda()

# In[7]:

subj_file = "./rotten_imdb/subj.data"
obj_file = "./rotten_imdb/obj.data"
tr, dev, te = load_data(subj_file, obj_file)

subj_train_reader = SubjObjReader(tr, 20, tt)
subj_valid_reader = SubjObjReader(dev, 20, tt)
subj_test_reader =  SubjObjReader(te, 20, tt)


load_data_fast()


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

# In[8]:

log_dir = "SubjRDM01"

joint_save_as = './%s/subj_best_Model.pkl'%log_dir
if os.path.exists(joint_save_as):
    checkpoint = torch.load(joint_save_as)
    bert.load_state_dict(checkpoint['bert'])
    transformer.load_state_dict(checkpoint['transformer'])
    task_embedding.load_state_dict(checkpoint['task_embedding'])
    subj_cls.load_state_dict(checkpoint['subj_classifier'])
else:
    subj_cls_train(subj_train_reader, subj_valid_reader, 
               bert, transformer, task_embedding, subj_cls, 
               2, cuda=True, log_dir=log_dir
              )

rdm_save_as = './%s/SubjRDM_best.pkl'%log_dir
if os.path.exists(rdm_save_as):
<<<<<<< HEAD
    checkpoint = torch.load(joint_save_as)
=======
    checkpoint = torch.load(rdm_save_as)
>>>>>>> ba55d7fadfffe427b74f1f4a9c7951224f398387
    bert.load_state_dict(checkpoint['bert'])
    transformer.load_state_dict(checkpoint['transformer'])
    task_embedding.load_state_dict(checkpoint['task_embedding'])
    subj_cls.load_state_dict(checkpoint['subj_classifier'])
    rdm_model.load_state_dict(checkpoint['rmdModel'])
    rdm_classifier.load_state_dict(checkpoint["rdm_classifier"])
else:
    TrainRDMWithSubj(rdm_model, bert, rdm_classifier,
                         transformer, task_embedding, subj_cls, 
                         subj_train_reader, 
                        tt, 2000, new_data_len=[], logger=None, cuda=True, 
                            log_dir= log_dir)

print("train rdm model with subj task is completed!")

# for i in range(20):
#     if i==0:
#         TrainCMModel(bert, rdm_model, rdm_classifier, cm_model, tt, i, 0.5, 50000, "SubjERD/", None, FLAGS, cuda=True)
#     else:
#         TrainCMModel(bert, rdm_model, rdm_classifier, cm_model, tt, i, 0.5, 5000, "SubjERD/", None, FLAGS, cuda=True)
#     erd_save_as = './SubjERD/erdModel_epoch%03d.pkl'% (i)
#     torch.save(
#         {
#             "bert":bert.state_dict(),
#             "rmdModel":rdm_model.state_dict(),
#             "rdm_classifier": rdm_classifier.state_dict(),
#             "cm_model":cm_model.state_dict()
#         },
#         erd_save_as
#     )
#     s2vec = Sent2Vec_Generater(tt, bert, cuda=True)
#     new_len = get_new_len(s2vec, rdm_model, cm_model, FLAGS, cuda=True)
#     print("after new len:")
#     TrainRDMWithSubj(rdm_model, bert, rdm_classifier,
#                      transformer, task_embedding, subj_cls, 
#                      subj_train_reader, 
#                     tt, 1000, new_data_len=[], logger=None, cuda=True, 
#                         log_dir="SubjERD_%d"%i)



