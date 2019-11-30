import sys
import random
import torch
import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import pickle
import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from dataUtilsV0 import *
import json


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


def TrainCMModel_V2(sent_pooler, rdm_model, rdm_classifier, cm_model, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=True):
    batch_size = 20
    t_acc = 0.9
    gamma = 1.0
    lambda1 = -1.0
    lambda2 = 0.0 #regularizer
    sum_loss = 0.0
    sum_acc = 0.0
    t_acc = 0.9
    ret_acc = 0.0
    init_states = torch.zeros([1, batch_size, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adagrad([
                                {'params': sent_pooler.parameters(), 'lr': 2e-3},
                                {'params': rdm_model.parameters(), 'lr': 2e-3},
                                {'params': rdm_classifier.parameters(), 'lr': 2e-3},
                                {'params': cm_model.parameters(), 'lr':2e-3}
                             ]
    )
    
    writer = SummaryWriter(log_dir, filename_suffix="_ERD_CM_stage_%3d"%stage)
    best_valid_acc = 0.0

    rw_arr = np.zeros(10)
    len_arr = np.zeros(10)
    for step in range(t_steps):
            x, x_len, y = get_df_batch(step*batch_size, batch_size)        
        # for rep in range(100):
            optim.zero_grad()
            seq = sent_pooler(x)
            rdm_hiddens = rdm_model(seq)
            # rdm_hiddens, rdm_out, rdm_cell = rdm_model(seq, x_len.tolist())
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
            
            batchsize, max_seq_len, _ = rdm_hiddens.shape
            stopScore, isStop = cm_model(rdm_hiddens.reshape(-1, 256))
            isStop = isStop.reshape([batchsize, max_seq_len, -1])
            stopScore = stopScore.reshape([batchsize, max_seq_len, -1]).softmax(axis=-1)         
            E_rw = torch.zeros(len(x_len)).cuda()
            prob = torch.ones(len(x_len)).cuda()
            sum_len = 0.0
            for j in range(len(x_len)):
                sum_rw = 0.0
                start = 0
                end = -1
                # start = random.randint(0, x_len[j]-1)
                for t in range(start, x_len[j]):
                    rnd = random.random()
                    # pdb.set_trace()
                    if rnd > stopScore[j][t][1]:
                        rw = -2.0/x_len[j]
                        sum_rw += pow(gamma, t)*rw
                        prob[j] *= stopScore[j][t][0]            
                    else:
                        if y[j][1] == 1:
                            rw = 1.0
                        else:
                            rw = -1.0
                        sum_rw += pow(gamma, t)*rw
                        prob[j] *= stopScore[j][t][1]
                        rnd2 = random.random()
                        if  rnd2 > 0.6:
                            prob[j] *= 0.4
                        else:
                            prob[j] *= 0.6
                            if end == -1:
                                end = t+1
                
                if end == -1:
                    end = x_len[j]
                # sum_len += (t-start+1)*1.0/x_len[j]
                sum_len += (end-start)
                E_rw[j] = prob[j]*sum_rw
            # pdb.set_trace()
            rw_arr[int(step%10)] = E_rw.mean()
            len_arr[int(step%10)] = sum_len*1.0/batch_size

            loss_back = lambda1*E_rw.mean() + lambda2*loss
            loss_back.backward()
            optim.step()      
            if step%10 == 0:  
                # print('%3d | %d , train_loss/Expected Reward = %6.8f/%6.7f,  RDM_Loss/RDM Accuracy = %6.8f/%6.7f, mean_len = %2.3f'             % (step, t_steps, 
                #         loss_back, E_rw.mean(), loss, acc, sum_len*1.0/batch_size
                #         ))
                print('%3d | %d , train_loss/Expected Reward = %6.8f/%6.7f,  RDM_Loss/RDM Accuracy = %6.8f/%6.7f, mean_len = %2.3f'             % (step, t_steps, 
                        loss_back, rw_arr.mean(), loss, acc, len_arr.mean()
                        ))

            writer.add_scalar('RDM Loss', loss, step)
            writer.add_scalar('RDM Accuracy', acc, step)
            writer.add_scalar('Train Loss', loss_back, step)
            writer.add_scalar('Expected Reward', E_rw.mean(), step)

            torch.cuda.empty_cache()
        
        # print("weight grad:", cm_model.Classifier.weight.grad)
        # print("bias grad:", cm_model.Classifier.bias.grad)
    print(get_curtime() + " Train df Model End.")
    return ret_acc    
