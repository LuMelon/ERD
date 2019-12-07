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
import json
import pdb
from dataUtilsV0 import *
import numpy as np

def TrainCMModel_V0(sent_pooler, rdm_model, rdm_classifier, cm_model, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=True):
    batch_size = FLAGS.batch_size
    t_acc = 0.9
    ids = np.array(range(batch_size), dtype=np.int32)
    seq_states = np.zeros([batch_size], dtype=np.int32)
    isStop = torch.zeros([batch_size], dtype=torch.int32)
    max_id = batch_size
    df_init_states = torch.zeros([1, batch_size, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    writer = SummaryWriter(log_dir, filename_suffix="_ERD_CM_stage_%3d"%stage)
    D = deque()
    ssq = []
    print("in RL the begining")
    rl_optim = torch.optim.Adam([{'params': sent_pooler.parameters(), 'lr': 2e-5},
                                 {'params': rdm_model.parameters(), 'lr': 2e-5},
                                 {'params':cm_model.parameters(), 'lr':1e-3}])
    data_ID = get_data_ID()
    valid_data_len = get_valid_data_len()
    data_len = get_data_len()
    
    if len(data_ID) % batch_size == 0: # the total number of events
        flags = int(len(data_ID) / FLAGS.batch_size)
    else:
        flags = int(len(data_ID) / FLAGS.batch_size) + 1

    for i in range(flags):
        with torch.no_grad():
            x, x_len, y = get_df_batch(i, batch_size)
            seq = sent_pooler(x)
            rdm_hiddens = rdm_model(seq)
            batchsize, _, _ = rdm_hiddens.shape
            print("batch %d"%i)
            if len(ssq) > 0:
                ssq.extend([rdm_classifier(h) for h in rdm_hiddens])
            else:
                ssq = [rdm_classifier(h) for h in rdm_hiddens]
            torch.cuda.empty_cache()

    print(get_curtime() + " Now Start RL training ...")
    counter = 0
    sum_rw = 0.0 # sum of rewards
    
    while True:
    #         if counter > FLAGS.OBSERVE:
        if counter > FLAGS.OBSERVE:
            sum_rw += rw.mean()
            if counter % 200 == 0:
                sum_rw = sum_rw / 2000
                print(get_curtime() + " Step: " + str(counter-FLAGS.OBSERVE) + " REWARD IS " + str(sum_rw))
                if counter > t_steps:
                    print("Retch The Target Steps")
                    break
                sum_rw = 0.0
            s_state, s_x, s_isStop, s_rw = get_RL_Train_batch(D)
            word_tensors = torch.tensor(s_x)
            batchsize, max_sent_len, emb_dim = word_tensors.shape
            sent_tensor = sent_pooler.linear(word_tensors.reshape([-1, emb_dim]).cuda()).reshape([batchsize, max_sent_len, emb_dim]).max(axis=1)[0].unsqueeze(1)
            df_outs, df_last_state = rdm_model.gru_model(sent_tensor, s_state.unsqueeze(0).cuda())
            batchsize, _, hidden_dim = df_outs.shape
            stopScore, isStop = cm_model(df_outs.reshape([-1, hidden_dim]))
            out_action = (stopScore*s_isStop.cuda()).sum(axis=1)
            rl_cost = torch.pow(s_rw.cuda() - out_action, 2).mean()
            rl_optim.zero_grad()
            rl_cost.backward()
            torch.cuda.empty_cache()
            rl_optim.step()
            # print("RL Cost:", rl_cost)
            writer.add_scalar('RL Cost', rl_cost, counter - FLAGS.OBSERVE)
            if (counter - FLAGS.OBSERVE)%100 == 0:
                print("*** %6d|%6d *** RL Cost:%8.6f"%(counter, t_steps, rl_cost))
                valid_new_len = get_new_len_on_valid_data(sent_pooler, rdm_model, cm_model, FLAGS, cuda=True)
                print("diff len:", np.array(valid_data_len)-np.array(valid_new_len))

        x, y, ids, seq_states, max_id = get_rl_batch_0(ids, seq_states, isStop, max_id, 0)
        for j in range(FLAGS.batch_size):
            if seq_states[j] == 1:
                df_init_states[0][j].fill_(0.0)
                
        with torch.no_grad():
            word_tensors = torch.tensor(x)
            batchsize, max_sent_len, emb_dim = word_tensors.shape
            sent_tensor = sent_pooler.linear(word_tensors.reshape([-1, emb_dim]).cuda()).reshape([batchsize, max_sent_len, emb_dim]).max(axis=1)[0].unsqueeze(1)
            df_outs, df_last_state = rdm_model.gru_model(sent_tensor, df_init_states)
            batchsize, _, hidden_dim = df_outs.shape
            stopScore, isStop = cm_model(df_outs.reshape([-1, hidden_dim]))
            
        for j in range(batch_size):
            if random.random() < FLAGS.random_rate:
                isStop[j] = torch.randn(2).argmax()
            if seq_states[j] == data_len[ids[j]]:
                isStop[j] = 1
        rw, Q_val = get_reward_0(isStop, stopScore, ssq, ids, seq_states)
        for j in range(FLAGS.batch_size):
            D.append((df_init_states[0][j], x[j], isStop[j], rw[j]))
            if len(D) > FLAGS.max_memory:
                D.popleft()
        df_init_states = df_last_state
        counter += 1


def TrainCMModel_V1(sent_pooler, rdm_model, rdm_classifier, cm_model, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=True):
    batch_size = FLAGS.batch_size
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
    batch_size = FLAGS.batch_size
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
                                {'params': sent_pooler.parameters(), 'lr': 1e-3},
                                {'params': rdm_model.parameters(), 'lr': 1e-3},
                                {'params': rdm_classifier.parameters(), 'lr': 1e-3},
                                {'params': cm_model.parameters(), 'lr':1e-3}
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





def TrainCMModel_V3(sent_pooler, rdm_model, rdm_classifier, cm_model, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=True):
    batch_size = FLAGS.batch_size
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
    loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='mean')
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

            with torch.no_grad():
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
            
            batchsize, max_seq_len, hidden_dim = rdm_hiddens.shape
            stopScore, isStop = cm_model(rdm_hiddens.reshape(-1, 256))
            isStop = isStop.reshape([batchsize, max_seq_len, -1])
            stopProb = stopScore.reshape([batchsize, max_seq_len, -1]).softmax(axis=-1)         

            sum_rw = torch.zeros(len(x_len)).cuda()
            prob = torch.ones(len(x_len)).cuda()
            sum_len = 0.0

            preds_list = []
            label_list = []

            preds = rdm_classifier(rdm_hiddens.reshape(-1, hidden_dim)).reshape(batchsize, max_seq_len, -1)
            for j in range(len(x_len)):
                start = random.randint(0, x_len[j]-1)
                # delay_punish = np.log( np.arange(1, x_len[j]+1)*1.0/(x_len[j]+1) )
                delay_punish = -0.1*torch.arange(1, x_len[j]+1).cuda()
                for t in range(start, x_len[j]):
                    rnd = random.random()
                    # pdb.set_trace()
                    if rnd > stopProb[j][t][1]:
                        prob[j] *= stopProb[j][t][0]            
                    else:
                        prob[j] *= stopProb[j][t][1]
                        rnd2 = random.random()
                        if  rnd2 < 0.4:
                            break
                label_list.append( torch.tensor(y[j]).repeat(x_len[j]-t, 1).cuda() )
                # pdb.set_trace()
                sum_rw[j] = -1*delay_punish[t]+ loss_fn(preds[j][t:x_len[j]], label_list[-1].argmax(axis=1))
                sum_len += (t-start+1)*1.0/x_len[j]
            if step < 1000:
                E_rw = (prob.detach().cuda()*sum_rw).mean()
                E_rw.backward()
                if step%10 == 0:  
                    print('*****Optimizing RDM***** %3d | %d , train_loss/Expected Reward = %6.8f/%6.7f,  RDM_Loss/RDM Accuracy = %6.8f/%6.7f, mean_len = %2.3f'             % (step, t_steps, 
                            rw_arr.mean(), rw_arr.mean(), loss, acc, len_arr.mean()
                            ))

            elif step < 2000: #偶数个1000轮，训练PG
                E_rw = (prob*sum_rw.detach().cuda()).mean()
                E_rw.backward()
                if step%10 == 0:  
                    print('*****Optimizing Policy***** %3d | %d , train_loss/Expected Reward = %6.8f/%6.7f,  RDM_Loss/RDM Accuracy = %6.8f/%6.7f, mean_len = %2.3f'             % (step, t_steps, 
                            rw_arr.mean(), rw_arr.mean(), loss, acc, len_arr.mean()
                            ))
            else:
                E_rw = (prob.detach().cuda()*sum_rw).mean()
                E_rw.backward(retain_graph=True)    
                E_rw = (prob*sum_rw.detach().cuda()).mean()
                E_rw.backward(retain_graph=False)

                if step%10 == 0:  
                        print('*****Optimizing Policy & RDM***** %3d | %d , train_loss/Expected Reward = %6.8f/%6.7f,  RDM_Loss/RDM Accuracy = %6.8f/%6.7f, mean_len = %2.3f'             % (step, t_steps, 
                                rw_arr.mean(), rw_arr.mean(), loss, acc, len_arr.mean()
                                ))
            optim.step()      

            # pdb.set_trace()
            rw_arr[int(step%10)] = float(E_rw)
            len_arr[int(step%10)] = sum_len*1.0/batch_size

            writer.add_scalar('RDM Loss', loss, step)
            writer.add_scalar('RDM Accuracy', acc, step)
            writer.add_scalar('Train Loss', float(E_rw), step)
            writer.add_scalar('Expected Reward', float(E_rw), step)

            torch.cuda.empty_cache()
        
        # print("weight grad:", cm_model.Classifier.weight.grad)
        # print("bias grad:", cm_model.Classifier.bias.grad)
    print(get_curtime() + " Train df Model End.")
    return ret_acc    


def TrainCMModel_V4(sent_pooler, rdm_model, rdm_classifier, cm_model, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=True):
    batch_size = FLAGS.batch_size
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
    loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='mean')
    optim = torch.optim.Adagrad([
                                {'params': sent_pooler.parameters(), 'lr': 2e-3},
                                {'params': rdm_model.parameters(), 'lr': 2e-3},
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

            batchsize, max_seq_len, hidden_dim = rdm_hiddens.shape
            stopScore, isStop = cm_model(rdm_hiddens.reshape(-1, 256))
            isStop = isStop.reshape([batchsize, max_seq_len, -1])
            stopProb = stopScore.reshape([batchsize, max_seq_len, -1]).softmax(axis=-1)         

            with torch.no_grad():
                preds_score = torch.stack([stopProb[jj][x_len[jj]-1] for jj in range(len(x_len))])
                rdm_preds = preds_score.argmax(axis=1)
                y_label = torch.tensor(y).argmax(axis=1).cuda() if cuda else torch.tensor(y).argmax(axis=1)
                acc, _, _, _, _, _, _ = Count_Accs(y_label, rdm_preds)
                loss = loss_fn(preds_score, y_label)

            sum_rw = torch.zeros(len(x_len)).cuda()
            prob = torch.ones(len(x_len)).cuda()
            sum_len = 0.0

            preds_list = []
            label_list = []
            for j in range(len(x_len)):
                start = random.randint(0, x_len[j]-1)
                # delay_punish = np.log( np.arange(1, x_len[j]+1)*1.0/(x_len[j]+1) )
                delay_punish = -0.1*torch.arange(1, x_len[j]+1).cuda()
                coeff = -1 if y[j][0] == 1 else 1
                for t in range(start, x_len[j]):
                    if stopProb[j][t][0] > 0.5:
                        prob[j] *= stopProb[j][t][0]            
                    else:
                        prob[j] *= stopProb[j][t][1]
                        rnd2 = random.random()
                        if  rnd2 < 0.4:
                            break
                label_list.append( torch.tensor(y[j]).repeat(x_len[j]-t, 1).cuda() )
                sum_rw[j] += -1*delay_punish[t] - (stopProb[j][t:x_len[j], 1] - 0.5).sum()*coeff
                sum_len += (t-start+1)*1.0/x_len[j]
            E_rw = (prob.detach().cuda()*sum_rw).mean()
            E_rw.backward(retain_graph=True)
            E_rw = (prob*sum_rw.detach().cuda()).mean()
            E_rw.backward(retain_graph=False)
            optim.step()      

            # pdb.set_trace()
            rw_arr[int(step%10)] = float(E_rw)
            len_arr[int(step%10)] = sum_len*1.0/batch_size

            
            if step%10 == 0:  
                # print('%3d | %d , train_loss/Expected Reward = %6.8f/%6.7f,  RDM_Loss/RDM Accuracy = %6.8f/%6.7f, mean_len = %2.3f'             % (step, t_steps, 
                #         loss_back, E_rw.mean(), loss, acc, sum_len*1.0/batch_size
                #         ))
                print('%3d | %d , train_loss/Expected Reward = %6.8f/%6.7f,  RDM_Loss/RDM Accuracy = %6.8f/%6.7f, mean_len = %2.3f'             % (step, t_steps, 
                        rw_arr.mean(), rw_arr.mean(), loss, acc, len_arr.mean()
                        ))

            writer.add_scalar('RDM Loss', loss, step)
            writer.add_scalar('RDM Accuracy', acc, step)
            writer.add_scalar('Train Loss', float(E_rw), step)
            writer.add_scalar('Expected Reward', float(E_rw), step)

            torch.cuda.empty_cache()
        
        # print("weight grad:", cm_model.Classifier.weight.grad)
        # print("bias grad:", cm_model.Classifier.bias.grad)
    print(get_curtime() + " Train df Model End.")
    return ret_acc    

