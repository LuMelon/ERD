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
import json
from torch import nn
import torch
from pytorch_transformers import *
import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import tsentiLoader


# In[2]:


from emotionLoader import *
from SubjObjLoader import *
import numpy as np


# ### BERT RDM CM 模型代码
# > 需要改动的地方
# > - 损失函数要从训练函数中拆分出来，方便后面的联合训练
# > 

# In[3]:


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
        
    def forward(self, x_emb, x_len, init_states): 
        """
        input_x: [batchsize, max_seq_len, sentence_embedding_dim] 
        x_emb: [batchsize, max_seq_len, 1, embedding_dim]
        x_len: [batchsize]
        init_states: [batchsize, hidden_dim]
        """
        batchsize, max_seq_len, _ , emb_dim = x_emb.shape
#         pool_feature = self.PoolLayer(x_emb)
#         sent_feature = sentiModel( 
#                 x_emb.reshape(
#                     [-1, max_sent_len, emb_dim]
#                 ) 
#             ).reshape(
#                 [batchsize, max_seq_len, -1]
#             )
#         pooled_input_x_dp = self.DropLayer(input_x)
        pool_feature = x_emb.reshape(
                [-1, max_seq_len, emb_dim]
        )
        df_outputs, df_last_state = self.gru_model(pool_feature, init_states)
        hidden_outs = [df_outputs[i][:x_len[i]] for i in range(batchsize)]
        final_outs = [df_outputs[i][x_len[i]-1] for i in range(batchsize)]
        return hidden_outs, final_outs


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
        
    def forward(self, rdm_model, s_model, rl_input, rl_state):
        """
        rl_input: [batchsize, max_word_num, sentence_embedding_dim]
        rl_state: [1, batchsize, hidden_dim]
        """
        assert(rl_input.ndim==3)
        batchsize, max_word_num, embedding_dim = rl_input.shape
#         assert(embedding_dim==self.embedding_dim)
        sentence = s_model(rl_input).reshape(batch_size, 1, self.sentence_embedding_dim)
#         pooled_rl_input = self.PoolLayer(
#             rl_input.reshape(
#                 [-1, 1, max_word_num, self.embedding_dim]
#             )
#         ).reshape([-1, 1, self.hidden_dim])
        
#         print("sentence:", sentence.shape)
#         print("rl_state:", rl_state.shape)
        rl_output, rl_new_state = rdm_model.gru_model(
                                            sentence, 
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


# In[3]:


def layer2seq(bert, layer, cuda=False):
    if cuda:
        outs = [bert( torch.tensor([input_]).cuda())
                for input_ in layer]   
    else: 
        outs = [bert( torch.tensor([input_]))
                    for input_ in layer]
    states = [item[1] for item in outs]
    return rnn_utils.pad_sequence(states, batch_first=True)

def Word_ids2SeqStates(word_ids, bert, ndim, cuda=False):
    assert(ndim == 3)
    if cuda:
        embedding = [layer2seq(bert, layer, cuda) for layer in word_ids]
    else:
        embedding = [layer2seq(bert, layer) for layer in word_ids]
    return padding_sequence(embedding)


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


# In[5]:
def rdm_loss(x, y, x_len, bert, rdm_model, rdm_classifier, loss_fn):
    x_emb = Word_ids2SeqStates(x, bert, 3, cuda=True) 
    batchsize, max_seq_len, max_sent_len, emb_dim = x_emb.shape
    init_states = torch.zeros([1, batchsize, rdm_model.hidden_dim]).cuda()
    rdm_hiddens, rdm_outs = rdm_model(x_emb, x_len, init_states)
    rdm_scores = rdm_classifier(
        torch.cat(
            rdm_outs # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
        ).reshape(
            [-1, rdm_model.hidden_dim]
        )
    )
    rdm_preds = rdm_scores.argmax(axis=1)
    y_label = y.argmax(axis=1)
    acc, _, _, _, _, _, _ = Count_Accs(y_label, rdm_preds)
    loss = loss_fn(rdm_scores, torch.tensor(y_label).cuda())
    return loss, acc


def TrainRDMModel(rdm_model, bert, rdm_classifier, 
                    tokenizer, t_steps, new_data_len=[], logger=None, 
                        log_dir="RDMBertTrain"):
    batch_size = 20 
    max_gpu_batch = 5 #cannot load a larger batch into the limited memory, but we could  accumulates grads
    assert(batch_size%max_gpu_batch == 0)
    sum_loss = 0.0
    sum_acc = 0.0
    t_acc = 0.9
    ret_acc = 0.0
    init_states = torch.zeros([1, 5, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':5e-5},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-5},
                                {'params': rdm_model.parameters(), 'lr': 5e-5}
                             ]
    )
    
    writer = SummaryWriter(log_dir)
    acc_l = np.zeros(int(batch_size/max_gpu_batch))
    loss_l = np.zeros(int(batch_size/max_gpu_batch))
    for step in range(t_steps):
        optim.zero_grad()
        for j in range(int(batch_size/max_gpu_batch)):
            if len(new_data_len) > 0:
                x, x_len, y = get_df_batch(step, max_gpu_batch, new_data_len, tokenizer=tokenizer)
            else:
                x, x_len, y = get_df_batch(step, max_gpu_batch, tokenizer=tokenizer)
                
            loss, acc = rdm_loss(x, y, bert, rdm_model, rdm_classifier, loss_fn)
            loss.backward()
            loss_l[j] = loss
            acc_l[j] = acc
            
        optim.step()        
        writer.add_scalar('Train Loss', loss_l.mean(), step)
        writer.add_scalar('Train Accuracy', acc_l.mean(), step)

        sum_loss += loss_l.mean()
        sum_acc += acc_l.mean()
        

        if step % 10 == 9:
            sum_loss = sum_loss / 10
            sum_acc = sum_acc / 10
            print('%3d | %d , train_loss/accuracy = %6.8f/%6.7f'             % (step, t_steps, 
                sum_loss, sum_acc,
                ))
            logger.info('%3d | %d , train_loss/accuracy = %6.8f/%6.7f'             % (step, t_steps, 
                sum_loss, sum_acc,
                ))
            if step%500 == 499:
                rdm_save_as = './%s/rdmModel_epoch%03d.pkl'                                    % (log_dir, step/500, sum_acc)
                torch.save(
                    {
                        "bert":bert.state_dict(),
                        "sentiModel":sentiModel.state_dict(),
                        "rmdModel":rdm_model.state_dict(),
                        "rdm_classifier": rdm_classifierdm.state_dict()
                    },
                    rdm_save_as
                )
#                 rdm_model, bert, sentiModel, rdm_classifier
            if sum_acc > t_acc:
                break
            sum_acc = 0.0
            sum_loss = 0.0

    print(get_curtime() + " Train df Model End.")
    logger.info(get_curtime() + " Train df Model End.")
    return ret_acc


# In[6]:


def TrainCMModel(bert, rdm_model, rdm_classifier, cm_model, tokenizer, log_dir, logger, FLAGS):
    batch_size = 20
    t_acc = 0.9
    ids = np.array(range(batch_size), dtype=np.int32)
    seq_states = np.zeros([batch_size], dtype=np.int32)
    isStop = np.zeros([batch_size], dtype=np.int32)
    max_id = batch_size
    df_init_states = torch.zeros([1, batch_size, FLAGS.hidden_dim], dtype=torch.float32).cuda()
    state = df_init_states
    D = deque()
    ssq = []
    print("in RL the begining")
    rdm_optim = torch.optim.Adagrad([
                            {'params': bert.parameters(), 'lr':1e-3},
    #                                 {'params': rdm_classifier.parameters(), 'lr': 5e-2},
                            {'params': rdm_model.parameters(), 'lr': 5e-2},
                            {'params': sentiModel.parameters(), 'lr': 1e-2}
                         ],
                            weight_decay = 0.2
    )
    rl_optim = torch.optim.Adam([{'params':cm_model.parameters(), 'lr':1e-3}])
    # get_new_len(sess, mm)
    data_ID = get_data_ID()

    if len(data_ID) % batch_size == 0: # the total number of events
        flags = int(len(data_ID) / FLAGS.batch_size)
    else:
        flags = int(len(data_ID) / FLAGS.batch_size) + 1
    for i in range(flags):
        with torch.no_grad():
            x, x_len, y = get_df_batch(i, batch_size, tokenizer=tokenizer)
            x_emb = Word_ids2SeqStates(x, bert, 3, cuda=True) 
            batchsize, max_seq_len, max_sent_len,                                     emb_dim = x_emb.shape
            sent_feature = sentiModel( 
                x_emb.reshape(
                    [-1, max_sent_len, emb_dim]
                ) 
            ).reshape(
                [batchsize, max_seq_len, -1]
            )
            rdm_hiddens, rdm_outs = rdm_model(sent_feature, x_len, df_init_states)
        #         t_ssq = sess.run(rdm_train.out_seq, feed_dic)# t_ssq = [batchsize, max_seq, scores]
            print("batch %d"%i)
            if len(ssq) > 0:
                ssq.extend([rdm_classifier(h) for h in rdm_hiddens])
            else:
                ssq = [rdm_classifier(h) for h in rdm_hiddens]

    print(get_curtime() + " Now Start RL training ...")
    counter = 0
    sum_rw = 0.0 # sum of rewards

    data_len = get_data_len()

    while True:
        if counter > FLAGS.OBSERVE:
            sum_rw += np.mean(rw)
            if counter % 200 == 0:
                sum_rw = sum_rw / 2000
                print( get_curtime() + " Step: " + str(step) 
                       + " REWARD IS " + str(sum_rw) 
                     )
                logger.info( get_curtime() + 
                             " Step: " + str(step) + 
                            " REWARD IS " + str(sum_rw)
                           )
                if sum_rw > t_rw:
                    print("Retch The Target Reward")
                    logger.info("Retch The Target Reward")
                    break
                if counter > t_steps:
                    print("Retch The Target Steps")
                    logger.info("Retch The Target Steps")
                    break
                sum_rw = 0.0
            s_state, s_x, s_isStop, s_rw = get_RL_Train_batch(D, FLAGS)
            stopScore, isStop, rl_new_state = cm_model(rdm_model, sentiModel, s_x, s_state)
            out_action = (stopScore*s_isStop).sum(axis=1)
            rl_cost = torch.mean((s_rw - out_action)*(s_rw - out_action))
            rl_cost.backward()
            rl_optim.step()

        input_x, input_y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, FLAGS, tokenizer=tokenizer)
        with torch.no_grad():
            x_emb = layer2seq(bert, input_x, cuda=True)
            batchsize, max_sent_len, emb_dim = x_emb.shape
            mss, isStop, mNewState = cm_model(rdm_model, sentiModel, x_emb, state)

        for j in range(FLAGS.batch_size):
            if random.random() < FLAGS.random_rate:
    #             isStop[j] = np.argmax(np.random.rand(2))
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
                # init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)
                # feed_dic = {rl_model.init_states: init_states}
                # state[j] = sess.run(rl_model.df_state, feed_dic)
    #             state[j] = np.zeros([FLAGS.hidden_dim], dtype=np.float32)
                state[0][j].fill_(0)
        counter += 1.0


# ## 联合学习部分
# 

# ### 工具函数

# In[4]:


def words2tensor(bert, layer, cuda=False):
    if cuda:
        outs = [bert( torch.tensor([input_]).cuda())
                for input_ in layer]   
    else: 
        outs = [bert( torch.tensor([input_]))
                    for input_ in layer]
    states = [item[0][0] for item in outs]
    return rnn_utils.pad_sequence(states, batch_first=True)


# 
# > normalized the data for unbalanced learning:
# ```python
# weights = WeightsForUmbalanced(emoReader.label)
# weights
# ```

# In[5]:


def WeightsForUmbalanced(data_label):
    _, _, labels = data_label.shape
    label_cnt = data_label.reshape([-1, labels]).sum(axis=0)
    weights = 1.0/label_cnt
    normalized_weights = weights/sum(weights)
    return normalized_weights


# ### 求各个任务的损失值

# In[6]:


def senti_loss( xst, yst, lst, 
                bert, transformer, task_emb,
                senti_classifier, senti_loss_fn,
                cuda=False
               ):
    tensors = words2tensor(bert, xst, cuda) + task_emb
    y_label = torch.tensor(yst.argmax(axis=1)).cuda() if cuda else torch.tensor(yst.argmax(axis=1))
    senti_feature = transformer(tensors.transpose(0, 1)).transpose(0, 1)
    cls_feature = senti_feature.max(axis=1)[0]
    senti_scores = senti_classifier(cls_feature)
    loss = senti_loss_fn(senti_scores, y_label)
    acc, _, _, _, _, _, _ = Count_Accs(y_label, senti_scores.argmax(axis=1))
    return loss, acc


# In[7]:


def subj_loss(  xsj, ysj, lsj,
                bert, transformer, task_emb,
                subj_classifier, subj_loss_fn,
                cuda=False
               ):
    tensors = words2tensor(bert, xsj, cuda) + task_emb
    y_label = torch.tensor(ysj.argmax(axis=1)) if not cuda else torch.tensor(ysj.argmax(axis=1)).cuda()
    subj_feature = transformer(tensors.transpose(0, 1)).transpose(0, 1)
    cls_feature = subj_feature.max(axis=1)[0]
    subj_scores = subj_classifier(cls_feature)
    loss = subj_loss_fn(subj_scores, y_label)
    acc, _, _, _, _, _, _ = Count_Accs(y_label, subj_scores.argmax(axis=1))
    return loss, acc


# In[8]:


def emotion_loss(   xem, yem, lem,
                    bert, transformer, task_emb,
                    emotion_classifier, emo_loss_fn, 
                    cuda
                ):
    tensors = words2tensor(bert, xem, cuda) + task_emb
    y_label = torch.tensor(yem.argmax(axis=1)) if not cuda else torch.tensor(yem.argmax(axis=1)).cuda()
    emo_feature = transformer(tensors.transpose(0, 1)).transpose(0, 1)
    cls_feature = emo_feature.max(axis=1)[0]
    emo_scores = emotion_classifier(cls_feature)
    loss = emo_loss_fn(emo_scores, y_label)
    acc, _, _, _, _, _, _ = Count_Accs(y_label, emo_scores.argmax(axis=1))
    return loss, acc


# ### 训练各个任务

# In[9]:


def emo_cls_train(emotion_reader, bert, transformer, 
                  task_embedding, emotion_classifier,
                  train_epochs, emo_loss_fn,
                  cuda=False
                 ):
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':1e-6},
                                {'params': task_embedding.parameters(), 'lr':1e-6},
                                {'params': transformer.parameters(), 'lr': 1e-6},
                                {'params': emotion_classifier.parameters(), 'lr': 1e-6}
                             ]
    )
    losses = np.zeros([10]) 
    accs = np.zeros([10])
    
    task_emb = task_embedding(torch.tensor([2]).cuda()) if cuda else task_embedding(torch.tensor([2]))
    
    optim.zero_grad()
    for epoch in range(train_epochs):
        step = 0
        for x, y, l in emotion_reader.iter():
            emo_loss, emo_acc = emotion_loss(
                                            x, y, l,
                                            bert, transformer, task_emb,
                                            emotion_classifier, emo_loss_fn, 
                                            cuda
                                   )
            emo_loss.backward(retain_graph=True)
            optim.step()
            losses[int(step%10)] = emo_loss.cpu()
            accs[int(step%10)] = emo_acc
            print("step:%d | loss/acc = %.3f/%.3f"%(step, emo_loss, emo_acc))
            if step %10 == 9:
                print('emotion task: %6d: [%5d/%5d], senti_loss/senti_acc = %6.8f/%6.7f ' % ( step,
                                                                                epoch, train_epochs,
                                                                                losses.mean(), accs.mean(),
                                                                            )
                         )       
            step += 1 


# In[10]:


def subj_cls_train(subj_reader, bert, transformer, task_embedding,
                   subj_classifier, train_epochs, subj_loss_fn,
                   cuda=False
                  ):
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':1e-6},
                                {'params': task_embedding.parameters(), 'lr':5e-5},
                                {'params': transformer.parameters(), 'lr': 5e-5},
                                {'params': subj_classifier.parameters(), 'lr': 5e-5}
                             ]
    )
    losses = np.zeros([10]) 
    accs = np.zeros([10])
    task_emb = task_embedding(torch.tensor([1]).cuda()) if cuda else task_embedding(torch.tensor([1]))
    
    optim.zero_grad()
    for epoch in range(train_epochs):
        step = 0
        for x, y, l in subj_reader.iter():
            sj_loss, sj_acc = subj_loss(x, y, l,
                                    bert, transformer, task_emb,
                                    subj_classifier, subj_loss_fn, 
                                    cuda
                                   )
            sj_loss.backward(retain_graph=True)
            optim.step()
            losses[int(step%10)] = sj_loss.cpu()
            accs[int(step%10)] = sj_acc
            print("step:%d | loss/acc = %.3f/%.3f"%(step, sj_loss, sj_acc))
            if step %10 == 9:
                print('subjective task: %6d: [%5d/%5d], senti_loss/senti_acc = %6.8f/%6.7f ' % ( step,
                                                                                epoch, train_epochs,
                                                                                losses.mean(), accs.mean(),
                                                                            )
                         )       
            step += 1 


# In[11]:


def senti_cls_train(senti_reader, bert, transformer,
                    task_embedding, senti_classifier,
                    train_epochs, senti_loss_fn,
                    cuda=False
                   ):
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':1e-6},
                                {'params': task_embedding.parameters(), 'lr':1e-5},
                                {'params': transformer.parameters(), 'lr': 1e-5},
                                {'params': senti_classifier.parameters(), 'lr': 1e-5}
                             ]
    )
    losses = np.zeros([10]) 
    accs = np.zeros([10])
    
    task_emb = task_embedding(torch.tensor([0]).cuda()) if cuda else task_embedding(torch.tensor([0]))
    
    optim.zero_grad()
    for epoch in range(train_epochs):
        step = 0
        for x, y, l in senti_reader.iter():
            st_loss, st_acc = senti_loss(x, y, l, 
                                    bert, transformer, task_emb,
                                    senti_classifier, senti_loss_fn,
                                    cuda
                                   )
            st_loss.backward(retain_graph=True)
            optim.step()
            losses[int(step%10)] = st_loss.cpu()
            accs[int(step%10)] = st_acc
            print("step:%d | loss/acc = %.3f/%.3f"%(step, st_loss, st_acc))
            if step %10 == 9:
                print('sentiment task: %6d: [%5d/%5d], senti_loss/senti_acc = %6.8f/%6.7f ' % ( step,
                                                                                epoch, train_epochs,
                                                                                losses.mean(), accs.mean(),
                                                                            )
                         )    
            step += 1 


# ### 联合训练函数

# #### 三人舞
# 情感任务，主观客观性任务，　情绪分析任务，　三个共同训练的函数

# In[12]:


def JointLearning(senti_reader, subj_reader, emotion_reader, 
                  bert, transformer, task_embedding, 
                  senti_classifier, subj_classifier, emotion_classifier,
                  cuda=False
                 ):
    
    #stage 3: deploy the trainning on the emotion classification task
#     emo_weights = torch.tensor(
#             WeightsForUmbalanced(
#                 emotion_reader.label
#             ),
#             dtype = torch.float32
#     )
#     emo_loss_fn = nn.CrossEntropyLoss(weight=emo_weights) if not cuda else nn.CrossEntropyLoss(weight=emo_weights.cuda())
#     emo_cls_train(
#         emotion_reader, 
#         bert, 
#         transformer, 
#         task_embedding, 
#         emotion_classifier, 
#         2, 
#         emo_loss_fn,
#         cuda
#     )
#     emotion_reader.reset_batchsize(5)    
    
    # stage 1: deploy the trainning on the senti classification task
    senti_weights = torch.tensor(
            WeightsForUmbalanced(
                senti_reader.label
            ),
            dtype=torch.float32
    )
    senti_loss_fn = nn.CrossEntropyLoss(weight=senti_weights.cuda()) if cuda else nn.CrossEntropyLoss(weight=senti_weights)
    senti_cls_train(
                    senti_reader, 
                    bert, 
                    transformer, 
                    task_embedding, 
                    senti_classifier, 
                    2, 
                    senti_loss_fn,
                    cuda
                   )
    #stage 2: deploy the trainning on the subj classification task
    subj_weights = torch.tensor(
            WeightsForUmbalanced(
                subj_reader.label
            ),
            dtype = torch.float32
    )
    subj_loss_fn = nn.CrossEntropyLoss(weight=subj_weights) if not cuda else nn.CrossEntropyLoss(weight=subj_weights.cuda())
    subj_cls_train(
                   subj_reader, 
                   bert, 
                   transformer, 
                   task_embedding, 
                   subj_classifier, 
                   2, 
                   subj_loss_fn,
                   cuda
                  )
    
    joint_model_save_as = './MTLTrain/PreTrModel_epoch.pkl'
    torch.save(
        {
            "bert":bert.state_dict(),
            "transformer":transformer.state_dict(),
            "task_embedding":task_embedding.state_dict(),
            "senti_classifier": senti_classifier.state_dict(),
            "subj_classifier": subj_classifier.state_dict(),
#             "emotion_classifier": emotion_classifier.state_dict()
        },
        joint_model_save_as
    )
    print("saved pretrained model!")

    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':5e-7},
                                {'params': task_embedding.parameters(), 'lr':1e-6},
                                {'params': transformer.parameters(), 'lr': 1e-6},
                                {'params': senti_classifier.parameters(), 'lr': 1e-6},
                                {'params': subj_classifier.parameters(), 'lr':1e-6},
                                {'params': emotion_classifier.parameters(), 'lr':1e-6}
                             ]
    )
    
    max_epoch = 100
    
    losses = np.zeros([3, 10]) 
    accs = np.zeros([3, 10])
    # [[senti_loss_1, ..., senti_loss_10], [subj_loss_1, ..., subj_loss_10], [emo_loss_1, ..., emo_loss_10]] 
    senti_task_id = torch.tensor([0]) if not cuda else torch.tensor([0]).cuda()
    subj_task_id = torch.tensor([1]) if not cuda else torch.tensor([1]).cuda()
#     emo_task_id = torch.tensor([2]) if not cuda else torch.tensor([2]).cuda()
    
    loss_weight = torch.tensor([0.333, 0.333, 0.333]) if not cuda else torch.tensor([0.333, 0.333, 0.333]).cuda()
    
    batchs = min(senti_reader.label.shape[0], subj_reader.label.shape[0], emotion_reader.label.shape[0])
    optim.zero_grad()
    for epoch in range(max_epoch):
        step = 0
        for ((xst, yst, lst), (xsj, ysj, lsj), (xem, yem, lem)) in zip(senti_reader.iter(), subj_reader.iter(), emotion_reader.iter()):
            st_loss, st_acc = senti_loss(xst, yst, lst, 
                                    bert, transformer, task_embedding(senti_task_id),
                                    senti_classifier, senti_loss_fn,
                                    cuda
                                   )
#             st_loss.backward()
            MTL_Loss = st_loss*loss_weight[0]
    
            losses[0][step%10] = st_loss.tolist()
            accs[0][step%10] = st_acc
             
            sj_loss, sj_acc = subj_loss(xsj, ysj, lsj,
                                    bert, transformer, task_embedding(subj_task_id),
                                    subj_classifier, subj_loss_fn,
                                    cuda
                                   )
#             sj_loss.backward()
            MTL_Loss += sj_loss*loss_weight[1]

            losses[1][step%10] = sj_loss.tolist()
            accs[1][step%10] = sj_acc
            
            emo_loss, emo_acc = emotion_loss(xem, yem, lem,
                                    bert, transformer, task_embedding(emo_task_id),
                                    emotion_classifier, emo_loss_fn,
                                    cuda
                                   )
#             emo_loss.backward()
            MTL_Loss += emo_loss*loss_weight[2]
    
            losses[2][step%10] = emo_loss.tolist()
            accs[2][step%10] = emo_acc
            
            MTL_Loss.backward()
            optim.step()
            optim.zero_grad()
            print("%6d|MTL_Loss:%6.8f, senti_loss/senti_acc = %6.8f/%6.7f | subj_loss/subj_acc = %6.8f/%6.7f | emo_loss/emo_acc = %6.8f/%6.7f" % (
                                                                                                step, MTL_Loss,        
                                                                                                losses[0].mean(), accs[0].mean(),
                                                                                                losses[1].mean(), accs[1].mean(),
                                                                                                losses[2].mean(), accs[2].mean()
            )
            )
            if step % 10 == 9:
                print('%6d: [%5d/%5d], MTL_Loss|%6.8f, senti_loss/senti_acc = %6.8f/%6.7f | subj_loss/subj_acc = %6.8f/%6.7f | emo_loss/emo_acc = %6.8f/%6.7f' % (
                                                                                                step,
                                                                                                epoch,max_epoch, MTL_Loss,
                                                                                                losses[0].mean(), accs[0].mean(),
                                                                                                losses[1].mean(), accs[1].mean(),
                                                                                                losses[2].mean(), accs[2].mean()
                                                                                            )
                     )
                loss_weight = torch.tensor(
                                            (1.0/accs.mean(axis=1))/sum(1.0/accs.mean(axis=1)),
                                            dtype=torch.float32
                                )
                print("\n\n loss_weight:", loss_weight)
            step += 1
        joint_model_save_as = './MTLTrain/jointModel_epoch%03d.pkl'% (epoch)
        torch.save(
            {
                "bert":bert.state_dict(),
                "transformer":transformer.state_dict(),
                "task_embedding":task_embedding.state_dict(),
                "senti_classifier": senti_classifier.state_dict(),
                "subj_classifier": subj_classifier.state_dict(),
                "emotion_classifier": emotion_classifier.state_dict()
            },
            joint_model_save_as
        )


# #### 双人舞
# 情感分析和主观客观性任务联合训练的函数

# In[13]:


def SentiSubjLearning(senti_reader, subj_reader, 
                  bert, transformer, task_embedding, 
                  senti_classifier, subj_classifier,
                  cuda=False
                 ):  
    
    # stage 1: deploy the trainning on the senti classification task
    senti_weights = torch.tensor(
            WeightsForUmbalanced(
                senti_reader.label
            ),
            dtype=torch.float32
    )
    senti_loss_fn = nn.CrossEntropyLoss(weight=senti_weights.cuda()) if cuda else nn.CrossEntropyLoss(weight=senti_weights)
    #stage 2: deploy the trainning on the subj classification task
    subj_weights = torch.tensor(
            WeightsForUmbalanced(
                subj_reader.label
            ),
            dtype = torch.float32
    )
    subj_loss_fn = nn.CrossEntropyLoss(weight=subj_weights) if not cuda else nn.CrossEntropyLoss(weight=subj_weights.cuda())

    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':5e-7},
                                {'params': task_embedding.parameters(), 'lr':1e-6},
                                {'params': transformer.parameters(), 'lr': 1e-6},
                                {'params': senti_classifier.parameters(), 'lr': 1e-6},
                                {'params': subj_classifier.parameters(), 'lr':1e-6}
                             ]
    )
    
    max_epoch = 10
    
    losses = np.zeros([3, 10]) 
    accs = np.zeros([3, 10])
    # [[senti_loss_1, ..., senti_loss_10], [subj_loss_1, ..., subj_loss_10], [emo_loss_1, ..., emo_loss_10]] 
    senti_task_id = torch.tensor([0]) if not cuda else torch.tensor([0]).cuda()
    subj_task_id = torch.tensor([1]) if not cuda else torch.tensor([1]).cuda()
    
    loss_weight = torch.tensor([0.5, 0.5]) if not cuda else torch.tensor([0.5, 0.5]).cuda()
    
    batchs = min(senti_reader.label.shape[0], subj_reader.label.shape[0])
    optim.zero_grad()
    for epoch in range(10, max_epoch+10):
        step = 0
        for ((xst, yst, lst), (xsj, ysj, lsj)) in zip(senti_reader.iter(), subj_reader.iter() ):
            st_loss, st_acc = senti_loss(xst, yst, lst, 
                                    bert, transformer, task_embedding(senti_task_id),
                                    senti_classifier, senti_loss_fn,
                                    cuda
                                   )
            MTL_Loss = st_loss*loss_weight[0]
            losses[0][step%10] = st_loss.tolist()
            accs[0][step%10] = st_acc
             
            sj_loss, sj_acc = subj_loss(xsj, ysj, lsj,
                                    bert, transformer, task_embedding(subj_task_id),
                                    subj_classifier, subj_loss_fn,
                                    cuda
                                   )
            MTL_Loss += sj_loss*loss_weight[1]
            losses[1][step%10] = sj_loss.tolist()
            accs[1][step%10] = sj_acc
            
            MTL_Loss.backward()
            optim.step()
            optim.zero_grad()
            print("%6d %6d|MTL_Loss:%6.8f, senti_loss/senti_acc = %6.8f/%6.7f | subj_loss/subj_acc = %6.8f/%6.7f " % (
                                                                                                step, batchs, MTL_Loss,        
                                                                                                losses[0].mean(), accs[0].mean(),
                                                                                                losses[1].mean(), accs[1].mean()
            )
            )
            if step % 10 == 9:
                print('%6d %6d: [%5d/%5d], MTL_Loss|%6.8f, senti_loss/senti_acc = %6.8f/%6.7f | subj_loss/subj_acc = %6.8f/%6.7f' % (
                                                                                                step, batchs, 
                                                                                                epoch,max_epoch, MTL_Loss,
                                                                                                losses[0].mean(), accs[0].mean(),
                                                                                                losses[1].mean(), accs[1].mean()
                                                                                            )
                     )
                loss_weight = torch.tensor(
                                            (1.0/( accs.mean(axis=1)[:2] ) )/sum( 1.0/accs.mean(axis=1)[:2] ),
                                            dtype=torch.float32
                                )
                print("\n\n loss_weight:", loss_weight)
            step += 1
        joint_model_save_as = './MTLTrain/jointModel_epoch%03d.pkl'% (epoch)
        torch.save(
            {
                "bert":bert.state_dict(),
                "transformer":transformer.state_dict(),
                "task_embedding":task_embedding.state_dict(),
                "senti_classifier": senti_classifier.state_dict(),
                "subj_classifier": subj_classifier.state_dict()
            },
            joint_model_save_as
        )


# #### 双任务增强
# subj任务和sentiment任务来增强谣言检测模块
# 
# > 此时应当要能从Reader中随机地挑出一个batch出来训练

# In[14]:


def MTLTrainRDMModel(rdm_model, bert, rdm_classifier,
                     transformer, task_embedding, senti_classifier, subj_classifier, 
                     sentiReader, subjReader, 
                    tokenizer, t_steps, new_data_len=[], logger=None, cuda=False, 
                        log_dir="RDMBertTrain"):
    batch_size = 20 
    max_gpu_batch = 5 #cannot load a larger batch into the limited memory, but we could  accumulates grads
    sentiReader.reset_batchsize(max_gpu_batch)
    subjReader.reset_batchsize(max_gpu_batch)
    
    assert(batch_size%max_gpu_batch == 0)
    
    sum_loss = np.zeros(4)
    sum_acc = np.zeros(3)
    
    t_acc = 0.9
    ret_acc = 0.0
    init_states = torch.zeros([1, 5, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    
    senti_task_id = torch.tensor([0]) if not cuda else torch.tensor([0]).cuda()
    subj_task_id = torch.tensor([1]) if not cuda else torch.tensor([1]).cuda()
    
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    
    senti_weights = torch.tensor(
            WeightsForUmbalanced(
                sentiReader.label
            ),
            dtype=torch.float32
    )
    senti_loss_fn = nn.CrossEntropyLoss(weight=senti_weights.cuda()) if cuda else nn.CrossEntropyLoss(weight=senti_weights)
    
    subj_weights = torch.tensor(
            WeightsForUmbalanced(
                subjReader.label
            ),
            dtype = torch.float32
    )
    subj_loss_fn = nn.CrossEntropyLoss(weight=subj_weights) if not cuda else nn.CrossEntropyLoss(weight=subj_weights.cuda())

    loss_weight = torch.tensor([0.8, 0.1, 0.1]) if not cuda else torch.tensor([0.8, 0.1, 0.1]).cuda()   
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':5e-5},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-5},
                                {'params': rdm_model.parameters(), 'lr': 5e-5},
                                {'params': task_embedding.parameters(), 'lr':1e-6},
                                {'params': transformer.parameters(), 'lr': 1e-6},
                                {'params': senti_classifier.parameters(), 'lr': 1e-6},
                                {'params': subj_classifier.parameters(), 'lr':1e-6}
                             ]
    )
    
    writer = SummaryWriter(log_dir)
    
    acc_tmp = np.zeros([3, int(batch_size/max_gpu_batch)])
    loss_tmp = np.zeros([4, int(batch_size/max_gpu_batch)])
    
    for step in range(t_steps):
        optim.zero_grad()
        for j in range(int(batch_size/max_gpu_batch)):
            if len(new_data_len) > 0:
                x, x_len, y = get_df_batch(step, max_gpu_batch, new_data_len, tokenizer=tokenizer)
            else:
                x, x_len, y = get_df_batch(step, max_gpu_batch, tokenizer=tokenizer) 
            loss, acc = rdm_loss(x, y, x_len, bert, rdm_model, rdm_classifier, loss_fn)
            MTL_Loss = loss*loss_weight[0]
            
            xst, yst, lst = sentiReader.sample()
            st_loss, st_acc = senti_loss(xst, yst, lst, 
                                    bert, transformer, task_embedding(senti_task_id),
                                    senti_classifier, senti_loss_fn,
                                    cuda
                                   )
            MTL_Loss += st_loss*loss_weight[1]
            
            xsj, ysj, lsj = subjReader.sample()
            sj_loss, sj_acc = subj_loss(xsj, ysj, lsj,
                                    bert, transformer, task_embedding(subj_task_id),
                                    subj_classifier, subj_loss_fn,
                                    cuda
                                   )
            MTL_Loss += sj_loss*loss_weight[2]
            
            MTL_Loss.backward()
            
            loss_tmp[:, j] = np.array([MTL_Loss, loss, st_loss, sj_loss])
            acc_tmp[:, j] = np.array([acc, st_acc, sj_acc])
            
        optim.step()        
        writer.add_scalar('Train Loss', loss_tmp[0].mean(), step)
        writer.add_scalar('Train Accuracy', acc_tmp[0].mean(), step)
        
        sum_acc += acc_tmp.mean(axis=1)
        sum_loss += loss_tmp.mean(axis=1)
        
        print("%6d %6d|MTL_Loss:%6.8f, rdm_loss/rdm_acc = %6.8f/%6.7f | senti_loss/senti_acc = %6.8f/%6.7f | subj_loss/subj_acc = %6.8f/%6.7f " % (
                                                                                                step, t_steps, loss_tmp[0].mean(),        
                                                                                            loss_tmp[1].mean(), acc_tmp[0].mean(),
                                                                                            loss_tmp[2].mean(), acc_tmp[1].mean(),
                                                                                            loss_tmp[3].mean(), acc_tmp[2].mean()
            )
            )
        
        if step % 10 == 9:
            sum_loss = sum_loss / 10
            sum_acc = sum_acc / 10
            print("MTL_Loss:%6.8f, rdm_loss/rdm_acc = %6.8f/%6.7f | senti_loss/senti_acc = %6.8f/%6.7f | subj_loss/subj_acc = %6.8f/%6.7f " % (
                                                                                            sum_loss[0],        
                                                                                            sum_loss[1], sum_acc[0],
                                                                                            sum_loss[2], sum_acc[1],
                                                                                            sum_loss[3], sum_acc[2]
            )
            )
            if step%100 == 99:
                rdm_save_as = './%s/rdmModel_epoch%03d.pkl'% (log_dir, step/100)
                torch.save(
                    {
                        "bert":bert.state_dict(),
                        "transformer":transformer.state_dict(),
                        "task_embedding":task_embedding.state_dict(),
                        "senti_classifier": senti_classifier.state_dict(),
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


# ### 主函数部分

# #### 模型部分

# In[15]:


tt = BertTokenizer.from_pretrained("./bertModel/")


# In[16]:


bb = BertModel.from_pretrained("./bertModel/")

task_embedding = nn.Embedding(3, 768)

encoder_layer = nn.TransformerEncoderLayer(768, 8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, 1)

subj_cls = nn.Linear(768, 2)

bert = bb.cuda()
transformer = transformer_encoder.cuda()
task_embedding = task_embedding.cuda()
subj_cls = subj_cls.cuda()

senti_cls = nn.Linear(768, 2).cuda()


# #### 各个任务的数据

# In[17]:


subj_file = "./rotten_imdb/subj.data"
obj_file = "./rotten_imdb/obj.data"
tr, dev, te = load_data(subj_file, obj_file)

subj_train_reader = SubjObjReader(tr, 20, tt)
subj_valid_reader = SubjObjReader(dev, 20, tt)
subj_test_reader =  SubjObjReader(te, 20, tt)


# In[18]:


train_file = "./trainingandtestdata/training.1600000.processed.noemoticon.csv"
test_file = "./trainingandtestdata/testdata.manual.2009.06.14.csv"
train_set, test_set = tsentiLoader.load_data(train_file, test_file)


# In[19]:


senti_train_reader = tsentiLoader.tSentiReader(train_set[:10000], 20, tt)
senti_train_reader.label = np.delete(senti_train_reader.label, 1, axis=2)
# senti_valid_reader = tsentiLoader.tSentiReader(train_set[10000:10100], 20, tt)
# senti_test_reader =  tsentiLoader.tSentiReader(test_set, 20, tt)


# In[20]:


load_data_fast()

# ### 联合训练部分

# #### 导入预训练模型

joint_save_as = './MTLTrain/jointModel_epoch015.pkl'
checkpoint = torch.load(joint_save_as)
senti_cls.load_state_dict(checkpoint['senti_classifier'])
bert.load_state_dict(checkpoint['bert'])
transformer.load_state_dict(checkpoint['transformer'])
task_embedding.load_state_dict(checkpoint['task_embedding'])
subj_cls.load_state_dict(checkpoint['subj_classifier'])


# #### 联合训练

rdm_model = RDM_Model(768, 300, 256, 0.2).cuda()
cm_model = CM_Model(300, 256, 2).cuda()
rdm_classifier = nn.Linear(256, 2).cuda()


# In[ ]:


MTLTrainRDMModel(rdm_model, bert, rdm_classifier,
                     transformer, task_embedding, senti_cls, subj_cls, 
                     senti_train_reader, subj_train_reader, 
                    tt, 20000, new_data_len=[], logger=None, cuda=True, 
                        log_dir="RDMBertTrain")


# In[ ]:




