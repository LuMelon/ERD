#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
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


# In[2]:


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


def TrainRDMModel(rdm_model, bert, rdm_classifier, 
                    tokenizer, t_steps, new_data_len=[], logger=None, 
                        log_dir="RDMBertTrain"):
    batch_size = 20 
    max_gpu_batch = 2 #cannot load a larger batch into the limited memory, but we could  accumulates grads
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
                                {'params': bert.parameters(), 'lr':5e-5},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-3},
                                {'params': rdm_model.parameters(), 'lr': 5e-3}
                             ]
    )
    
    writer = SummaryWriter(log_dir)
    acc_l = np.zeros(splits)
    loss_l = np.zeros(splits)
    for step in range(t_steps):
        optim.zero_grad()
        try:
            for j in range(splits):
                if len(new_data_len) > 0:
                    x, x_len, y = get_df_batch(step*splits+j, max_gpu_batch, new_data_len, tokenizer=tokenizer)
                else:
                    x, x_len, y = get_df_batch(step, max_gpu_batch, tokenizer=tokenizer)
                x_emb = Word_ids2SeqStates(x, bert, 3, cuda=True) 
                batchsize, max_seq_len, max_sent_len,                                     emb_dim = x_emb.shape
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
                acc_l[j], _, _, _, _, _, _ = Count_Accs(y_label, rdm_preds)
                loss = loss_fn(rdm_scores, torch.tensor(y_label).cuda())
                loss.backward()
                loss_l[j] = float(loss)
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
            logger.info('%3d | %d , train_loss/accuracy = %6.8f/%6.7f'             % (step, t_steps, 
                sum_loss, sum_acc,
                ))
            if step%500 == 499:
                rdm_save_as = '/home/hadoop/ERD/%s/rdmModel_epoch%03d.pkl'                                    % (log_dir, step/500)
                torch.save(
                    {
                        "bert":bert.state_dict(),
                        "rmdModel":rdm_model.state_dict(),
                        "rdm_classifier": rdm_classifier.state_dict()
                    },
                    rdm_save_as
                )
#                 rdm_model, bert, sentiModel, rdm_classifier
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
        counter += 1


# In[7]:


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cached_dir = "/home/hadoop/transformer_pretrained_models/bert-base-uncased-pytorch_model.bin")


# In[8]:


bert = BertModel.from_pretrained("bert-base-uncased").cuda()


# In[9]:


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


# In[10]:


CM_logger = MyLogger("CMTest")
load_data_fast()


# In[11]:


rdm_model = RDM_Model(768, 300, 256, 0.2).cuda()
cm_model = CM_Model(300, 256, 2).cuda()
rdm_classifier = nn.Linear(256, 2).cuda()
cm_log_dir="CMBertTrain"


# senti_save_as = '/home/hadoop/ERD/%s/sentiModel_epoch%03d.pkl' % ("BERTSubjObj/", 0)

# checkpoint = torch.load(senti_save_as)

# bert.load_state_dict(checkpoint['bert'])

rdm_logger = MyLogger("RDMLogger")


# In[ ]:


TrainRDMModel(rdm_model, bert, rdm_classifier, 
                    tokenizer, 10000, new_data_len=[], logger=rdm_logger, 
                        log_dir="RDMBertTrain")




