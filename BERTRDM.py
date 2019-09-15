#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8
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
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils

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
#         self.gru_model = nn.GRU(word_embedding_dim, 
#                                 self.hidden_dim, 
#                                 batch_first=True
# #                                 dropout=dropout_prob
#                             )
        encoder_layer = nn.TransformerEncoderLayer(word_embedding_dim, 8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2, norm=nn.LayerNorm(word_embedding_dim))
        self.transpose_layer = nn.Linear(word_embedding_dim, self.hidden_dim)
        self.DropLayer = nn.Dropout(dropout_prob)
        self.PoolLayer = pooling_layer(word_embedding_dim, sent_embedding_dim) 
        
    def forward(self, sentiModel, x_emb, x_len, init_states): 
        """
        input_x: [batchsize, max_seq_len, sentence_embedding_dim] 
        x_emb: [batchsize, max_seq_len, max_word_num, embedding_dim]
        x_len: [batchsize]
        init_states: [batchsize, hidden_dim]
        """
#         print("x_emb:", x_emb.shape)
#         print("x_len:", x_len)
        batchsize, max_seq_len, max_sent_len, emb_dim = x_emb.shape
        pool_feature = x_emb[:, :, 0, :] + x_emb[:, :, 1:, :].max(axis=2)[0]
        print("pool_feature shape:", pool_feature.shape)
#         print("pool_feature shape:", pool_feature.shape)
#         pool_feature = self.PoolLayer(x_emb[:, :, 1:-1, :])
#         sent_feature = sentiModel( 
#                 x_emb.reshape(
#                     [-1, max_sent_len, emb_dim]
#                 ) 
#             ).reshape(
#                 [batchsize, max_seq_len, -1]
#             )
#         pooled_input_x_dp = self.DropLayer(input_x)
#         df_outputs, df_last_state = self.gru_model(pool_feature, init_states)
        mask = torch.triu( torch.ones(max_seq_len, max_seq_len)*float('-inf'), diagonal=1).cuda()
        print("mask shape:", mask.shape)
        df_outputs = self.transpose_layer(
            self.transformer_encoder( 
                pool_feature.transpose(0, 1).cuda(), 
                mask=mask
            ).transpose(0, 1)
        )
        print("df_outputs shape:", df_outputs.shape)
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

class SentimentModel(nn.Module):
    """
        Input: the outputs of bert
        Model: BiLSTM
        Output: sentence embedding
    """
    def __init__(   self,
                    hidden_size,
                    rnn_size, 
                    drop_out
                ):
        super(SentimentModel, self).__init__()
        self.gru = nn.GRU(hidden_size, rnn_size, 1, batch_first=True)
        
    def forward(self, seq_input):
        h_outs, h_final = self.gru(seq_input)
        cls_feature = h_final[0] + h_outs.max(axis=1)[0]
        return cls_feature

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
    diff = (ylabel - pred_scores)*(ylabel - pred_scores)
    l1 = ylabel.argmax(axis=1).tolist()
    l2 = pred_scores.argmax(axis=1).tolist()
#     print("l1:", l1)
#     print("l2:", l2)
    weight = torch.tensor([1.0 if y1==y2 else 3.0 for y1, y2 in zip(l1, l2)]).cuda()
    return (weight*diff.mean(axis=1)).mean()

def TrainSentiModel(bert, senti_model, classifier, 
                    train_reader, valid_reader, test_reader, 
                    logger, logdir):
    max_epoch = 10
    loss_fn = Loss_Fn
    senti_model.cuda()
    optim = torch.optim.Adagrad([
#                                 {'params': senti_model.bert.parameters(), 'lr':1e-2},
            {'params': classifier.parameters(), 'lr': 1e-1},
            {'params': senti_model.parameters(), 'lr': 1e-1}
         ],
            weight_decay = 0.2
        
    )
    writer = SummaryWriter(logdir)
    batches = train_reader.label.shape[0]
    step = 0
    for epoch in range(max_epoch):
        sum_acc = 0.0
        sum_loss = 0.0
        for x, y, l in  train_reader.iter():
            embedding = layer2seq(bert, x, cuda=True)
            feature = senti_model(embedding)
            pred_scores = classifier(feature)
            ylabel = y.argmax(axis=1)
            acc, pos_acc, neg_acc, _, _, _, _ =                         Count_Accs(ylabel, pred_scores.argmax(axis=1))
            print("step %d| pos_acc/neg_acc = %6.8f/%6.7f,                 pos_pred/all_pred = %2d/%2d"%(
                            step, pos_acc, neg_acc,
                            sum(pred_scores.argmax(axis=1)),
                            len(pred_scores)
                                ))
            loss = loss_fn(pred_scores, torch.tensor(y, dtype=torch.float32).cuda())
            optim.zero_grad()
            loss.backward()
            optim.step()
            writer.add_scalar('Train Loss', loss, step)
            writer.add_scalar('Train Accuracy', acc, step)
            
            sum_acc += acc
            sum_loss += loss
            step += 1
            
            if step % 10 == 0:
                sum_loss = sum_loss / 10
                sum_acc = sum_acc / 10
                ret_acc = sum_acc
                print('%6d: %d [%5d/%5d], train_loss/accuracy = %6.8f/%6.7f' % (step,
                                                        epoch, step%batches,
                                                        batches,
                                                        sum_loss, sum_acc,
                                                        ))
                logger.info('%6d: %d [%5d/%5d], train_loss/accuracy = %6.8f/%6.7f' % (step,
                                                            epoch, step%batches,
                                                            batches,
                                                            sum_loss, sum_acc,
                                                        ))
                sum_acc = 0.0
                sum_loss = 0.0
                
        sum_loss = 0
        sum_acc = 0
        with torch.no_grad():
            for x, y, l in  valid_reader.iter():
                embedding = layer2seq(bert, x, cuda=True)
                feature = senti_model(embedding)
                pred_scores = classifier(feature)
                loss = loss_fn(pred_scores, torch.tensor(y, dtype=torch.float32).cuda())
                acc, _, _, _, _, _, _ =                             Count_Accs(y.argmax(axis=1), pred_scores.argmax(axis=1))
                sum_acc += acc
                sum_loss += loss
            sum_acc = sum_acc/(1.0*valid_reader.label.shape[0])
            sum_loss = sum_loss/(1.0*valid_reader.label.shape[0])
        print('[%5d/%5d], valid_loss/accuracy = %6.8f/%6.7f' % (epoch, max_epoch,
                                                    sum_loss, sum_acc,
                                                    ))
        logger.info('[%5d/%5d], valid_loss/accuracy = %6.8f/%6.7f' % (epoch, max_epoch,
                                                                sum_loss, sum_acc,
                                                                ))
        writer.add_scalar('Valid Loss', sum_loss, epoch)
        writer.add_scalar('Valid Accuracy', sum_acc, epoch)
        senti_save_as = '/home/hadoop/ERD/%s/sentiModel_epoch%03d.pkl' % (logdir, epoch)
        torch.save(
            {
                'sentiModel':senti_model.state_dict() ,
                'sentiClassifier':classifier.state_dict(),
                'bert': bert.state_dict()
            },
            senti_save_as
        )
        sum_acc = 0.0
        sum_loss = 0.0

def TrainRDMModel(rdm_model, bert, sentiModel, rdm_classifier, 
                    tokenizer, t_steps, new_data_len=[], logger=None, 
                        log_dir="RDMBertTrain"):
    batch_size = 20 
    max_gpu_batch = 5 #cannot load a larger batch into the limited memory, but we could  accumulates grads
    assert(batch_size%max_gpu_batch == 0)
    sum_acc = 0.0
    sum_loss = 0.0
    t_acc = 0.9
    ret_acc = 0.0
    init_states = torch.zeros([1, 5, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    layer_norm = torch.nn.LayerNorm(rdm_model.hidden_dim).cuda()
    loss_CE = nn.CrossEntropyLoss()
    loss_fn = Loss_Fn
    optim = torch.optim.Adam([
                                {'params': bert.parameters(), 'lr':1e-1},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-1},
                                {'params': rdm_model.parameters(), 'lr': 5e-1}
#                                 {'params': sentiModel.parameters(), 'lr': 1e-2}
                             ]
#                                 weight_decay = 0.2
    )
    #print grad in rdm_classifier 
    for par in rdm_classifier.parameters():
        par.register_hook(lambda grad:print("rdm_cls grad:", torch.sum(grad)))
        
   #print grad in bert
#     for par in bert.parameters():
#         par.register_hook(lambda grad:print("bert grad:", grad))
    
    #print grad in gru_model
    for par in rdm_model.parameters():
        par.register_hook(lambda grad: print("rdm_gru grad:", torch.sum(grad)))
    
    writer = SummaryWriter(log_dir)
    for step in range(t_steps):
        optim.zero_grad()
#         sum_loss = torch.tensor(0.0).cuda()
        for j in range(int(batch_size/max_gpu_batch)):
            if len(new_data_len) > 0:
                x, x_len, y = get_df_batch(step, max_gpu_batch, new_data_len, tokenizer=tokenizer)
            else:
                x, x_len, y = get_df_batch(step, max_gpu_batch, tokenizer=tokenizer)
            
#             with torch.no_grad():
            x_emb = Word_ids2SeqStates(x, bert, 3, cuda=True) 
            batchsize, max_seq_len, max_sent_len, emb_dim = x_emb.shape
            rdm_hiddens, rdm_outs = rdm_model(sentiModel, x_emb, x_len, init_states)
            rdm_feature = torch.cat(
                    rdm_outs # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
                ).reshape(
                    [-1, rdm_model.hidden_dim]
            )
            rdm_scores = F.relu( 
                rdm_classifier(
                    layer_norm(
                        rdm_feature
                    )
                )
            )
            rdm_preds = rdm_scores.argmax(axis=1)
            y_label = y.argmax(axis=1)
#             loss = loss_fn(rdm_scores, torch.tensor(y, dtype=torch.float32).cuda())
            loss = loss_CE(rdm_scores, torch.tensor(y_label).cuda() )
            loss.backward()
            print("\n\n\n\n\n\n\n\n\nrdm_feature:\n", rdm_feature[:, :5])
            print("rdm_scores:\n", rdm_scores)
            print("y:\n", y)
#         sum_loss.backward()
        optim.step()
        acc, pos_acc, neg_acc, y_idxs, pos_idxs,         neg_idxs, correct_preds = Count_Accs(y_label, rdm_preds)
        print("\n\n\n\n\n\n\nstep %d| pos_acc/pos_cnt = %3.3f/%3d,                neg_acc/neg_cnt = %3.3f/%3d                 pos_pred/all_pred = %2d/%2d"%(
                step, pos_acc, len(pos_idxs),
                neg_acc, len(neg_idxs), 
                sum(rdm_scores.argmax(axis=1)),
                len(rdm_scores)
                )
        )
#         print("gru_model W_hh:", rdm_model.gru_model.weight_hh_l0[:10, :10])
#         print("gru_w_hh:\n", rdm_model.gru_model.weight_hh_l0[:10, :10])
#             logger.info("correct_preds:%d|%s \n \
#                          pos_idxs:%d|%s \n \
#                          neg_idxs:%d|%s \
#                          "%(
#                             4*step+j, str(correct_preds.tolist()), 
#                             4*step+j, str(pos_idxs),
#                             4*step+j, str(neg_idxs)
#                            )
#                        )
        writer.add_scalar('Train Loss', loss, step)
        writer.add_scalar('Train Accuracy', acc, step)

        sum_loss += loss
        sum_acc += acc
        

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
#                         "bert":bert.state_dict(),
#                         "sentiModel":sentiModel.state_dict(),
                        "rmdModel":rdm_model.state_dict(),
                        "rdm_classifier": rdm_classifier.state_dict()
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

def TrainCMModel(bert, sentiModel, rdm_model, rdm_classifier, cm_model, tokenizer, log_dir, logger, FLAGS):
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

    import BertRDMLoader

    del get_rl_batch
    del get_reward

    importlib.reload(BertRDMLoader)
    load_data_fast()
    get_reward = BertRDMLoader.get_reward
    get_rl_batch = BertRDMLoader.get_rl_batch

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


# In[2]:


b, t, c = BertModel,       BertTokenizer,      'bert-base-uncased'
tt = t.from_pretrained("bert-base-uncased", cached_dir = "/home/hadoop/transformer_pretrained_models/bert-base-uncased-pytorch_model.bin")

bb = b.from_pretrained("bert-base-uncased")

s_model = SentimentModel(
                    768,
                    300, 
                    0.2
                )


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

bert = bb.cuda()
sentiModel = s_model.cuda()
tokenizer = tt


# In[7]:


# sentiClassifier = nn.Linear(300, 2).cuda()

# [   20/   20], test_loss/accuracy = 0.08379094/0.8850000

# [   20/   20], valid_loss/accuracy = 0.07104027/0.9040000

# subj_file = "/home/hadoop/rotten_imdb/subj.data"
# obj_file = "/home/hadoop/rotten_imdb/obj.data"
# tr, dev, te = SubjObjLoader.load_data(subj_file, obj_file)

# train_reader = SubjObjLoader.SubjObjReader(tr, 20, tokenizer)
# valid_reader = SubjObjLoader.SubjObjReader(dev, 20, tokenizer)
# test_reader =  SubjObjLoader.SubjObjReader(te, 20, tokenizer)

# sentiLogger = MyLogger("SubjObjTrainer")

# sentiClassifier.load_state_dict(checkpoint['sentiClassifier'])

# TrainSentiModel(bert, sentiModel, sentiClassifier, 
#                 train_reader, valid_reader, test_reader, 
#                 sentiLogger, "BERTSubjObj/")


# In[3]:


CM_logger = MyLogger("CMTest")
load_data_fast()


# In[4]:


rdm_model = RDM_Model(768, 300, 64, 0.2)
cm_model = CM_Model(300, 64, 2)
rdm_model = rdm_model.cuda()
cm_model = cm_model.cuda()
rdm_classifier = nn.Linear(64, 2).cuda()
cm_log_dir="CMBertTrain"


# In[5]:


senti_save_as = '/home/hadoop/ERD/%s/sentiModel_epoch%03d.pkl' % ("BERTSubjObj/", 0)

checkpoint = torch.load(senti_save_as)

sentiModel.load_state_dict(checkpoint['sentiModel'])

bert.load_state_dict(checkpoint['bert'])

rdm_logger = MyLogger("RDMLogger")


# In[ ]:



TrainRDMModel(rdm_model, bert, sentiModel, rdm_classifier, 
                    tokenizer, 100000, new_data_len=[], logger=rdm_logger, 
                        log_dir="RDMBertTrain")


# In[ ]:




