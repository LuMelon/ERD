#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

class pooling_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(pooling_layer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        
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
    def __init__(self, sent_embedding_dim, hidden_dim, dropout_prob):
        super(RDM_Model, self).__init__()
        self.embedding_dim = sent_embedding_dim
        self.hidden_dim = hidden_dim
        self.gru_model = nn.GRU(self.embedding_dim, 
                                self.hidden_dim, 
                                batch_first=True, 
                                dropout=dropout_prob
                            )
        self.DropLayer = nn.Dropout(dropout_prob)
        
    def forward(self, input_x, x_len, init_states): 
        """
        input_x: [batchsize, max_seq_len, sentence_embedding_dim] 
        x_len: [batchsize]
        init_states: [batchsize, hidden_dim]
        """
        pooled_input_x_dp = self.DropLayer(input_x)
        df_outputs, df_last_state = self.gru_model(pooled_input_x_dp, init_states)
        hidden_outs = [df_outputs[i][:(x_len[i]-1)] for i in range(len(input_x))]
        final_outs = [df_outputs[i][x_len[i]-1] for i in range(len(input_x))]
        return hidden_outs, final_outs


class CM_Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, action_num):
        super(CM_Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_num = action_num
        self.PoolLayer = pooling_layer(self.embedding_dim, 
                                            self.hidden_dim)
        self.DenseLayer = nn.Linear(self.hidden_dim, 64)
        self.Classifier = nn.Linear(64, self.action_num)
        
    def forward(self, rdm_model, rl_input, rl_state):
        """
        rl_input: [batchsize, max_word_num, embedding_dim]
        rl_state: [1, batchsize, hidden_dim]
        """
        assert(rl_input.ndim==3)
        batchsize, max_word_num, embedding_dim = rl_input.shape
        assert(embedding_dim==self.embedding_dim)
        
        pooled_rl_input = self.PoolLayer(
            rl_input.reshape(
                [-1, 1, max_word_num, self.embedding_dim]
            )
        ).reshape([-1, 1, self.hidden_dim])
        
        print("pooled_rl_input:", pooled_rl_input.shape)
        print("rl_state:", rl_state.shape)
        rl_output, rl_new_state = rdm_model.gru_model(
                                            pooled_rl_input, 
                                            rl_state
                                        )
        rl_h1 = nn.relu(
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

def Word_ids2SeqStates(word_ids, bert):
    outs = [bert( torch.tensor([input_]).cuda() )
                for input_ in word_ids]
    states = [item[0][0] for item in outs]
    pad = rnn_utils.pad_sequence(states, batch_first=True)
    return pad

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
        h_outs, h_final = self.gru(pad)
        cls_feature = h_final[0] + h_outs.max(axis=1)[0]
        return cls_feature

def Count_Accs(ylabel, pred_scores):
    correct_preds = np.array(
        [1 if y1==y2 else 0 
        for (y1, y2) in zip(ylabel, pred_scores.argmax(axis=1))]
    )
    y_idxs = [idx if yl >0 else idx - len(ylabel) 
            for (idx, yl) in enumerate(ylabel)]
    pos_idxs = list(filter(lambda x: x > 0, y_idxs))
    neg_idxs = list(filter(lambda x: x < 0, y_idxs))
    acc = sum(correct_preds) / (1.0 * len(ylabel))
    pos_acc = sum(correct_preds[pos_idxs])/(1.0*len(pos_idxs))
    neg_acc = sum(correct_preds[neg_idxs])/(1.0*len(neg_idxs))
    return acc, pos_acc, neg_acc, y_idxs, pos_idxs, neg_idxs, correct_preds

def Loss_Fn(ylabel, pred_scores):
    diff = (ylabel - pred_scores)*(ylabel - pred_scores)
    return diff.mean()

def TrainSentiModel(bert, senti_model, classifier, 
                    train_reader, valid_reader, test_reader, 
                    hidden_dim, logger, logdir):
    max_epoch = 5
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
        for x, y, l in  valid_reader.iter():
            embedding = Word_ids2SeqStates(x)
            feature = senti_model(embedding)
            pred_scores = classifier(feature)
            ylabel = y.argmax(axis=1)
            acc, pos_acc, neg_acc, _, _, _, _ = \
                        Count_Accs(ylabel, pred_scores)
            print("step %d| pos_acc/neg_acc = %6.8f/%6.7f, \
                pos_pred/all_pred = %2d/%2d"%(
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
            
            if step % 20 == 0:
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
        for x, y, l in  valid_reader.iter():
            print("valid count:", count)
            embedding = Word_ids2SeqStates(x)
            feature = senti_model(embedding)
            pred_scores = classifier(feature)
            loss = loss_fn(pred_scores, torch.tensor(y, dtype=torch.float32).cuda())
            loss.backward() # to release the GPU cache...... 
            acc, _, _, _, _, _, _ = \
                        Count_Accs(ylabel, pred_scores)
            sum_acc += acc
            sum_loss += loss
            count += 1
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
        senti_save_as = '/home/hadoop/ERD/BERTTwitter/sentiModel_epoch%03d_%.4f.pkl' % (step%1000, sum_acc)
        cls_save_as = '/home/hadoop/ERD/BERTTwitter/sentiCLS_epoch%03d_%.4f.pkl' % (step%1000, sum_acc)
        bert_save_as = '/home/hadoop/ERD/BERTTwitter/sentiBERT_epoch%03d_%.4f.pkl' % (step%1000, sum_acc)
        torch.save(senti_model.state_dict(), senti_save_as)
        torch.save(classifier.state_dict(), cls_save_as)
        torch.save(bert.state_dict(), bert_save_as)
        sum_acc = 0.0
        sum_loss = 0.0


def padding_sequence(sequences):
    max_size = sequences[0].size()
    trailing_dims = max_size[2:]
    max_seq_len = max([s.size(0) for s in sequences])
    max_sent_len = max([s.size(1) for s in sequences])
    out_dims = (len(sequences), max_seq_len, max_sent_len) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(0.0)
    for i, tensor in enumerate(sequences):
        seq_len = tensor.size(0)
        sent_len = tensor.size(1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :seq_len, :sent_len, ...] = tensor
    return out_tensor

def layer2seq(bert, layer):
    outs = [bert( torch.tensor([input_]))
                for input_ in layer]
    states = [item[0][0] for item in outs]
    return rnn_utils.pad_sequence(states, batch_first=True)

def Word_ids2SeqStates(word_ids, bert, ndim):
    assert(ndim == 3)
    embedding = [layer2seq(bert, layer) for layer in word_ids]
    return padding_sequence(embedding)

def TrainRDMModel(rdm_model, bert, sentiModel, rdm_classifier, 
                    tokenizer, t_steps, new_data_len=[], logger=None, 
                        log_dir="RDMBertTrain"):
    batch_size = 5 #cannot load a larger batch into the limited memory, but we could  accumulates grads
    sum_loss = 0.0
    sum_acc = 0.0
    ret_acc = 0.0
    init_states = np.zeros([1, 5, FLAGS.hidden_dim], dtype=np.float32)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adagrad([
                                {'params': bert.parameters(), 'lr':1e-3},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-2},
                                {'params': rdm_model.parameters(), 'lr': 5e-2},
                                {'params': sentiModel.parameters(), 'lr': 1e-2}
                             ],
                                weight_decay = 0.2
    )
    
    for step in range(t_steps):
        optim.zero_grad()
        for j in range(4): 
            if len(new_data_len) > 0:
                x, x_len, y = get_df_batch(step*4+j, batch_size, new_data_len, tokenizer=tokenizer)
            else:
                x, x_len, y = get_df_batch(step*4+j, batch_size, tokenizer=tokenizer)
            x_emb = Word_ids2SeqStates(x, bert, 3) 
            batchsize, max_seq_len, max_sent_len,\
                                     emb_dim = x_emb.shape
            sent_feature = sentiModel( 
                x_emb.reshape(
                    [-1, max_sent_len, emb_dim]
                ) 
            ).reshape(
                [batchsize, max_seq_len, -1]
            )
            rdm_hiddens, rdm_outs = rdm_model(sent_feature, x_len, init_states)
            rdm_scores = rdm_classifier(rdm_outs[0])
            rdm_preds = rdm_scores.argmax(axis=1)
            loss = loss_fn(rdm_scores, y)
            loss.backward()
            acc, pos_acc, neg_acc, y_idxs, pos_idxs, \
            neg_idxs, correct_preds = Count_Accs(y, rdm_preds)
            logger.info("correct_preds:%d|%s"%(4*step+j, 
                                        str(correct_preds.tolist())))
            logger.info("pos_idxs:%d|%s"%(4*step+j, 
                                        str(pos_idxs)))
            logger.info("neg_idxs:%d|%s"%(4*step+j, 
                                        str(neg_idxs)))
            sum_loss += loss
            sum_acc += acc
        optim.step()
        

        if i % 10 == 9:
            sum_loss = sum_loss / 40
            sum_acc = sum_acc / 40
            print('%3d | %d , train_loss/accuracy = %6.8f/%6.7f'\
             % (step, t_steps, 
                sum_loss, sum_acc,
                ))
            logger.info('%3d | %d , train_loss/accuracy = %6.8f/%6.7f'\
             % (i, t_steps, 
                sum_loss, sum_acc,
                ))
            if sum_acc > t_acc:
                break
            sum_acc = 0.0
            sum_loss = 0.0

    print(get_curtime() + " Train df Model End.")
    logger.info(get_curtime() + " Train df Model End.")
    return ret_acc

def TrainCMModel(cm_model, rdm_model, bert, sentiModel, 
                logger, log_dir="CMBertTrain"):
    pass


# In[ ]:




