#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from pytorch_transformers import *
from torch import nn
from SubjObjLoader import *
import numpy as np
import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils


# In[2]:


b, t, c = BertModel,       BertTokenizer,      'bert-base-uncased'
tt = t.from_pretrained("bert-base-uncased", cached_dir = "/home/hadoop/transformer_pretrained_models/bert-base-uncased-pytorch_model.bin")


# In[3]:


bb = b.from_pretrained("bert-base-uncased")


# In[4]:


subj_file = "/home/hadoop/rotten_imdb/subj.data"
obj_file = "/home/hadoop/rotten_imdb/obj.data"
tr, dev, te = load_data(subj_file, obj_file)


# In[5]:


train_reader = SubjObjReader(tr, 20, tt)
valid_reader = SubjObjReader(dev, 20, tt)
test_reader =  SubjObjReader(te, 20, tt)


# In[6]:


# for x, y, l in train_reader.iter():
#     break

# y


# In[6]:


class SentimentModel(nn.Module):
    """
        Input: the outputs of bert
        Model: BiLSTM
        Output: sentence embedding
    """
    def __init__(   self,
                    bert ,
                    hidden_size,
                    rnn_size, 
                    senti_num,
                    drop_out
                ):
        super(SentimentModel, self).__init__()
        self.bert = bert
        self.gru = nn.GRU(hidden_size, rnn_size, 1, batch_first=True)
        self.classifier = nn.Linear(rnn_size, senti_num)
        
    def forward(self, word_ids):
        outs = [self.bert( torch.tensor([input_]).cuda() )
                for input_ in word_ids]
#         cls_feature =torch.cat([item[0][0][0] + item[1][0] for item in outs], axis=0).reshape([-1, 768]).cuda()
        states = [item[0][0] for item in outs]
        pad = rnn_utils.pad_sequence(states, batch_first=True)
        h_outs, h_final = self.gru(pad)
        cls_feature = h_final[0] + h_outs.max(axis=1)[0]
        pred_scores = self.classifier(cls_feature)
        return pred_scores, cls_feature


# In[7]:


def Count_Accs(ylabel, pred_scores):
    correct_preds = np.array([1 if y1==y2 else 0 for (y1, y2) in zip(ylabel, pred_scores.argmax(axis=1))])
    y_idxs = [idx if yl >0 else idx - len(ylabel) for (idx, yl) in enumerate(ylabel)]
    pos_idxs = list(filter(lambda x: x > 0, y_idxs))
    neg_idxs = list(filter(lambda x: x < 0, y_idxs))
    acc = sum(correct_preds) / (1.0 * len(ylabel))
    pos_acc = sum(correct_preds[pos_idxs])/(1.0*len(pos_idxs))
    neg_acc = sum(correct_preds[neg_idxs])/(1.0*len(neg_idxs))
    return acc, pos_acc, neg_acc


# In[8]:


def Loss_Fn(ylabel, pred_scores):
    diff = (ylabel - pred_scores)*(ylabel - pred_scores)
    return diff.mean()


# In[11]:


def Train(bert, train_reader, valid_reader, test_reader, hidden_dim, logger, logdir):
    max_epoch = 5
    loss_fn = Loss_Fn
    senti_model = SentimentModel(bert, hidden_dim, 300, 2, 0.8).cuda()
    optim = torch.optim.Adagrad([
#                                 {'params': senti_model.bert.parameters(), 'lr':1e-2},
                                {'params': senti_model.classifier.parameters(), 'lr': 1e-1},
                                {'params': senti_model.gru.parameters(), 'lr': 1e-1}
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
            pred_scores, _ = senti_model(x)
            ylabel = y.argmax(axis=1)
            acc, pos_acc, neg_acc = Count_Accs(ylabel, pred_scores)
            print("step %d| pos_acc/neg_acc = %6.8f/%6.7f, pos_pred/all_pred = %2d/%2d"%(step, 
                                                                                         pos_acc, neg_acc,
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
        for x, y, l in  valid_reader.iter():
            pred_scores, _ = senti_model(x)
            loss = loss_fn(pred_scores, torch.tensor(y, dtype=torch.float32).cuda())
            loss.backward() # to release the GPU cache...... 
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
        save_as = '/home/hadoop/ERD/BERTTwitter/epoch%03d_%.4f.pkl' % (step%1000, sum_acc)
        torch.save(senti_model.state_dict(), save_as)
        sum_acc = 0.0
        sum_loss = 0.0


# In[12]:


from logger import MyLogger
logger = MyLogger("BERTSubObj")
Train(bb, train_reader, valid_reader, test_reader, 768, logger, "BERTSubjObj")


# In[ ]:


# # Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
# BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
#                       BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
#                       BertForQuestionAnswering]

# # All the classes for an architecture can be initiated from pretrained weights for this architecture
# # Note that additional weights added for fine-tuning are only initialized
# # and need to be trained on the down-stream task
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# for model_class in BERT_MODEL_CLASSES:
#     # Load pretrained model/tokenizer
#     model = model_class.from_pretrained('bert-base-uncased')

# # Models can return full list of hidden-states & attentions weights at each layer
# model = model_class.from_pretrained(pretrained_weights,
#                                     output_hidden_states=True,
#                                     output_attentions=True)
# input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
# all_hidden_states, all_attentions = model(input_ids)[-2:]

# # Models are compatible with Torchscript
# model = model_class.from_pretrained(pretrained_weights, torchscript=True)
# traced_model = torch.jit.trace(model, (input_ids,))

# # Simple serialization for models and tokenizers
# model.save_pretrained('./directory/to/save/')  # save
# model = model_class.from_pretrained('./directory/to/save/')  # re-load
# tokenizer.save_pretrained('./directory/to/save/')  # save
# tokenizer = tokenizer_class.from_pretrained('./directory/to/save/')  # re-load

