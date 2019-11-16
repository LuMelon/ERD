import json
import os
import time
import datetime
import numpy as np
import gensim
import random
import math
import re
import pickle
import torch

files = []
data = {}
data_ID = []
data_len = []
data_y = []
# word2vec = gensim.models.KeyedVectors.load_word2vec_format('/home/hadoop/word2vec.model')
with open("/home/hadoop/word2vec.txt", "rb") as handle:
        word2vec = pickle.load(handle)
print("load glove finished")
# c2vec = chars2vec.load_model('eng_300')
reward_counter = 0
eval_flag = 0

def get_data_ID():
    global data_ID
    return data_ID

def get_data_len():
    global data_len
    return data_len

def get_curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def list_files(data_path):
    global data, files
    fs = os.listdir(data_path)
    for f1 in fs:
        tmp_path = os.path.join(data_path, f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.split('.')[-1] == 'json':
                files.append(tmp_path)
        else:
            list_files(tmp_path)


def str2timestamp(str_time):
    month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
             'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
             'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    ss = str_time.split(' ')
    m_time = ss[5] + "-" + month[ss[1]] + '-' + ss[2] + ' ' + ss[3]
    d = datetime.datetime.strptime(m_time, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp


def data_process(file_path):
    ret = {}
    ss = file_path.split("/")
    data = json.load(open(file_path, mode="r", encoding="utf-8"))
    # 'Wed Jan 07 11:14:08 +0000 2015'
    # print("SS:", ss)
    ret[ss[6]] = {'label': ss[5], 'text': [data['text'].lower()], 'created_at': [str2timestamp(data['created_at'])]}
    return ret

def transIrregularWord(line):
    if not line:
        return ''
    line.lower()
    line = re.sub("@[^ ]*", "{ mention someone }", line)
    line = re.sub("#[^ ]*", "{ special topic }", line)
    line = re.sub("http(.?)://[^ ]*", "{ a special link }", line)
    return line 


def load_test_data_fast():
    global data, data_ID, data_len, data_y, eval_flag
    with open("../../data/data_dict.txt", "rb") as handle:
        data = pickle.load(handle)
    data_ID = np.load("../../data/test_data_ID.npy").tolist()
    data_len = np.load("../../data/test_data_len.npy").tolist()
    data_y = np.load("../../data/test_data_y.npy").tolist()
    max_sent = max( map(lambda value: max(map(lambda txt_list: len(txt_list), value['text']) ), list(data.values()) ) )
    print("max_sent:", max_sent, ",  max_seq_len:", max(data_len))
    eval_flag = int(len(data_ID) / 4) * 3
    print("{} data loaded".format(len(data))) 

def load_data_fast():
    global data, data_ID, data_len, data_y, valid_data_ID, valid_data_y, valid_data_len
    with open("data/data_dict.txt", "rb") as handle:
        data = pickle.load(handle)
    data_ID = np.load("data/data_ID.npy").tolist()
    data_len = np.load("data/data_len.npy").tolist()
    data_y = np.load("data/data_y.npy").tolist()
    # valid_data_ID = np.load("data/valid_data_ID.npy").tolist()
    # valid_data_len = np.load("data/valid_data_len.npy").tolist()
    # valid_data_y = np.load("data/valid_data_y.npy").tolist()
    valid_data_ID = np.load("data/test_data_ID.npy").tolist()
    valid_data_len = np.load("data/test_data_len.npy").tolist()
    valid_data_y = np.load("data/test_data_y.npy").tolist()
    max_sent = max( map(lambda value: max(map(lambda txt_list: len(txt_list), value['text']) ), list(data.values()) ) )
    print("max_sent:", max_sent, ",  max_seq_len:", max(data_len))
    eval_flag = int(len(data_ID) / 4) * 3
    print("{} data loaded".format(len(data)))    



def sortTempList(temp_list):
    time = np.array([item[0] for item in temp_list])
    posts = np.array([item[1] for item in temp_list])
    idxs = time.argsort().tolist()
    rst = [[t, p] for (t, p) in zip(time[idxs], posts[idxs])]
    del time, posts
    return rst

def load_data(data_path, FLAGS):
    # get data files path
    global data, files, data_ID, data_len, eval_flag
    data = {}
    files = []
    data_ID = []
    data_len = []
    list_files(data_path) #load all filepath to files
    max_sent = 0
    # load data to json
    for file in files:
        td = data_process(file) # read out the information from json file, and organized it as {dataID:{'key':val}}
        for key in td.keys(): # use temporary data to organize the final whole data
            if key in data:
                data[key]['text'].append(td[key]['text'][0])
                data[key]['created_at'].append(td[key]['created_at'][0])
            else:
                data[key] = td[key]
    # convert to my data style
    for key, value in data.items():
        temp_list = []
        for i in range(len(data[key]['text'])):
            temp_list.append([data[key]['created_at'][i], data[key]['text'][i]])
        temp_list = sortTempList(temp_list)
        data[key]['text'] = []
        data[key]['created_at'] = []
        ttext = ""
        last = 0
        for i in range(len(temp_list)):
            # if temp_list[i][0] - temp_list[0][0] > FLAGS.time_limit * 3600 or len(data[key]['created_at']) >= 100:
            #     break
            if i % FLAGS.post_fn == 0: # merge the fixed number of texts in a time interval
                if len(ttext) > 0: # if there are data already in ttext, output it as a new instance
                    words = transIrregularWord(ttext)
                    data[key]['text'].append(words)
                    data[key]['created_at'].append(temp_list[i][0])
                ttext = temp_list[i][1]
            else:
                ttext += " " + temp_list[i][1]
            last = i
        # keep the last one
        if len(ttext) > 0:
            words = transIrregularWord(ttext)
            data[key]['text'].append(words)
            data[key]['created_at'].append(temp_list[last][0])

    for key in data.keys():
        data_ID.append(key)
    data_ID = random.sample(data_ID, len(data_ID)) #shuffle the data id
    for i in range(len(data_ID)): #pre processing the extra informations
        data_len.append(len(data[data_ID[i]]['text']))
        if data[data_ID[i]]['label'] == "rumours":
            data_y.append([1.0, 0.0])
        else:
            data_y.append([0.0, 1.0])
    eval_flag = int(len(data_ID) / 4) * 3
    print("{} data loaded".format(len(data)))


def get_df_batch(start, batch_size, new_data_len=[], cuda=True):
    data_x = []
    m_data_y = np.zeros([batch_size, 2], dtype=np.int32)
    m_data_len = np.zeros([batch_size], dtype=np.int32)
    miss_vec = 0
    hit_vec = 0
    if len(new_data_len) > 0:
        t_data_len = new_data_len
    else:
        t_data_len = data_len
    mts = start * batch_size
    if mts >= len(data_ID):
        mts = mts % len(data_ID)

    for i in range(batch_size):
        m_data_y[i] = data_y[mts]
        m_data_len[i] = t_data_len[mts]
        seq = []
        for j in range(t_data_len[mts]):
            sent = []
            t_words = transIrregularWord(data[data_ID[mts]]['text'][j]).split(" ")
            for k in range(len(t_words)):
                m_word = t_words[k]
                try:
                    sent.append( torch.tensor([word2vec[m_word]], dtype=torch.float32) )
                except KeyError:
                    miss_vec += 1
                    sent.append( torch.tensor([word2vec['{'] +word2vec['an'] +  word2vec['unknown'] + word2vec['word'] + word2vec['}'] ], dtype=torch.float32) )
                except IndexError:
                    raise
                else:
                    hit_vec += 1
            sent_tensor = torch.cat(sent)
            seq.append(sent_tensor)
        data_x.append(seq)
        mts += 1
        if mts >= len(data_ID): # read data looply
            mts = mts % len(data_ID)
            
    return data_x, m_data_len, m_data_y


# seq_states is the date_x to get
# max_id is the next corpus to take
def get_rl_batch(ids, seq_states, stop_states, counter_id, start_id, total_data):
    input_x = np.zeros([FLAGS.batch_size, FLAGS.max_sent_len, FLAGS.embedding_dim], dtype=np.float32)
    input_y = np.zeros([FLAGS.batch_size, FLAGS.class_num], dtype=np.float32)
    miss_vec = 0
    total_data = len(data_len)

    for i in range(FLAGS.batch_size):
        # seq_states:records the id of a sentence in a sequence
        # stop_states: records whether the sentence is judged by the program
        if stop_states[i] == 1 or seq_states[i] >= data_len[ids[i]]: 
            ids[i] = counter_id + start_id
            seq_states[i] = 0
            try:
                t_words = data[ data_ID[ids[i]] ]['text'][seq_states[i]]
            except:
                print(ids[i], seq_states[i])
            for j in range(len(t_words)):
                m_word = t_words[j]
                try:
                    input_x[i][j] = word2vec[m_word]
                except:
                    miss_vec = 1
            # if len(t_words) != 0:
            #     input_x[i][:len(t_words)] = c2vec.vectorize_words(t_words)
            input_y[i] = data_y[ids[i]]
            counter_id += 1
            counter_id = counter_id % total_data
        else:
            try:
                t_words = data[ data_ID[ids[i]] ]['text'][seq_states[i]]
            except:
                print("ids and seq_states:", ids[i], seq_states[i])
                t_words = []
            for j in range(len(t_words)):
                m_word = t_words[j]
                try:
                    input_x[i][j] = word2vec[m_word]
                except:
                    miss_vec += 1

            # if len(t_words) != 0:
            #     input_x[i][:len(t_words)] = c2vec.vectorize_words(t_words)
            input_y[i] = data_y[ids[i]]
        # point to the next sequence
        seq_states[i] += 1

    return input_x, input_y, ids, seq_states, counter_id

def accuracy_on_valid_data(rdm_model = None, sent_pooler = None, rdm_classifier=None, new_data_len=[], cuda=True):
    def Count_Acc(ylabel, preds):
        correct_preds = np.array(
            [1 if y1==y2 else 0 
            for (y1, y2) in zip(ylabel, preds)]
        )
        acc = sum(correct_preds) / (1.0 * len(ylabel))
        return acc
    
    batch_size = 20
    t_steps = int(len(valid_data_ID)/batch_size)
    sum_acc = 0.0
    miss_vec = 0
    mts = 0
    hit_vec = 0
    if len(new_data_len) > 0:
        t_data_len = new_data_len
    else:
        t_data_len = valid_data_len
    
    for step in range(t_steps):
        data_x = []
        m_data_y = np.zeros([batch_size, 2], dtype=np.int32)
        m_data_len = np.zeros([batch_size], dtype=np.int32)
        for i in range(batch_size):
            m_data_y[i] = valid_data_y[mts]
            m_data_len[i] = t_data_len[mts]
            seq = []
            for j in range(t_data_len[mts]):
                sent = []
                t_words = transIrregularWord(data[valid_data_ID[mts]]['text'][j]).split(" ")
                for k in range(len(t_words)):
                    m_word = t_words[k]
                    try:
                        sent.append( torch.tensor([word2vec[m_word]], dtype=torch.float32))
                    except KeyError:
                        miss_vec += 1
                        sent.append( torch.tensor([word2vec['{'] +word2vec['an'] +  word2vec['unknown'] + word2vec['word'] + word2vec['}'] ], dtype=torch.float32) )
                    except IndexError:
                        raise
                    else:
                        hit_vec += 1
                if len(sent) != 0 :
                    sent_tensor = torch.cat(sent)
                else:
                    print("empty sentence:", t_words)
                seq.append(sent_tensor)
            data_x.append(seq)
            mts += 1
            if mts >= len(data_ID): # read data looply
                mts = mts % len(data_ID)
        
        
        if rdm_model is not None and sent_pooler is not None and rdm_classifier is not None:
            with torch.no_grad():
                seq = sent_pooler(data_x)
                rdm_hiddens = rdm_model(seq)
                batchsize, _, _ = rdm_hiddens.shape
                rdm_outs = torch.cat(
                    [ rdm_hiddens[i][m_data_len[i]-1].unsqueeze(0) for i in range(batchsize)] 
                    # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
                )
                rdm_scores = rdm_classifier(
                    rdm_outs
                )
                rdm_preds = rdm_scores.argmax(axis=1)
                y_label = torch.tensor(m_data_y).argmax(axis=1).cuda() if cuda else torch.tensor(m_data_y).argmax(axis=1)
                acc = Count_Acc(y_label, rdm_preds)
        sum_acc += acc
    mean_acc = sum_acc / (1.0*t_steps)
    return mean_acc


# not to stop -0.1, so that to be early
# DDQN y = r + Q(S, argmax(Q))
def get_reward(isStop, ss, pys, ids, seq_ids):
    global reward_counter
    reward = np.zeros([len(isStop)], dtype=np.float32)
    for i in range(len(isStop)):
        if isStop[i] == 1:
            if np.argmax(pys[ids[i]][seq_ids[i]-1]) == np.argmax(data_y[ids[i]]):
                reward_counter += 1 # more number of correct prediction, more rewards
                r = 1 + FLAGS.reward_rate * math.log(reward_counter)
                reward[i] = r   
            else:
                reward[i] = -100
        else:
            reward[i] = -0.01 + 0.99 * max(ss[i])
    return reward


def get_new_len(sess, mm):
    new_x_len = np.zeros([len(data_ID)], dtype=np.int32)

    for i in range(len(data_ID)):
        init_state = np.zeros([1, FLAGS.hidden_dim], dtype=np.float32)
        e_state = sess.run(mm.df_state, feed_dict={mm.init_states: init_state})
        for j in range(data_len[i]):
            t_words = data[data_ID[i]]['text'][j]
            e_x = np.zeros([1, FLAGS.max_sent_len, FLAGS.embedding_dim], dtype=np.float32)
            for k in range(len(t_words)):
                m_word = t_words[k]
                try:
                    e_x[0][k] = word2vec[m_word]
                except:
                    miss_word = 1
            # if len(t_words) != 0:
            #     e_x[0][:len(t_words)] = c2vec.vectorize_words(t_words)
            batch_dic = {mm.rl_state: e_state, mm.rl_input: e_x, mm.dropout_keep_prob: 1.0}
            e_isStop, mNewState = sess.run([mm.isStop, mm.rl_new_state], batch_dic)
            e_state = mNewState

            if e_isStop == 1:
                new_x_len[i] = j+1
                break
        if new_x_len[i] == 0 or new_x_len[i] > data_len[i]:
            new_x_len[i] = data_len[i]

    # print(" Max Length: " + str(max(new_x_len)) +
    #       " Min Length: " + str(min(new_x_len)) +
    #       " Ave Length: " + str(np.mean(new_x_len))) + " (" + str(np.mean(data_len)) + ")"

    return new_x_len


def get_RL_Train_batch(D):
    s_state = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32) 
    s_x = np.zeros([FLAGS.batch_size, FLAGS.max_sent_len, FLAGS.embedding_dim], dtype=np.float32)
    s_isStop = np.zeros([FLAGS.batch_size, FLAGS.action_num], dtype=np.float32)
    s_rw = np.zeros([FLAGS.batch_size], dtype=np.float32)

    m_batch = random.sample(D, FLAGS.batch_size)
    for i in range(FLAGS.batch_size):
        s_state[i] = m_batch[i][0]
        s_x[i] = m_batch[i][1]
        s_isStop[i][m_batch[i][2]] = 1
        s_rw[i] = m_batch[i][3]

    return s_state, s_x, s_isStop, s_rw
