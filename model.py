#!/usr/bin/env python
# coding: utf-8
from collections import deque
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.losses import Reduction
import dataloader
from dataUtils import *
from config import *

def Test():
    print(get_df_batch(10))

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def shared_pooling_layer(inputs, input_dim, hidden_dim, 
                            max_seq_len, max_word_num, output_dim, 
                                scope = 'share_pooling'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w_t = tf.Variable(tf.random_uniform([input_dim, hidden_dim], -1.0, 1.0), name="w_t")
        b_t = tf.Variable(tf.constant(0.01, shape=[hidden_dim]), name="b_t")
        t_inputs = tf.reshape(inputs, [-1, input_dim])
        t_h = tf.nn.xw_plus_b(t_inputs, w_t, b_t)
        t_h = tf.reshape(t_h, [-1, max_word_num, hidden_dim])
        t_h_expended = tf.expand_dims(t_h, -1)
        pooled = tf.nn.max_pool(
            t_h_expended,
            ksize=[1, max_word_num, 1, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="max_pool"
        )
        outs = tf.reshape(pooled, [-1, max_seq_len, hidden_dim])
    return outs

def pooling_layer(inputs, input_dim, max_seq_len, 
                        max_word_num, output_dim, 
                            scope='pooling_layer'):
    t_inputs = tf.reshape(inputs, [-1, input_dim])
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        b = tf.Variable(tf.constant(0.01, shape=[output_dim]))

        h = tf.nn.xw_plus_b(t_inputs, w, b)
        hs = tf.reshape(h, [-1, max_word_num, output_dim])

        inputs_expended = tf.expand_dims(hs, -1)
        # [seq, words, out] --> [seq, words, out, 1] --> [seq, 1, out, 1] --> [1, seq, out]
        pooled = tf.nn.max_pool(
            inputs_expended,
            ksize=[1, max_word_len, 1, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="max_pool"
        )
        cnn_outs = tf.reshape(pooled, [-1, max_seq_len, output_dim]) 
    return cnn_outs 


# In[ ]:


class CM_Model:
    def __init__(self, max_word_num, embedding_dim, hidden_dim, action_num):
        self.max_word_num = max_word_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_num = action_num
    def __call__(self, rdm_model, rl_input, rl_state):
        pooled_rl_input = shared_pooling_layer(rl_input, self.embedding_dim, 
                                                    self.hidden_dim, 1, self.max_word_num, 
                                                        self.hidden_dim)
        pooled_rl_input = tf.reshape(pooled_rl_input, [-1, self.hidden_dim])
        print("pooled_rl_input:", pooled_rl_input)
        print("rl_state:", rl_state)
        rl_output, rl_new_state = rdm_model.df_cell(pooled_rl_input, rl_state)
        with tf.variable_scope("CM_Model", reuse=tf.AUTO_REUSE):
            w_ss1 = tf.Variable(tf.truncated_normal([self.hidden_dim, 64], stddev=0.01))
            b_ss1 = tf.Variable(tf.constant(0.01, shape=[64]))
            rl_h1 = tf.nn.relu(tf.nn.xw_plus_b(rl_state, w_ss1, b_ss1))  # replace the process here
            w_ss2 = tf.Variable(tf.truncated_normal([64, self.action_num], stddev=0.01))
            b_ss2 = tf.Variable(tf.constant(0.01, shape=[self.action_num]))
            stopScore = tf.nn.xw_plus_b(rl_h1, w_ss2, b_ss2, name="stopScore")
            isStop = tf.argmax(stopScore, 1, name="isStop")
        return stopScore, isStop, rl_new_state


class RDM_Model:
    def __init__(self, max_seq_len, max_word_num, embedding_dim, hidden_dim):
        self.max_seq_len = max_seq_len
        self.max_word_num = max_word_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        with tf.variable_scope("RDM_Model", reuse=tf.AUTO_REUSE):
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")
            self.df_cell = rnn.GRUCell(self.hidden_dim)
            self.df_cell = rnn.DropoutWrapper(self.df_cell, output_keep_prob=self.dropout_keep_prob)
        
    def __call__(self, input_x, x_len, init_states): #input_x: [batchsize, max_seq_len, max_word_num, max_char_num] 
        with tf.variable_scope('pooling_layer', reuse=tf.AUTO_REUSE):
            # pooled_input_x = shared_pooling_layer(input_x, self.embedding_dim, self.max_seq_len, self.max_word_num, self.hidden_dim) # replace the shared_pooling_layer with a sentiment analysis model
            # dropout layer
            pooled_input_x_dp = tf.nn.dropout(input_x, self.dropout_keep_prob)
            df_outputs, df_last_state = tf.nn.dynamic_rnn(
                                                            self.df_cell, 
                                                            pooled_input_x_dp, 
                                                            x_len, 
                                                            initial_state=init_states, 
                                                            dtype=tf.float32
                                                          )
        return df_outputs, df_last_state

class SentiModel:
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
    def __call__(self, sentence):
        with tf.variable_scope("SentiModel", reuse=tf.AUTO_REUSE):
            conv_input = tf.layers.conv1d(
                            sentence, 
                            self.num_filters, 
                            self.kernel_size, 
                            strides=1, 
                            padding='valid', 
                            name='conv2', 
                            trainable=True
            )
            feature_map = tf.nn.relu(conv_input) # [batchsize, conv_feats, filters]
            pooled_feat = tf.reduce_max(feature_map, 1) #[batchsize, 1, filters]
            # pooled_feat = tf.reshape(pooled_feat, [-1, self.num_filters])
        return pooled_feat 

def InferSentiTrainGraph(char_model, lm, senti_model, max_word_num, max_char_num, embedding_dim, hidden_dim, sent_num):
    sent_x = tf.placeholder(tf.int32, shape = [None, max_word_num, max_char_num])
    sent_y = tf.placeholder(tf.float32, shape = [None, sent_num])
    words2vec = char_model(sent_x) #[None, max_word_num, kernerl_size]
    cnn_outs = tf.reshape(words2vec, [-1, max_word_num, sum(char_model.kernel_features)])
    words_embedding, sentence_embedding = lm(cnn_outs)
    words_embedding = tf.identity(words_embedding, "rnn_out_puts")
    words_embedding = tf.transpose(words_embedding, [1, 0, 2])
    senti_features = senti_model(words_embedding)
    classifier = tf.layers.Dense(sent_num, activation=tf.nn.relu, trainable=True)
    senti_rst = classifier(senti_features)
    sent_scores = tf.nn.softmax(senti_rst, axis=1)
    sent_pred = tf.argmax(sent_scores, 1, name="predictions")
    sent_loss = tf.losses.softmax_cross_entropy(
                        sent_y,
                        senti_rst,
                        weights=1.0,
                        label_smoothing=0,
                        scope=None,
                        loss_collection=tf.GraphKeys.LOSSES,
                        reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
                    )
    sent_correct_predictions = tf.equal(sent_pred, tf.argmax(sent_y, 1))
    sent_acc = tf.reduce_mean(tf.cast(sent_correct_predictions, "float"), name="accuracy")
    sent_global_step = tf.Variable(0, name="global_step", trainable=False)
    sent_train_op = tf.train.AdagradOptimizer(0.01).minimize(sent_loss, sent_global_step)        
    return adict(
                # dropout_keep_prob = self.dropout_keep_prob,
                sent_x = sent_x,
                sent_y = sent_y,
                feature = senti_features,
                sent_scores = sent_scores,
                sent_pred = sent_pred,
                sent_loss = sent_loss,
                sent_acc = sent_acc,
                sent_global_step = sent_global_step,
                sent_train_op = sent_train_op
            )


def InferRDMTrainGraph(char_model, lm, senti_model, rdm_model, 
                            max_seq_len, max_word_num, max_char_num, 
                                hidden_dim, embedding_dim, class_num):
    input_x = tf.placeholder(tf.int32, shape = [None, max_seq_len, max_word_num, max_char_num], name="input_x")
    input_y = tf.placeholder(tf.float32, shape = [None, class_num], name="input_y")
    x_len = tf.placeholder(tf.int32, [None], name="x_len")
    init_states = tf.placeholder(tf.float32, [None, hidden_dim], name="init_states")
    x_reshape = tf.reshape(input_x, [-1, max_word_num, max_char_num])
    print("x_reshape:", x_reshape)
    x_embedding = char_model(x_reshape)
    print("x_embedding:", x_embedding)
    cnn_outs = tf.reshape(x_embedding, [-1, 20, max_word_num, sum(char_model.kernel_features)])
    print("cnn_outs:", cnn_outs)
    words_embedding, sentence_embedding = lm(cnn_outs)
    words_embedding = tf.identity(words_embedding, "rnn_out_puts")
    words_embedding = tf.transpose(words_embedding, [1, 0, 2])
    print("RDM words_embedding:", words_embedding)
    x_senti = senti_model(words_embedding)
    print("x_senti:", x_senti)
    RDM_Input = tf.reshape(x_senti, [-1, max_seq_len, hidden_dim])  

    with tf.variable_scope("Train_RDM", reuse=tf.AUTO_REUSE):
        # fcn_layer = tf.layers.Dense(rdm_model.embedding_dim, activation=tf.keras.activation.sigmoid)
        # x_features = fcn_layer(x_senti)
        # RDM_Input = tf.reshape(x_features, [-1, max_seq_len, max_word_num, hidden_dim])
        df_outputs, df_last_state = rdm_model(RDM_Input, x_len, init_states)
        
        l2_loss = tf.constant(0.0)
        w_ps = tf.Variable(tf.truncated_normal([hidden_dim, class_num], stddev=0.1)) #
        b_ps = tf.Variable(tf.constant(0.01, shape=[class_num])) #
        l2_loss += tf.nn.l2_loss(w_ps) 
        l2_loss += tf.nn.l2_loss(b_ps) 

        pre_scores = tf.nn.xw_plus_b(df_last_state, w_ps, b_ps, name="p_scores")
        predictions = tf.argmax(pre_scores, 1, name="predictions")

        r_outputs = tf.reshape(df_outputs, [-1, hidden_dim]) #[batchsize*max_seq_len, output_dim]
        scores_seq = tf.nn.softmax(tf.nn.xw_plus_b(r_outputs, w_ps, b_ps)) # [batchsize * max_seq_len, class_num] 
        out_seq = tf.reshape(scores_seq, [-1, max_seq_len, class_num], name="out_seq") #[batchsize, max_seq_len, class_num]

        df_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pre_scores, labels=input_y)
        loss = tf.reduce_mean(df_losses) + 0.1 * l2_loss

        correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
    df_global_step = tf.Variable(0, name="global_step", trainable=False)
    df_train_op = tf.train.AdamOptimizer(0.01).minimize(loss, df_global_step)
    return adict(
                dropout_keep_prob = rdm_model.dropout_keep_prob,
                input_x = input_x,
                input_y = input_y,
                x_len = x_len,
                init_states = init_states,
                pre_scores = pre_scores,
                predictions = predictions,
                r_outputs = r_outputs,
                scores_seq = scores_seq,
                out_seq = out_seq,
                df_losses = df_losses,
                loss = loss,
                correct_predictions = correct_predictions,
                accuracy = accuracy,
                df_global_step = df_global_step,
                df_train_op = df_train_op
            )


def InferCMTrainGraph(char_model, senti_model, rdm_model, cm_model, max_word_num, embedding_dim, hidden_dim, action_num):
    rl_state = tf.placeholder(tf.float32, [None, hidden_dim], name="rl_states")
    rl_input = tf.placeholder(tf.float32, [None, max_word_num, embedding_dim], name="rl_input")
    action = tf.placeholder(tf.float32, [None, action_num], name="action")
    reward = tf.placeholder(tf.float32, [None], name="reward")
    
    stopScore, isStop, rl_new_state = cm_model(rdm_model, rl_input, rl_state)

    out_action = tf.reduce_sum(tf.multiply(stopScore, action), reduction_indices=1)
    rl_cost = tf.reduce_mean(tf.square(reward - out_action), name="rl_cost")
    
    rl_global_step = tf.Variable(0, name="global_step", trainable=False)
    rl_train_op = tf.train.AdamOptimizer(0.001).minimize(rl_cost, rl_global_step)
    
    return adict(
            dropout_keep_prob = rdm_model.dropout_keep_prob,
            rl_state = rl_state, 
            rl_input = rl_input,
            action = action,
            reward = reward,  
            rl_new_state = rl_new_state,
            stopScore = stopScore,
            isStop = isStop,
            rl_cost = rl_cost,
            rl_global_step = rl_global_step,
            rl_train_op = rl_train_op
        )


# In[ ]:


def TrainSentiModel(sess, saver, logger, train_model, senti_reader, train_batch, test_batch):
    train_iter = 100
    for t_epoch in range(100): 
        # for validation
        sum_acc = 0.0
        sum_loss = 0.0
        for t_iter in range(train_iter):
            data_X, data_Y = senti_reader.GetTrainingBatch(
                                            t_iter, 
                                            train_batch
                            )
            feed_dic = {
                        train_model.sent_x: data_X, 
                        train_model.sent_y: data_Y
            }
            _, step, loss, acc = sess.run(
                                        [train_model.sent_train_op, 
                                         train_model.sent_global_step, 
                                         train_model.sent_loss, 
                                         train_model.sent_acc], 
                                        feed_dic)
            sum_loss += loss
            sum_acc += acc
            if t_iter % 10 == 9:
                sum_loss = sum_loss / 10
                sum_acc = sum_acc / 10
                ret_acc = sum_acc
                print(get_curtime() + " Step: " + str(step) + " Training loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
                logger.info(get_curtime() + " Step: " + str(step) + " Training loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
                sum_acc = 0.0
                sum_loss = 0.0
        # for validation
        sum_acc = 0.0
        sum_loss = 0.0
        for t_iter in range(100):
            data_X, data_Y = senti_reader.GetTestData(t_iter, test_batch)
            feed_dic = {train_model.sent_x: data_X, train_model.sent_y: data_Y}
            loss, acc = sess.run([train_model.sent_loss, train_model.sent_acc], feed_dic)
            sum_loss += loss
            sum_acc += acc    
        sum_loss = sum_loss / 100
        sum_acc = sum_acc / 100
        ret_acc = sum_acc
        print(get_curtime() + " Step: " + str(step) + " validation loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
        logger.info(get_curtime() + " Step: " + str(step) + " validation loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
        sum_acc = 0.0
        sum_loss = 0.0

        saver.save(sess, "df_saved/sent_model")


# In[ ]:


def TrainRDMModel(sess, logger, mm, t_acc, t_steps, new_data_len=[]):
    sum_loss = 0.0
    sum_acc = 0.0
    ret_acc = 0.0
    init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)

    for i in range(t_steps):
        if len(new_data_len) > 0:
            x, x_len, y = get_df_batch(i, new_data_len)
        else:
            x, x_len, y = get_df_batch(i)
        feed_dic = {mm.input_x: x, mm.x_len: x_len, mm.input_y: y, mm.init_states: init_states, mm.dropout_keep_prob: 0.8}
        _, step, loss, acc = sess.run([mm.df_train_op, mm.df_global_step, mm.loss, mm.accuracy], feed_dic)
        sum_loss += loss
        sum_acc += acc

        if i % 10 == 9:
            sum_loss = sum_loss / 10
            sum_acc = sum_acc / 10
            ret_acc = sum_acc
            print(get_curtime() + " Step: " + str(step) + " Training loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
            logger.info(get_curtime() + " Step: " + str(step) + " Training loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
            if sum_acc > t_acc:
                break
            sum_acc = 0.0
            sum_loss = 0.0

    print(get_curtime() + " Train df Model End.")
    logger.info(get_curtime() + " Train df Model End.")
    return ret_acc


# In[ ]:


def TrainCMModel(sess, logger, rdm_train, cm_train, t_rw, t_steps):
    ids = np.array(range(FLAGS.batch_size), dtype=np.int32)
    seq_states = np.zeros([FLAGS.batch_size], dtype=np.int32)
    isStop = np.zeros([FLAGS.batch_size], dtype=np.int32)
    max_id = FLAGS.batch_size
    init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)
    state = init_states
    D = deque()
    ssq = []
    print("in RL the begining")
    logger.info("in RL the begining")
    # get_new_len(sess, mm)
    data_ID = get_data_ID()
    if len(data_ID) % FLAGS.batch_size == 0: # the total number of events
        flags = int(len(data_ID) / FLAGS.batch_size)
    else:
        flags = int(len(data_ID) / FLAGS.batch_size) + 1
    for i in range(flags):
        x, x_len, y = get_df_batch(i)
        feed_dic = { rdm_train.input_x: x, 
                     rdm_train.x_len: x_len, 
                     rdm_train.input_y: y, 
                     rdm_train.init_states:init_states, 
                     rdm_train.dropout_keep_prob: 1.0 }
        t_ssq = sess.run(rdm_train.out_seq, feed_dic)# t_ssq = [batchsize, max_seq, scores]
        if len(ssq) > 0:
            ssq = np.append(ssq, t_ssq, axis=0)
        else:
            ssq = t_ssq

    print(get_curtime() + " Now Start RL training ...")
    logger.info(get_curtime() + " Now Start RL training ...")
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
            s_state, s_x, s_isStop, s_rw = get_RL_Train_batch(D)
            feed_dic = {
                        cm_train.rl_state: s_state, 
                        cm_train.rl_input: s_x, 
                        cm_train.action: s_isStop, 
                        cm_train.reward: s_rw, 
                        cm_train.dropout_keep_prob: 0.8
            }
            
            _, step = sess.run([rl_train_op, rl_global_step], feed_dic)

        x, y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, 3150)
        batch_dic = {
                     cm_train.rl_state: state, 
                     cm_train.rl_input: x, 
                     cm_train.dropout_keep_prob: 1.0
        }
        
        isStop, mss, mNewState = sess.run(
            [cm_train.isStop, cm_train.stopScore, cm_train.rl_new_state], 
            batch_dic
        )

        for j in range(FLAGS.batch_size):
            if random.random() < FLAGS.random_rate:
                isStop[j] = np.argmax(np.random.rand(2))
            if seq_states[j] == data_len[ids[j]]:
                isStop[j] = 1

        # eval
        rw = get_reward(isStop, mss, ssq, ids, seq_states)

        for j in range(FLAGS.batch_size):
            D.append((state[j], x[j], isStop[j], rw[j]))
            if len(D) > FLAGS.max_memory:
                D.popleft()

        state = mNewState
        for j in range(FLAGS.batch_size):
            if isStop[j] == 1:
                # init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)
                # feed_dic = {rl_model.init_states: init_states}
                # state[j] = sess.run(rl_model.df_state, feed_dic)
                state[j] = np.zeros([FLAGS.hidden_dim], dtype=np.float32)
        counter += 1

