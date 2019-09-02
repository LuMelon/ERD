#!/usr/bin/env python
# coding: utf-8

# In[1]:


import config
import tensorflow as tf
tf.app.flags.DEFINE_string('f', '', 'kernel')
from collections import deque
import model
from dataUtils import *
from logger import MyLogger
import sys
import PTB_data_reader
import time
import numpy as np
import lstm_char_cnn
import pickle
import dataloader
tf.logging.set_verbosity(tf.logging.ERROR)


logger = MyLogger("MTLCharMain")


# In[2]:


# load twitter data
# load_data(FLAGS.data_file_path)
load_data_fast()

#load PTB data
# word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
#     PTB_data_reader.load_data(FLAGS.data_dir, FLAGS.max_word_length, char_vocab, eos=FLAGS.EOS)
word_vocab, char_vocab, word_tensors, char_tensors, word_len =     PTB_data_reader.load_data_fast()
max_word_length = FLAGS.max_word_length
train_reader = PTB_data_reader.DataReader(word_tensors['train'], char_tensors['train'], word_len['train'],
                          FLAGS.batch_size, FLAGS.max_sent_len) 
valid_reader = PTB_data_reader.DataReader(word_tensors['valid'], char_tensors['valid'], word_len['valid'],
                          FLAGS.batch_size, FLAGS.max_sent_len) 
test_reader = PTB_data_reader.DataReader(word_tensors['test'], char_tensors['test'], word_len['test'],
                          FLAGS.batch_size, FLAGS.max_sent_len) 
#load sentiment analysis data
# sentiReader = dataloader.SentiDataLoader(
#                                         dirpath = '/home/hadoop/trainingandtestdata',
#                                         trainfile = 'training.1600000.processed.noemoticon.csv', 
#                                         testfile = 'testdata.manual.2009.06.14.csv', 
#                                         charVocab = char_vocab
#                         )
# # sentiReader.load_data()
# sentiReader.load_data_fast(
#                         '/home/hadoop/ERD/data/senti_train_data.pickle',
#                         '/home/hadoop/ERD/data/senti_train_label.pickle',
#                         '/home/hadoop/ERD/data/senti_test_data.pickle',
#                         '/home/hadoop/ERD/data/senti_test_label.pickle'
#                           )


# (self, input_dim, hidden_dim, max_seq_len, max_word_num, class_num, action_num):
print(  FLAGS.embedding_dim, FLAGS.hidden_dim, 
            FLAGS.max_seq_len, FLAGS.max_sent_len, 
                FLAGS.class_num, FLAGS.action_num   )
logger.info(    (FLAGS.embedding_dim, FLAGS.hidden_dim, 
                    FLAGS.max_seq_len, FLAGS.max_sent_len, 
                        FLAGS.class_num, FLAGS.action_num)  )

print(get_curtime() + " Data loaded.")
logger.info(get_curtime() + " Data loaded.")


# In[7]:


# # save the Twitter data
# data = get_data()
# with open('data/data_dict.txt', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save the PTB data
# with open('data/char_tensors.txt', 'wb') as handle:
#     pickle.dump(char_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/word_tensors.txt', 'wb') as handle:
#     pickle.dump(word_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('data/char_vocab.txt', 'wb') as handle:
#     pickle.dump(char_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/word_vocab.txt', 'wb') as handle:
#     pickle.dump(word_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/word_len.txt', 'wb') as handle:
#     pickle.dump(x_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
# save the senti data
# with open('data/senti_train_data.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/senti_train_label.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.train_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('data/senti_test_data.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/senti_test_label.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[5]:


import importlib


# In[6]:


importlib.reload(model)


# In[7]:


from model import adict


# In[8]:


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear", reuse=tf.AUTO_REUSE):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


# In[9]:


class LSTMCharNet:
    def __init__(self, max_word_length, char_vocab_size, 
                        char_embed_size, embedding_dim):
        self.max_word_length = max_word_length
        self.char_vocab_size = char_vocab_size
        self.char_embed_size = char_embed_size
        self.embedding_dim = embedding_dim
        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            self.char_embedding = tf.get_variable('char_embedding', [self.char_vocab_size, self.char_embed_size])
            ''' this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
            of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
            zero embedding vector and ignores gradient updates. For that do the following in TF:
            1. after parameter initialization, apply this op to zero out padding embedding vector
            2. after each gradient update, apply this op to keep padding at zero'''
            self.clear_char_embedding_padding = tf.scatter_update(self.char_embedding, [0], tf.constant(0.0, shape=[1, self.char_embed_size]))
            self.drop_out_prob_keep = tf.placeholder(tf.float32, name="lstm_char_net_dp")
            self.fw_cell = self.create_rnn_cell(self.embedding_dim)
            self.bw_cell = self.create_rnn_cell(self.embedding_dim)

    def __call__(self, input_words, x_len, fw_init, bw_init):
        input_ = input_words
        print("input_:", input_)
        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            input_embedded = tf.nn.embedding_lookup(self.char_embedding, input_)
            print("input_embedded1:", input_embedded)
            input_embedded = tf.reshape(input_embedded, [-1, self.max_word_length, self.char_embed_size])
            print("input_embedded2:", input_embedded)
            (fw_outs, bw_outs), (fw_final, bw_final) = tf.nn.bidirectional_dynamic_rnn(
                                        self.fw_cell,
                                        self.bw_cell,
                                        input_embedded,
                                        sequence_length=x_len,
                                        initial_state_fw=fw_init,
                                        initial_state_bw=bw_init,
                                        dtype=None,
                                        parallel_iterations=None,
                                        swap_memory=False,
                                        time_major=False,
                                        scope=None
                                    )
        return fw_outs, bw_outs, fw_final, bw_final

    def create_rnn_cell(self, rnn_size):
        cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.drop_out_prob_keep)
        return cell


# In[10]:


class LSTM_LM:
    def __init__(self, batch_size, num_unroll_steps, rnn_size, num_rnn_layers, word_vocab_size):
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.word_vocab_size = word_vocab_size
        with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
            self.drop_out = tf.placeholder(tf.float32, name="Dropout")
            def create_rnn_cell():
                cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.drop_out)
                return cell
            if self.num_rnn_layers > 1:
                self.cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(self.num_rnn_layers)], state_is_tuple=True)
            else:
                self.cell = create_rnn_cell()
            self.initial_rnn_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
            
    def __call__(self, input_cnn):
        with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
            input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, self.num_unroll_steps, 1)]
            outputs, final_rnn_state = tf.contrib.rnn.static_rnn(self.cell, input_cnn2,
                                             initial_state=self.initial_rnn_state, dtype=tf.float32)     
            return outputs, final_rnn_state


# In[11]:


def infer_train_model(char_rnn_net, LM, 
                      batch_size, 
                      num_unroll_steps, 
                      max_word_length, 
                      learning_rate,
                      max_grad_norm, 
                     ):
    input_ = tf.placeholder(tf.int32, shape=[batch_size, num_unroll_steps, max_word_length], name="input")
    targets = tf.placeholder(tf.int64, [batch_size, num_unroll_steps], name='targets')
    input_len = tf.placeholder(tf.int32, shape = [batch_size, num_unroll_steps]) 
    
    # input_cnn = word2vec(input_) #[batch_size*num_unroll_steps, k_features]
    rnn_len = tf.reshape(input_len, shape = [batch_size*num_unroll_steps])
    fw_init = char_rnn_net.fw_cell.zero_state(batch_size*num_unroll_steps, dtype=tf.float32)
    bw_init = char_rnn_net.bw_cell.zero_state(batch_size*num_unroll_steps, dtype=tf.float32)
    print("bw_init:", bw_init)
    fw_outs, bw_outs, fw_final, bw_final = char_rnn_net(input_, rnn_len, fw_init, bw_init)
    print("fw_out:", fw_outs)
    fw_final = fw_final[-1]
    bw_final = bw_final[-1]
    
    # add to final tensor
    out_merge = fw_final + bw_final
    
#     # concat two tensor
#     out_merge = tf.concat([fw_final, bw_final], axis = -1)
    
#     # max_pooling two tensor
#     out_merge = tf.reduce_max(
#         tf.transpose(
#             tf.identity([fw_final, bw_final]),
#             [1, 0, 2]
#         ),
#         axis = 1
#     )
    
    input_cnn = tf.reshape(out_merge, [batch_size, num_unroll_steps, -1])
    outputs, final_rnn_state = LM(input_cnn)
    
    # linear projection onto output (word) vocab
    logits = []
    with tf.variable_scope('WordEmbedding') as scope:
        for idx, output in enumerate(outputs):
            if idx > 0:
                scope.reuse_variables()
            logits.append(linear(output, LM.word_vocab_size))

    word_embedding = tf.identity(outputs, "lstm_word_embedding")
    sent_embedding = tf.identity(final_rnn_state[-1][-1], "lstm_sent_embedding")

    with tf.variable_scope('Loss', reuse=tf.AUTO_REUSE):
            target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, num_unroll_steps, 1)]
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = target_list), name='loss')
            
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('SGD_Training'):
        # SGD learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        # collect all trainable variables
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        
    return adict(
        input = input_,
        input_len = input_len, 
        drop_out_char_lstm = char_rnn_net.drop_out_prob_keep,
        drop_out_lm = LM.drop_out,
        fw_init = fw_init,
        bw_init = bw_init,
        initial_rnn_state=LM.initial_rnn_state,
        final_rnn_state=final_rnn_state,
        rnn_outputs=outputs,
        logits = logits,
        targets=targets,
        loss=loss,
        learning_rate=learning_rate,
        global_step=global_step,
        global_norm=global_norm,
        train_op=train_op
    )


# In[13]:


def Validation(session, train_model, valid_reader, summary_writer, fw_init, bw_init, rnn_state, epoch):
    sum_loss = 0
    cnt = 0
    for x, y, x_len in valid_reader.iter():
        start_time = time.time()
        loss, rnn_state = session.run([
            train_model.loss,
            train_model.final_rnn_state
        ], {
            train_model.input: x,
            train_model.targets: y,
            train_model.input_len: x_len, 
            train_model.drop_out_char_lstm: 1.0, 
            train_model.drop_out_lm: 1.0,
            train_model.fw_init: fw_init,
            train_model.bw_init: bw_init,
            train_model.initial_rnn_state: rnn_state
        })
        sum_loss += loss
        cnt += 1
    sum_loss =  sum_loss/(1.0*cnt)
    summary = tf.Summary(value=[
        tf.Summary.Value(tag="epoch_valid_loss", simple_value=sum_loss),
        tf.Summary.Value(tag="epoch_valid_perplexity", simple_value=np.exp(sum_loss)),
    ])
    summary_writer.add_summary(summary, epoch)
    print("Valid loss:", sum_loss, ", | Valid perplexity:", np.exp(sum_loss))

def Test(session, train_model, test_reader, summary_writer, fw_init, bw_init, rnn_state):
    sum_loss = 0
    cnt = 0
    for x, y, x_len in test_reader.iter():
        start_time = time.time()
        loss, rnn_state = session.run([
            train_model.loss,
            train_model.final_rnn_state
        ], {
            train_model.input: x,
            train_model.targets: y,
            train_model.input_len: x_len, 
            train_model.drop_out_char_lstm: 1.0, 
            train_model.drop_out_lm: 1.0,
            train_model.fw_init: fw_init,
            train_model.bw_init: bw_init,
            train_model.initial_rnn_state: rnn_state
        })
        sum_loss += loss
        cnt += 1
    sum_loss = sum_loss/(1.0*cnt)
    summary = tf.Summary(value=[
        tf.Summary.Value(tag="test_loss", simple_value=sum_loss),
        tf.Summary.Value(tag="test_perplexity", simple_value=np.exp(sum_loss)),
    ])
    summary_writer.add_summary(summary, 0)
    print("Test loss:", sum_loss, ", | Test perplexity:", np.exp(sum_loss))
    
def Train_Char_Model(session, train_model, train_reader, valid_reader, test_reader,  saver, summary_writer):
    best_valid_loss = None
    rnn_state = session.run(train_model.initial_rnn_state)
    fw_init =  session.run(train_model.fw_init)
    bw_init = session.run(train_model.bw_init)
    for epoch in range(FLAGS.max_epochs):
    # for epoch in range(1):
        epoch_start_time = time.time()
        avg_train_loss = 0.0
        count = 0
        for x, y, x_len in train_reader.iter():
            count += 1
            start_time = time.time()

            loss, _, rnn_state, gradient_norm, step = session.run([
                train_model.loss,
                train_model.train_op,
                train_model.final_rnn_state,
                train_model.global_norm,
                train_model.global_step
            ], {
                train_model.input: x,
                train_model.targets: y,
                train_model.input_len: x_len, 
                train_model.drop_out_char_lstm:0.8, 
                train_model.drop_out_lm: 0.8,
                train_model.fw_init: fw_init,
                train_model.bw_init: bw_init,
                train_model.initial_rnn_state: rnn_state
            })

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="step_train_loss", simple_value=loss),
                tf.Summary.Value(tag="step_train_perplexity", simple_value=np.exp(loss)),
            ])
            summary_writer.add_summary(summary, step)

            avg_train_loss += 0.05 * (loss - avg_train_loss)

            time_elapsed = time.time() - start_time

            if count % FLAGS.print_every == 0:
                print('%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f secs/batch = %.4fs, grad.norm=%6.8f' % (step,
                                                        epoch, count,
                                                        train_reader.length,
                                                        loss, np.exp(loss),
                                                        time_elapsed,
                                                        gradient_norm))
        Validation(session, train_model, valid_reader, summary_writer, fw_init, bw_init, rnn_state, epoch)
        print('Epoch training time:', time.time()-epoch_start_time)
        save_as = '%s/epoch%03d_%.4f.model' % (FLAGS.train_dir, epoch, avg_train_loss)
        saver.save(session, save_as)
        print('Saved char model', save_as)
    Test(session, train_model, test_reader, summary_writer, fw_init, bw_init, rnn_state)


# In[15]:


# reuse model to train senti model
gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4
# device = '/GPU:0'
with tf.Graph().as_default() as g:
    with tf.Session(graph=g, config=gpu_config) as sess:
        with tf.device('/GPU:0'):
            w2v = LSTMCharNet(
                            max_word_length = FLAGS.max_char_num, 
                            char_vocab_size = char_vocab.size, 
                            char_embed_size = FLAGS.char_embed_size,
                            embedding_dim = FLAGS.embedding_dim
                        )
            lstm_lm = LSTM_LM(
                        batch_size = FLAGS.batch_size, 
                        num_unroll_steps = FLAGS.max_sent_len, 
                        rnn_size = FLAGS.embedding_dim, 
                        num_rnn_layers = FLAGS.rnn_layers, 
                        word_vocab_size = word_vocab.size
                    )

            char_train_graph = infer_train_model(
                                w2v, lstm_lm, 
                                batch_size = FLAGS.batch_size, 
                                num_unroll_steps = FLAGS.max_sent_len, 
                                max_word_length = FLAGS.max_char_num, 
                                learning_rate = FLAGS.learning_rate,
                                max_grad_norm = FLAGS.max_grad_norm
                             )
            val_list1 = tf.global_variables()
            saver = tf.train.Saver(val_list1, max_to_keep=4)
            sess.run(tf.variables_initializer(val_list1))
            summary_writer = tf.summary.FileWriter("lstm_word_LM/", graph=sess.graph)
        Train_Char_Model(sess, char_train_graph, train_reader, valid_reader, test_reader, saver, summary_writer)
        


# In[ ]:




