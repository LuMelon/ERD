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


logger = MyLogger("SentiCNNModel")


# In[2]:


# load twitter data
# load_data(FLAGS.data_file_path)
load_data_fast()

#load PTB data
# word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
#     PTB_data_reader.load_data(FLAGS.data_dir, FLAGS.max_word_length, char_vocab, eos=FLAGS.EOS)
word_vocab, char_vocab, word_tensors, char_tensors =     PTB_data_reader.load_data_fast()
max_word_length = FLAGS.max_word_length
train_reader = PTB_data_reader.DataReader(word_tensors['train'], char_tensors['train'],
                          FLAGS.batch_size, FLAGS.max_sent_len) 

#load sentiment analysis data
sentiReader = dataloader.SentiDataLoader(
                                        dirpath = '/home/hadoop/trainingandtestdata',
                                        trainfile = 'training.1600000.processed.noemoticon.csv', 
                                        testfile = 'testdata.manual.2009.06.14.csv', 
                                        charVocab = char_vocab
                        )
# sentiReader.load_data()
sentiReader.load_data_fast(
                        '/home/hadoop/ERD/data/senti_train_data.pickle',
                        '/home/hadoop/ERD/data/senti_train_label.pickle',
                        '/home/hadoop/ERD/data/senti_test_data.pickle',
                        '/home/hadoop/ERD/data/senti_test_label.pickle'
                          )


# (self, input_dim, hidden_dim, max_seq_len, max_word_num, class_num, action_num):
print(  FLAGS.embedding_dim, FLAGS.hidden_dim, 
            FLAGS.max_seq_len, FLAGS.max_sent_len, 
                FLAGS.class_num, FLAGS.action_num   )
logger.info(    (FLAGS.embedding_dim, FLAGS.hidden_dim, 
                    FLAGS.max_seq_len, FLAGS.max_sent_len, 
                        FLAGS.class_num, FLAGS.action_num)  )

print(get_curtime() + " Data loaded.")
logger.info(get_curtime() + " Data loaded.")


# In[3]:


# # save the Twitter data
# data = get_data()
# with open('data/data_dict.txt', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # save the PTB data
# with open('data/char_tensors.txt', 'wb') as handle:
#     pickle.dump(char_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/word_tensors.txt', 'wb') as handle:
#     pickle.dump(word_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('data/char_vocab.txt', 'wb') as handle:
#     pickle.dump(char_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/word_vocab.txt', 'wb') as handle:
#     pickle.dump(word_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save the senti data
# with open('data/senti_train_data.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/senti_train_label.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.train_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('data/senti_test_data.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/senti_test_label.pickle', 'wb') as handle:
#     pickle.dump(sentiReader.test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[9]:


def Train_Char_Model(session, train_model, train_reader, saver, summary_writer):
    best_valid_loss = None
    rnn_state = session.run(train_model.initial_rnn_state)
    cnt = 0
    for epoch in range(FLAGS.max_epochs):
#     for epoch in range(1):
        epoch_start_time = time.time()
        avg_train_loss = 0.0
        count = 0
        for x, y in train_reader.iter():
            count += 1
            start_time = time.time()

            loss, _, rnn_state, gradient_norm, step, _ = session.run([
                train_model.loss,
                train_model.train_op,
                train_model.final_rnn_state,
                train_model.global_norm,
                train_model.global_step,
                train_model.clear_char_embedding_padding
            ], {
                train_model.input: x,
                train_model.targets: y,
                train_model.drop_out: 0.8,
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
        print('Epoch training time:', time.time()-epoch_start_time)
        save_as = '%s/epoch%03d_%.4f.model' % ("lstmCharCNNModel/", epoch, avg_train_loss)
        saver.save(session, save_as)
        print('Saved char model', save_as)

def TrainSentiModel(sess, saver, logger, summary_writer, train_model, senti_reader, train_batch, test_batch):
    train_iter = int(len(senti_reader.train_label)/train_batch)+1
    test_iter = int(len(senti_reader.test_label)/test_batch)+1
    print("train_iter:", train_iter, "| test_iter:", test_iter)
    rnn_state = sess.run(train_model.initial_rnn_state)
    for t_epoch in range(10): 
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
                        train_model.drop_out: 0.8,
                        train_model.sent_y: data_Y,
                        train_model.initial_rnn_state:rnn_state
            }
            _, step, loss, acc = sess.run(
                                        [train_model.sent_train_op, 
                                         train_model.sent_global_step, 
                                         train_model.sent_loss, 
                                         train_model.sent_acc], 
                                        feed_dic)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="step_train_loss", simple_value=loss),
                tf.Summary.Value(tag="step_train_acc", simple_value=acc),
            ])
            summary_writer.add_summary(summary, step)
            
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
                if t_iter % 100 ==99:
                    for t_iter in range(5):
                        data_X, data_Y = senti_reader.GetTestData(t_iter, test_batch)
                        feed_dic = {
                                    train_model.sent_x: data_X, 
                                    train_model.drop_out: 1.0,
                                    train_model.sent_y: data_Y,
                                    train_model.initial_rnn_state:rnn_state
                        }
                        loss, acc = sess.run([train_model.sent_loss, train_model.sent_acc], feed_dic)
                        sum_loss += loss
                        sum_acc += acc    
                    sum_loss = sum_loss / (5.0)
                    sum_acc = sum_acc / (5.0)
                    ret_acc = sum_acc
                    print(get_curtime() + " Step: " + str(step) + 
                              " validation loss: " + str(sum_loss) + 
                                  " accuracy: " + str(sum_acc))
                    logger.info(get_curtime() + " Step: " + str(step) +
                                " validation loss: " + str(sum_loss) + 
                                " accuracy: " + str(sum_acc))

                    summary = tf.Summary(value=[
                            tf.Summary.Value(tag="step_valid_loss", simple_value=sum_loss),
                            tf.Summary.Value(tag="step_valid_acc", simple_value=sum_acc),
                        ])
                    summary_writer.add_summary(summary, t_epoch)
                    sum_acc = 0.0
                    sum_loss = 0.0
                    
            if step % 1000 ==999:
                saver.save(sess, "SentiCNNModel/")
                # for validation
                sum_acc = 0.0
                sum_loss = 0.0
                for t_iter in range(5, test_iter, 1):
                    data_X, data_Y = senti_reader.GetTestData(t_iter, test_batch)
                    feed_dic = {
                                train_model.sent_x: data_X, 
                                train_model.drop_out: 1.0,
                                train_model.sent_y: data_Y,
                                train_model.initial_rnn_state:rnn_state
                    }
                    loss, acc = sess.run([train_model.sent_loss, train_model.sent_acc], feed_dic)
                    sum_loss += loss
                    sum_acc += acc    
                sum_loss = sum_loss / ((test_iter-5)*1.0)
                sum_acc = sum_acc / ((test_iter-5)*1.0)
                ret_acc = sum_acc
                print(get_curtime() + " Step: " + str(step) + 
                          " test loss: " + str(sum_loss) + 
                              " test accuracy: " + str(sum_acc))
                logger.info(get_curtime() + " Step: " + str(step) +
                            " test loss: " + str(sum_loss) + 
                            " test accuracy: " + str(sum_acc))

                summary = tf.Summary(value=[
                        tf.Summary.Value(tag="step_test_loss", simple_value=sum_loss),
                        tf.Summary.Value(tag="step_test_acc", simple_value=sum_acc),
                    ])
                summary_writer.add_summary(summary, t_epoch)
                
# In[8]:


# reuse model
with tf.Graph().as_default() as g:
    with tf.Session(graph=g) as sess:
        w2v = lstm_char_cnn.WordEmbedding(
                        max_word_length = FLAGS.max_char_num , 
                        char_vocab_size = char_vocab.size, 
                        char_embed_size = FLAGS.char_embed_size, 
                        kernels = eval(FLAGS.kernels), 
                        kernel_features = eval(FLAGS.kernel_features), 
                        num_highway_layers = FLAGS.highway_layers,
                        embedding_dim = FLAGS.embedding_dim
                    )
        lstm_lm = lstm_char_cnn.LSTM_LM(
                    batch_size = FLAGS.batch_size, 
                    num_unroll_steps = FLAGS.max_sent_len, 
                    rnn_size = FLAGS.embedding_dim, 
                    num_rnn_layers = FLAGS.rnn_layers, 
                    word_vocab_size = word_vocab.size
                )

        char_train_graph = lstm_char_cnn.infer_train_model(
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
        checkpoint = tf.train.get_checkpoint_state("lstmCharCNNModel/")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
#         Train_Char_Model(sess, char_train_graph, train_reader, saver, summary_writer)
        #sentiment analysis model
        s_model = model.SentiModel(FLAGS.hidden_dim, 5)
        senti_train_graph = model.InferSentiTrainGraph(
                                w2v, 
                                lstm_lm, 
                                s_model, 
                                batchsize=20,
                                max_word_num = sentiReader.max_sent_len, 
                                max_char_num = FLAGS.max_char_num, 
                                hidden_dim = FLAGS.hidden_dim, 
                                sent_num = FLAGS.sent_num,
                                embedding_dim = FLAGS.embedding_dim
                            )
        val_list2 = tf.global_variables()
        saver2 = tf.train.Saver(val_list2, max_to_keep=4)
        
        uninitialized_vars = list( filter(lambda var: var not in val_list1, val_list2) )
        sess.run(tf.variables_initializer(uninitialized_vars))
        summary_writer = tf.summary.FileWriter("SentiCNNModel/", graph=sess.graph)
        TrainSentiModel(sess, saver, logger, summary_writer, senti_train_graph, sentiReader, 20, 20)
        


# In[ ]:




