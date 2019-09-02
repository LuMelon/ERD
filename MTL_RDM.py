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


logger = MyLogger("RDMGPUTrain")

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

# (self, input_dim, hidden_dim, max_seq_len, max_word_num, class_num, action_num):
print(  FLAGS.embedding_dim, FLAGS.hidden_dim, 
            FLAGS.max_seq_len, FLAGS.max_sent_len, 
                FLAGS.class_num, FLAGS.action_num   )
logger.info(    (FLAGS.embedding_dim, FLAGS.hidden_dim, 
                    FLAGS.max_seq_len, FLAGS.max_sent_len, 
                        FLAGS.class_num, FLAGS.action_num)  )

print(get_curtime() + " Data loaded.")
logger.info(get_curtime() + " Data loaded.")

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



from model import adict


def shared_pooling_layer(inputs, input_dim, max_seq_len, max_word_len, output_dim, scope="shared_pooling_layer"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        t_inputs = tf.reshape(inputs, [-1, input_dim])
        mix_layer = tf.layers.Dense(output_dim)
        t_h = mix_layer(t_inputs)
        t_h = tf.reshape(t_h, [-1, max_word_len, output_dim])
        t_h_expended = tf.expand_dims(t_h, -1)
        pooled = tf.nn.max_pool(
            t_h_expended,
            ksize=[1, max_word_len, 1, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="max_pool"
        )
    outs = tf.reshape(pooled, [-1, max_seq_len, output_dim])
    return outs


def InferRDMTrainGraph(char_model, lm, senti_model, rdm_model, batchsize,
                            max_seq_len, max_word_num, max_char_num, 
                                hidden_dim, embedding_dim, class_num):
    input_x = tf.placeholder(
                        tf.int32, 
                        shape = [
                                 batchsize, 
                                 max_seq_len, 
                                 max_word_num, 
                                 max_char_num
                                 ], 
                        name="input_x"
                    )
    input_y = tf.placeholder(
                        tf.float32, 
                        shape = [batchsize, class_num], 
                        name="input_y"
                    )
    x_len = tf.placeholder(
                        tf.int32, 
                        [batchsize], 
                        name="x_len"
                    )
    init_states = tf.placeholder(
                        tf.float32, 
                        [batchsize, hidden_dim], 
                        name="init_states"
                    )
    x_reshape = tf.reshape(
                        input_x, 
                        [
                         batchsize*max_seq_len, 
                         max_word_num, 
                         max_char_num
                        ]
                    )
    print("x_reshape:", x_reshape)
    x_embedding = char_model(x_reshape)
    print("x_embedding:", x_embedding)
    cnn_outs = tf.reshape(
                        x_embedding, 
                        [
                         batchsize*max_seq_len, 
                         max_word_num, 
                         sum(char_model.kernel_features)
                        ]
                    )
    print("cnn_outs:", cnn_outs)
    # words_embedding, sentence_embedding = lm(cnn_outs)
    cnn_outs_list = [tf.squeeze(x, [1]) 
    for x in tf.split(cnn_outs, max_word_num, 1)]
    rdm_init_state = lm.cell.zero_state(
                            batchsize*max_seq_len, 
                            dtype=tf.float32
                        )
    words_embedding, sentence_embedding = tf.contrib.rnn.static_rnn(
                                        lm.cell, 
                                        cnn_outs_list,
                                        initial_state=rdm_init_state, 
                                        dtype=tf.float32
                                    )     
    words_embedding = tf.identity(words_embedding, 
                                    "rnn_out_puts") #[max_word_num, batchsize*max_seq_len, embedding_dim]
    words_embedding = tf.transpose(words_embedding, 
                                        [1, 0, 2]) #[batchsize*max_seq_len, max_word_num, embedding_dim]
    words_embedding = tf.reshape(words_embedding, 
                                 shape=[
                                  batchsize,
                                  max_seq_len,
                                  max_word_num,
                                  embedding_dim
                                 ]
                                )
    print("RDM words_embedding:", words_embedding)
#     x_senti = senti_model(words_embedding)
#     words_feature = tf.math.reduce_max( words_embedding , axis=1)
    
    with tf.variable_scope("Train_RDM", reuse=tf.AUTO_REUSE):
#         fcn_layer = tf.layers.Dense(hidden_dim, activation=tf.compat.v1.keras.activations.sigmoid)
#         x_senti =  fcn_layer(sentence_embedding[-1][-1] + words_feature )
        x_senti = shared_pooling_layer(inputs = words_embedding, 
                                       input_dim = embedding_dim, 
                                       max_seq_len = max_seq_len, 
                                       max_word_len = max_word_num, 
                                       output_dim = hidden_dim 
                                      )
        print("x_senti:", x_senti)
        RDM_Input = tf.reshape(
                            x_senti, 
                            [
                             batchsize, 
                             max_seq_len, 
                             hidden_dim
                            ]
                        )  
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
        
    return adict(
                lm_drop_out = lm.drop_out,
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
                accuracy = accuracy
            )


# In[ ]:





# In[2]:


def TrainRDMModel(sess, saver, summary_writter, logger, mm, batch_size, t_acc, t_steps, model_dir, new_data_len=[]):
    sum_loss = 0.0
    sum_acc = 0.0
    ret_acc = 0.0
    init_states = np.zeros([batch_size, FLAGS.hidden_dim], dtype=np.float32)

    for i in range(t_steps):
        if len(new_data_len) > 0:
            x, x_len, y = get_df_batch(i, batch_size, new_data_len)
        else:
            x, x_len, y = get_df_batch(i, batch_size)
        feed_dic = {
                        mm.input_x: x, 
                        mm.x_len: x_len, 
                        mm.input_y: y, 
                        mm.init_states: init_states, 
                        mm.dropout_keep_prob: 0.8,
                        mm.lm_drop_out: 0.8
        }
        _, step, loss, acc = sess.run([mm.df_train_op, mm.df_global_step, mm.loss, mm.accuracy], feed_dic)
        
        summary = tf.Summary(value=[
                tf.Summary.Value(tag="step_train_loss", simple_value=loss),
                tf.Summary.Value(tag="step_train_acc", simple_value=acc),
            ])
        
        summary_writer.add_summary(summary, step)    
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
            if i % 1000 == 999:
                save_as = '%s/epoch%03d_%.4f.model' % (model_dir, int(i/1000), sum_loss)
                saver.save(sess, save_as)
                print('Saved char model', save_as)
            sum_acc = 0.0
            sum_loss = 0.0
    print(get_curtime() + " Train df Model End.")
    logger.info(get_curtime() + " Train df Model End.")
    return ret_acc        


# In[3]:


# reuse model to train RDMModel
gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
device = "/CPU:0"
# device = "/GPU:0"
with tf.Graph().as_default() as g:
    with tf.Session(graph=g, config=gpu_config) as sess:
        with tf.device('/GPU:0'):
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
#             s_model = model.SentiModel(FLAGS.hidden_dim, 5)
#             senti_train_graph = model.InferSentiTrainGraph(
#                                     w2v, 
#                                     lstm_lm, 
#                                     s_model, 
#                                     batchsize=20,
#                                     max_word_num = sentiReader.max_sent_len, 
#                                     max_char_num = FLAGS.max_char_num, 
#                                     hidden_dim = FLAGS.hidden_dim, 
#                                     sent_num = FLAGS.sent_num,
#                                     embedding_dim = FLAGS.embedding_dim
#                                 )
            val_list1 = tf.global_variables()
            saver = tf.train.Saver(val_list1, max_to_keep=4)
            sess.run(tf.variables_initializer(val_list1))
            checkpoint = tf.train.get_checkpoint_state("lstmCharCNNModel/")
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
            #RDMModel
            rdm_model = model.RDM_Model(
                    max_seq_len = FLAGS.max_seq_len, 
                    max_word_num = FLAGS.max_sent_len, 
                    embedding_dim = FLAGS.embedding_dim, 
                    hidden_dim = FLAGS.hidden_dim
                )
            rdm_train_graph = InferRDMTrainGraph(
                            w2v, lstm_lm, None, rdm_model, 
                            batchsize=20,
                            max_seq_len = FLAGS.max_seq_len, 
                            max_word_num = FLAGS.max_sent_len, 
                            max_char_num = FLAGS.max_char_num, 
                            hidden_dim = FLAGS.hidden_dim, 
                            embedding_dim = FLAGS.embedding_dim,
                            class_num = FLAGS.class_num
                    )
            val_list2 = tf.global_variables()
            rdm_vars = list( filter(lambda var: var not in val_list1, val_list2) )
            df_global_step = tf.Variable(0, name="global_step", trainable=False)
#             df_train_op = tf.train.MomentumOptimizer(0.05, 0.9).minimize(rdm_train_graph.loss, df_global_step, var_list = rdm_vars)
            df_train_op = tf.train.AdagradOptimizer(0.05).minimize(rdm_train_graph.loss, df_global_step, var_list = rdm_vars)
            rdm_train_graph.update(
                adict(
                    df_global_step = df_global_step,
                    df_train_op = df_train_op
                )
            )
            val_list3 = tf.global_variables()
            saver2 = tf.train.Saver(val_list3, max_to_keep=4)
            uninitialized_vars = list( filter(lambda var: var not in val_list1, val_list3) )
#             print("uninitialized_vars:", uninitialized_vars)
            sess.run(tf.variables_initializer(uninitialized_vars))
            sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("RDMGPUTrain/", graph=sess.graph)
        TrainRDMModel(sess, saver, summary_writer, logger, rdm_train_graph, 20, 0.97, 100000, "RDMGPUTrain/", new_data_len=[])


# In[ ]:




