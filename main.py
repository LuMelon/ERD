# coding: utf-8
from collections import deque
import model
from dataUtils import *
from logger import MyLogger
import sys
import PTB_data_reader
import time
import numpy as np
import lstm_char_cnn
import config
import dataloader

tf.logging.set_verbosity(tf.logging.ERROR)

logger = MyLogger("ERDMain")

def df_train(sess, charm, mm, char_input, t_acc, t_steps, new_data_len=[]):
    sum_loss = 0.0
    sum_acc = 0.0
    ret_acc = 0.0
    init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)

    for i in range(t_steps):
        if len(new_data_len) > 0:
            x, x_len, y = get_df_batch(i, new_data_len)
        else:
            x, x_len, y = get_df_batch(i)
        feed_dic = {mm.input_x: x, mm.x_len: x_len, mm.input_y: y, mm.init_states: init_states, mm.dropout_keep_prob: 0.5}
        _, step, loss, acc = sess.run([df_train_op, df_global_step, mm.loss, mm.accuracy], feed_dic)
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


def rl_train(sess, df_model, rl_model, t_rw, t_steps):
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
        feed_dic = {df_model.input_x: x, df_model.x_len: x_len, df_model.input_y: y, df_model.init_states:init_states, df_model.dropout_keep_prob: 1.0}
        t_ssq = sess.run(df_model.out_seq, feed_dic)# t_ssq = [batchsize, max_seq, scores]
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
                print(get_curtime() + " Step: " + str(step) + " REWARD IS " + str(sum_rw))
                logger.info(get_curtime() + " Step: " + str(step) + " REWARD IS " + str(sum_rw))
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
            feed_dic = {rl_model.rl_state: s_state, rl_model.rl_input: s_x, rl_model.action: s_isStop, rl_model.reward:s_rw, rl_model.dropout_keep_prob: 0.5}
            _, step = sess.run([rl_train_op, rl_global_step], feed_dic)

        x, y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, 3150)
        batch_dic = {rl_model.rl_state: state, rl_model.rl_input: x, rl_model.dropout_keep_prob: 1.0}
        isStop, mss, mNewState = sess.run([rl_model.isStop, rl_model.stopScore, rl_model.rl_new_state], batch_dic)

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


# def eval(sess, mm):
#     start_ef = int(eval_flag / FLAGS.batch_size)
#     end_ef = int(len(data_ID) / FLAGS.batch_size) + 1
#     init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)

#     counter = 0
#     sum_acc = 0.0

#     for i in range(start_ef, end_ef):
#         x, x_len, y = get_df_batch(i)
#         feed_dic = {mm.input_x: x, mm.x_len: x_len, mm.input_y: y, mm.init_states: init_states, mm.dropout_keep_prob: 1.0}
#         _, step, loss, acc = sess.run([df_train_op, df_global_step, mm.loss, mm.accuracy], feed_dic)
#         counter += 1
#         sum_acc += acc

#     print(sum_acc / counter)

if __name__ == "__main__":
    print(get_curtime() + " Loading data ...")
    logger.info(get_curtime() + " Loading data ...")

    #load PTB data
    # word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
    #     PTB_data_reader.load_data(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS)
    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        PTB_data_reader.load_data_fast()
    train_reader = PTB_data_reader.DataReader(word_tensors['train'], char_tensors['train'],
                              FLAGS.batch_size, FLAGS.num_unroll_steps) 
    
    #load sentiment analysis data
    # sentiReader = dataloader.SentiDataLoader(
    #                                         dirpath = '/home/hadoop/trainingandtestdata',
    #                                         trainfile = 'training.1600000.processed.noemoticon.csv', 
    #                                         testfile = 'testdata.manual.2009.06.14.csv', 
    #                                         charVocab = char_vocab
    #                         )
    # sentiReader.load_data()
    with open('data/sentiReader.txt', 'rb') as handle:
        sentiReader = pickle.load(handle)

    # load twitter data
    # load_data(FLAGS.data_file_path)
    load_data_fast()

    # (self, input_dim, hidden_dim, max_seq_len, max_word_num, class_num, action_num):
    print(  FLAGS.embedding_dim, FLAGS.hidden_dim, 
                FLAGS.max_seq_len, FLAGS.max_sent_len, 
                    FLAGS.class_num, FLAGS.action_num   )
    logger.info(    (FLAGS.embedding_dim, FLAGS.hidden_dim, 
                        FLAGS.max_seq_len, FLAGS.max_sent_len, 
                            FLAGS.class_num, FLAGS.action_num)  )

    print(get_curtime() + " Data loaded.")
    logger.info(get_curtime() + " Data loaded.")

    # char-level language model
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
                num_unroll_steps = FLAGS.num_unroll_steps, 
                rnn_size = FLAGS.rnn_size, 
                num_rnn_layers = FLAGS.rnn_layers, 
                word_vocab_size = word_vocab.size
            )

    char_train_graph = lstm_char_cnn.infer_train_model(
                        w2v, lstm_lm, 
                        batch_size = FLAGS.batch_size, 
                        num_unroll_steps = FLAGS.num_unroll_steps, 
                        max_word_length = FLAGS.max_char_num, 
                        learning_rate = FLAGS.learning_rate,
                        max_grad_norm = FLAGS.max_grad_norm
                     )
    #sentiment analysis model
    s_model = model.SentiModel(FLAGS.hidden_dim, 5)
    senti_train_graph = model.InferSentiTrainGraph(
                            w2v, 
                            s_model, 
                            max_word_num = FLAGS.max_sent_len, 
                            max_char_num = FLAGS.max_char_num, 
                            hidden_dim = FLAGS.hidden_dim, 
                            sent_num = FLAGS.sent_num,
                            embedding_dim = FLAGS.embedding_dim
                        )
    # df model
    rdm_model = model.RDM_Model(
                    max_seq_len = FLAGS.max_seq_len, 
                    max_word_num = FLAGS.max_sent_len, 
                    embedding_dim = FLAGS.embedding_dim, 
                    hidden_dim = FLAGS.hidden_dim
                )
    rdm_train_graph = model.InferRDMTrainGraph(
                            w2v, s_model, rdm_model, 
                            max_seq_len = FLAGS.max_seq_len, 
                            max_word_num = FLAGS.max_sent_len, 
                            max_char_num = FLAGS.max_char_num, 
                            hidden_dim = FLAGS.hidden_dim, 
                            embedding_dim = FLAGS.embedding_dim,
                            class_num = FLAGS.class_num
                    )

    # rl model
    cm_model = model.CM_Model(
                        max_word_num = FLAGS.max_sent_len, 
                        embedding_dim = FLAGS.embedding_dim, 
                        hidden_dim = FLAGS.hidden_dim, 
                        action_num = FLAGS.action_num
                )
    cm_train_graph = model.InferCMTrainGraph(
                            w2v, s_model, rdm_model, cm_model, 
                            max_word_num = FLAGS.max_sent_len, 
                            embedding_dim = FLAGS.embedding_dim, 
                            hidden_dim = FLAGS.hidden_dim, 
                            action_num = FLAGS.action_num
                        )

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

    ckpt_dir = FLAGS.train_dir
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print(checkpoint.model_checkpoint_path+" is restored.")
        logger.info(checkpoint.model_checkpoint_path+" is restored.")
    else:
        lstm_char_cnn.Train_Char_Model(sess, char_train_graph, train_reader, saver, summary_writer)
        print("df_model "+" saved")
        logger.info("df_model "+" saved")

    ckpt_dir = "senti_saved"
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print(checkpoint.model_checkpoint_path+" is restored.")
        logger.info(checkpoint.model_checkpoint_path+" is restored.")
    else:
        model.TrainSentiModel(sess, saver, logger, senti_train_graph, FLAGS.batch_size, FLAGS.batch_size)
        saver.save(sess, "df_saved/model")
        print("df_model "+" saved")
        logger.info("df_model "+" saved")

    ckpt_dir = "df_saved"
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print(checkpoint.model_checkpoint_path+" is restored.")
        logger.info(checkpoint.model_checkpoint_path+" is restored.")
    else:
        model.TrainRDMModel(sess, logger, rdm_train_graph, t_acc, t_steps, new_data_len=[])
        saver.save(sess, "df_saved/model")
        print("df_model "+" saved")
        logger.info("df_model "+" saved")

    for i in range(20):
        model.TrainCMModel(sess, logger, rdm_train_graph, cm_train_graph, t_rw, t_steps)
        saver.save(sess, "rl_saved/model"+str(i))
        print("rl_model "+str(i)+" saved")
        logger.info("rl_model "+str(i)+" saved")
        new_len = get_new_len(sess, rl_model)
        model.TrainRDMModel(sess, logger, rdm_train_graph, t_acc, t_steps, new_data_len=[])
        saver.save(sess, "df_saved/model"+str(i))
        print("df_model "+str(i)+" saved")
        logger.info("df_model "+str(i)+" saved")
        if acc > 0.9:
            break

    print("The End of My Program")
    logger.info("The End of My Program")
