# coding: utf-8
from collections import deque
from ERDModel import RL_GRU2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from dataUtils import *
from logger import MyLogger
os.chdir("/home/hadoop/ERD")


tf.logging.set_verbosity(tf.logging.ERROR)

logger = MyLogger("ERDMain")

def df_train(sess, summary_writer, mm, t_acc, t_steps, new_data_len=[]):
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
        summary = tf.Summary(value=[
                    tf.Summary.Value(tag="df_train_loss", simple_value=loss),
                    tf.Summary.Value(tag="df_train_accuracy", simple_value=acc),
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
            sum_acc = 0.0
            sum_loss = 0.0

    print(get_curtime() + " Train df Model End.")
    logger.info(get_curtime() + " Train df Model End.")
    return ret_acc


def rl_train(sess, mm, t_rw, t_steps):
    ids = np.array(range(FLAGS.batch_size), dtype=np.int32)
    seq_states = np.zeros([FLAGS.batch_size], dtype=np.int32)
    isStop = np.zeros([FLAGS.batch_size], dtype=np.int32)
    max_id = FLAGS.batch_size
    init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)
    feed_dic = {mm.init_states: init_states}
    state = sess.run(mm.df_state, feed_dic)
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
        feed_dic = {mm.input_x: x, mm.x_len: x_len, mm.input_y: y, mm.init_states:init_states, mm.dropout_keep_prob: 1.0}
        t_ssq = sess.run(mm.out_seq, feed_dic)# t_ssq = [batchsize, max_seq, scores]
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
            feed_dic = {mm.rl_state: s_state, mm.rl_input: s_x, mm.action: s_isStop, mm.reward:s_rw, mm.dropout_keep_prob: 0.5}
            _, step = sess.run([rl_train_op, rl_global_step], feed_dic)

        x, y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, 3150)
        batch_dic = {mm.rl_state: state, mm.rl_input: x, mm.dropout_keep_prob: 1.0}
        isStop, mss, mNewState = sess.run([mm.isStop, mm.stopScore, mm.rl_new_state], batch_dic)

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
                # feed_dic = {mm.init_states: init_states}
                # state[j] = sess.run(mm.df_state, feed_dic)
                state[j] = np.zeros([FLAGS.hidden_dim], dtype=np.float32)
        counter += 1


def eval(sess, mm):
    start_ef = int(eval_flag / FLAGS.batch_size)
    end_ef = int(len(data_ID) / FLAGS.batch_size) + 1
    init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)

    counter = 0
    sum_acc = 0.0

    for i in range(start_ef, end_ef):
        x, x_len, y = get_df_batch(i)
        feed_dic = {mm.input_x: x, mm.x_len: x_len, mm.input_y: y, mm.init_states: init_states, mm.dropout_keep_prob: 1.0}
        _, step, loss, acc = sess.run([df_train_op, df_global_step, mm.loss, mm.accuracy], feed_dic)
        counter += 1
        sum_acc += acc

    print(sum_acc / counter)

if __name__ == "__main__":
    print(get_curtime() + " Loading data ...")
    logger.info(get_curtime() + " Loading data ...")
    # load_data(FLAGS.data_file_path)
    load_data_fast()
    print(get_curtime() + " Data loaded.")
    logger.info(get_curtime() + " Data loaded.")
    gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # (self, input_dim, hidden_dim, max_seq_len, max_word_len, class_num, action_num):
    print(FLAGS.embedding_dim, FLAGS.hidden_dim, FLAGS.max_seq_len, FLAGS.max_sent_len, FLAGS.class_num, FLAGS.action_num)
    logger.info((FLAGS.embedding_dim, FLAGS.hidden_dim, FLAGS.max_seq_len, FLAGS.max_sent_len, FLAGS.class_num, FLAGS.action_num))
    
    sess = tf.Session(config=gpu_config)
    with  sess.as_default():
        with tf.device('/GPU:0'):
            mm = RL_GRU2(FLAGS.embedding_dim, FLAGS.hidden_dim, FLAGS.max_seq_len,
                         FLAGS.max_sent_len, FLAGS.class_num, FLAGS.action_num, FLAGS.sent_num)
            
            # df model
            df_global_step = tf.Variable(0, name="global_step", trainable=False)
            df_train_op = tf.train.AdagradOptimizer(0.05).minimize(mm.loss, df_global_step)

            # rl model
            rl_global_step = tf.Variable(0, name="global_step", trainable=False)
            rl_train_op = tf.train.AdamOptimizer(0.001).minimize(mm.rl_cost, rl_global_step)
            
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)
            sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter("./reports/", graph=sess.graph)
        # ckpt_dir = "df_saved_erd"
        # checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     print("--------------Debug1------------------")
        #     saver.restore(sess, checkpoint.model_checkpoint_path)
        #     print(checkpoint.model_checkpoint_path+" is restored.")
        #     logger.info(checkpoint.model_checkpoint_path+" is restored.")
        # else:
        #     df_train(sess, summary_writer, mm, 0.80, 20000)
        #     saver.save(sess, "df_saved_erd/model")
        #     print("df_model "+" saved")
        #     logger.info("df_model "+" saved")
    df_train(sess, summary_writer, mm, 0.90, 20000)

    for i in range(20):
        rl_train(sess, mm, 0.5, 50000)
        saver.save(sess, "rl_saved_erd/model"+str(i))
        print("rl_model "+str(i)+" saved")
        logger.info("rl_model "+str(i)+" saved")
        new_len = get_new_len(sess, mm)
        acc = df_train(sess, summary_writer, mm, 0.9, 1000, new_len)
        saver.save(sess, "df_saved_erd/model"+str(i))
        print("df_model "+str(i)+" saved")
        logger.info("df_model "+str(i)+" saved")
        if acc > 0.9:
            break

    print("The End of My Program")
    logger.info("The End of My Program")
