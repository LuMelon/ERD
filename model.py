import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.losses import Reduction

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

class RL_GRU2:
    def __init__(self, max_char_num, max_word_num, max_seq_len, embedding_dim, hidden_dim):
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.max_char_num = max_char_num
        self.max_word_num = max_word_num
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # df model
        self.df_cell = rnn.GRUCell(self.hidden_dim)
        self.df_cell = rnn.DropoutWrapper(self.df_cell, output_keep_prob=self.dropout_keep_prob)

    def RDM_model(self, input_x, class_num):
        # input_x = tf.placeholder(tf.float32, [None, self.max_seq_len, self.max_word_num, self.embedding_dim], name="input_x")
        input_y = tf.placeholder(tf.float32, [None, class_num], name="input_y")
        x_len = tf.placeholder(tf.int32, [None], name="x_len")
        init_states = tf.placeholder(tf.float32, [None, self.hidden_dim], name="topics")
        #[batchsize, max_seq_len, max_word_len, input_dim] --> [batchsize, max_seq_len, output_dim]
        pooled_input_x = self.shared_pooling_layer(input_x, self.embedding_dim, self.max_seq_len, self.max_word_num, self.hidden_dim) # replace the shared_pooling_layer with a sentiment analysis model
        # dropout layer
        pooled_input_x_dp = tf.nn.dropout(pooled_input_x, self.dropout_keep_prob)
        df_outputs, df_last_state = tf.nn.dynamic_rnn(self.df_cell, pooled_input_x_dp, x_len, initial_state=init_states, dtype=tf.float32)
        l2_loss = tf.constant(0.0)
        w_ps = tf.Variable(tf.truncated_normal([self.hidden_dim, class_num], stddev=0.1)) #
        b_ps = tf.Variable(tf.constant(0.01, shape=[class_num])) #
        l2_loss += tf.nn.l2_loss(w_ps) 
        l2_loss += tf.nn.l2_loss(b_ps) 

        pre_scores = tf.nn.xw_plus_b(df_last_state, w_ps, b_ps, name="p_scores")
        predictions = tf.argmax(pre_scores, 1, name="predictions")

        r_outputs = tf.reshape(df_outputs, [-1, self.hidden_dim]) #[batchsize*max_seq_len, output_dim]
        scores_seq = tf.nn.softmax(tf.nn.xw_plus_b(r_outputs, w_ps, b_ps)) # [batchsize * max_seq_len, class_num] 
        out_seq = tf.reshape(scores_seq, [-1, self.max_seq_len, class_num], name="out_seq") #[batchsize, max_seq_len, class_num]

        df_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pre_scores, labels=input_y)
        loss = tf.reduce_mean(df_losses) + 0.1 * l2_loss

        correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return adict(
                dropout_keep_prob = self.dropout_keep_prob,
                # input_x = input_x,
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


    def CM_model(self, action_num):
        rl_state = tf.placeholder(tf.float32, [None, self.hidden_dim], name="rl_states")
        rl_input = tf.placeholder(tf.float32, [None, self.max_word_num, self.embedding_dim], name="rl_input")
        action = tf.placeholder(tf.float32, [None, action_num], name="action")
        reward = tf.placeholder(tf.float32, [None], name="reward")

        pooled_rl_input = self.shared_pooling_layer(rl_input, self.embedding_dim, 1, self.max_word_num, self.hidden_dim)
        pooled_rl_input = tf.reshape(pooled_rl_input, [-1, self.hidden_dim])

        rl_output, rl_new_state = self.df_cell(pooled_rl_input, rl_state)
        
        w_ss1 = tf.Variable(tf.truncated_normal([self.hidden_dim, 64], stddev=0.01))
        b_ss1 = tf.Variable(tf.constant(0.01, shape=[64]))
        rl_h1 = tf.nn.relu(tf.nn.xw_plus_b(rl_state, w_ss1, b_ss1))  # replace the process here

        w_ss2 = tf.Variable(tf.truncated_normal([64, action_num], stddev=0.01))
        b_ss2 = tf.Variable(tf.constant(0.01, shape=[action_num]))

        stopScore = tf.nn.xw_plus_b(rl_h1, w_ss2, b_ss2, name="stopScore")

        isStop = tf.argmax(stopScore, 1, name="isStop")

        out_action = tf.reduce_sum(tf.multiply(stopScore, action), reduction_indices=1)
        rl_cost = tf.reduce_mean(tf.square(reward - out_action), name="rl_cost")
        return adict(
            dropout_keep_prob = self.dropout_keep_prob,
            rl_state = rl_state, 
            rl_input = rl_input,
            action = action,
            reward = reward,  
            rl_new_state = rl_new_state,
            stopScore = stopScore,
            isStop = isStop,
            rl_cost = rl_cost
            )

    def Sentiment(self, sent_num):
        # sent_x : [batch_size, max_word_num, embedding_dim]
        # sent_y : [batch_size, sent_num]
        # Sentiment Analysis Task
        sent_x = tf.placeholder(tf.float32, shape = [None, self.max_word_num, self.embedding_dim])
        sent_y = tf.placeholder(tf.float32, shape = [None, sent_num])
        pooled_feat = self.SentCNN(sent_x)
        classifier = tf.layers.Dense(sent_num, activation= tf.nn.relu, trainable=True)
        sent_scores = tf.nn.softmax(classifier(pooled_feat), axis=1)
        sent_pred = tf.argmax(sent_scores, 1, name="predictions")
        sent_loss = tf.losses.softmax_cross_entropy(
                        sent_y,
                        sent_scores,
                        weights=1.0,
                        label_smoothing=0,
                        scope=None,
                        loss_collection=tf.GraphKeys.LOSSES,
                        reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
                    )
        sent_correct_predictions = tf.equal(sent_pred, tf.argmax(sent_y, 1))
        sent_acc = tf.reduce_mean(tf.cast(sent_correct_predictions, "float"), name="accuracy")
        return adict(
                dropout_keep_prob = self.dropout_keep_prob,
                sent_x = sent_x,
                sent_y = sent_y,
                feature = pooled_feat,
                sent_scores = sent_scores,
                sent_pred = sent_pred,
                sent_loss = sent_loss,
                sent_acc = sent_acc
            )

    def shared_pooling_layer(self, inputs, input_dim, max_seq_len, max_word_len, output_dim):
        w_t = tf.Variable(tf.random_uniform([input_dim, self.hidden_dim], -1.0, 1.0), name="w_t")
        b_t = tf.Variable(tf.constant(0.01, shape=[self.hidden_dim]), name="b_t")
        t_inputs = tf.reshape(inputs, [-1, input_dim])
        t_h = tf.nn.xw_plus_b(t_inputs, w_t, b_t)
        # t_h = tf.matmul(t_inputs, self.w_t)
        t_h = tf.reshape(t_h, [-1, self.max_word_num, self.hidden_dim])
        t_h_expended = tf.expand_dims(t_h, -1)
        pooled = tf.nn.max_pool(
            t_h_expended,
            ksize=[1, self.max_word_num, 1, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="max_pool"
        )
        outs = tf.reshape(pooled, [-1, self.max_seq_len, self.hidden_dim])
        return outs

    def pooling_layer(self, inputs, input_dim, max_seq_len, max_word_len, output_dim):
        t_inputs = tf.reshape(inputs, [-1, input_dim])
        w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        b = tf.Variable(tf.constant(0.01, shape=[output_dim]))

        h = tf.nn.xw_plus_b(t_inputs, w, b)
        hs = tf.reshape(h, [-1, max_word_len, output_dim])

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

    def SentCNN(self, input_x):
        num_filters = 256
        kernel_size = 5
        conv_input = tf.layers.conv1d(input_x, num_filters, kernel_size, strides=1, padding='valid', name='conv2', trainable=True)
        feature_map = tf.nn.relu(conv_input) # [batchsize, conv_feats, filters]
        pooled_feat = tf.reduce_max(feature_map, 1) #[batchsize, 1, filters]
        return pooled_feat 
