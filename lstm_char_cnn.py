from __future__ import print_function
from __future__ import division


import tensorflow as tf


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




class WordEmbedding:
    def __init__(self, max_word_length, char_vocab_size, char_embed_size, kernels, kernel_features, num_highway_layers):
#             self.input_ = tf.placeholder(tf.int32, [None, max_word_length, char_vocab_size],name="W2V_input")
        self.max_word_length = max_word_length
        self.char_vocab_size = char_vocab_size
        self.char_embed_size = char_embed_size
        self.kernels = kernels
        self.kernel_features = kernel_features
        self.num_highway_layers = num_highway_layers
        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            self.char_embedding = tf.get_variable('char_embedding', [self.char_vocab_size, self.char_embed_size])
            ''' this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
            of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
            zero embedding vector and ignores gradient updates. For that do the following in TF:
            1. after parameter initialization, apply this op to zero out padding embedding vector
            2. after each gradient update, apply this op to keep padding at zero'''
            self.clear_char_embedding_padding = tf.scatter_update(self.char_embedding, [0], tf.constant(0.0, shape=[1, self.char_embed_size]))
            
    def __call__(self, input_words):
        input_ = input_words
        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            # [batch_size x max_word_length, num_unroll_steps, char_embed_size]
            input_embedded = tf.nn.embedding_lookup(self.char_embedding, input_)
            input_embedded = tf.reshape(input_embedded, [-1, self.max_word_length, self.char_embed_size])
        input_cnn = self.tdnn(input_embedded, self.kernels, self.kernel_features)
        ''' Maybe apply Highway '''
#             if num_highway_layers > 0:
        assert self.num_highway_layers > 0
        input_cnn = self.highway(input_cnn, input_cnn.get_shape()[-1], num_layers=self.num_highway_layers, scope="CNN_OUT")
        return input_cnn
    
    def conv2d(self, input_, output_dim, k_h, k_w, name="conv2d"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
            b = tf.get_variable('b', [output_dim])
        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

    def highway(self, input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for idx in range(num_layers):
                g = f(linear(input_, size, scope='highway_lin_%d' % idx))

                t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

                output = t * g + (1. - t) * input_
                input_ = output
        print(output)
        return output

    def tdnn(self, input_, kernels, kernel_features, scope='TDNN'):
        '''
        :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
        :kernels:         array of kernel sizes
        :kernel_features: array of kernel feature sizes (parallel to kernels)
        '''
        assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'
        max_word_length = input_.get_shape()[1]
        embed_size = input_.get_shape()[-1]
        # input_: [batch_size*num_unroll_steps, 1, max_word_length, embed_size]
        input_ = tf.expand_dims(input_, 1)
        layers = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
                reduced_length = max_word_length - kernel_size + 1
                # [batch_size*num_unroll_steps, 1, reduced_length, kernel_feature_size]
                conv = self.conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
                # [batch_size*num_unroll_steps, 1, 1, kernel_feature_size]
                pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
                layers.append(tf.squeeze(pool, [1, 2]))
            if len(kernels) > 1:
                output = tf.concat(layers, 1)
            else:
                output = layers[0]
        return output


class LSTM_LM:
    def __init__(self, batch_size, num_unroll_steps, input_dim, rnn_size, num_rnn_layers, word_vocab_size, dropout):
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.input_dim = input_dim
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.word_vocab_size = word_vocab_size
        with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
            def create_rnn_cell():
                cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-dropout)
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
            # linear projection onto output (word) vocab
            logits = []
            with tf.variable_scope('WordEmbedding') as scope:
                for idx, output in enumerate(outputs):
                    if idx > 0:
                        scope.reuse_variables()
                    logits.append(linear(output, self.word_vocab_size))
            return logits, outputs, final_rnn_state
        

def infer_train_model(word2vec, LM, 
                      batch_size, 
                      num_unroll_steps, 
                      max_word_length, 
                      learning_rate,
                      max_grad_norm
                     ):
    drop_out = tf.placeholder(tf.float32)
    input_ = tf.placeholder(tf.int32, shape=[batch_size, num_unroll_steps, max_word_length], name="input")
    targets = tf.placeholder(tf.int64, [batch_size, num_unroll_steps], name='targets')
    
    input_cnn = word2vec(input_) #[batch_size*num_unroll_steps, k_features]
    input_cnn = tf.reshape(input_cnn, [batch_size, num_unroll_steps, -1])
    logits, outputs, final_rnn_state = LM(input_cnn)
    
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
        clear_char_embedding_padding=word2vec.clear_char_embedding_padding,
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


def Train_Char_Model(session, train_model, train_reader, saver, summary_writer):
    best_valid_loss = None
    rnn_state = session.run(train_model.initial_rnn_state)
    for epoch in range(FLAGS.max_epochs):
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
        save_as = '%s/epoch%03d_%.4f.model' % (FLAGS.train_dir, epoch, avg_train_loss)
        saver.save(session, save_as)
        print('Saved char model', save_as)
