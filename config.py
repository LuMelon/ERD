import tensorflow as tf


tf.flags.DEFINE_string("w2v_file_path", "/home/hadoop/glove.840B.300d.txt", "w2v file")
tf.flags.DEFINE_string("data_file_path", "/home/hadoop/pheme-rnr-dataset", "data_file")

tf.flags.DEFINE_integer("post_fn", 1, "Fixed Number of Posts")
tf.flags.DEFINE_integer("time_limit", 48, "Posts Time Limitation (The Posts in 48 Hours)")

tf.flags.DEFINE_integer("batch_size", 50, "Batch size (default: 50)")
tf.flags.DEFINE_integer("hidden_dim", 200, "Dimensionality of hidden states (default: 100)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding (default: 300)")
# tf.flags.DEFINE_integer("max_seq_len", 300, "Max length of sequence (default: 300)")
# tf.flags.DEFINE_integer("max_sent_len", 500, "Max length of sentence (default: 300)")
tf.flags.DEFINE_integer("max_char_num", 50, "Max length of sentence (default: 300)")

tf.flags.DEFINE_integer("class_num", 2, "#Class (Non-Rumor, Rumor)")
tf.flags.DEFINE_integer("action_num", 2, "#Action (Continue, Stop)")
tf.flags.DEFINE_integer("sent_num", 3, "#Sentiment Degree (negative, neural, positive)")

# RL parameters;
tf.flags.DEFINE_float("random_rate", 0.01, "RL Random Action Rate")
tf.flags.DEFINE_integer("OBSERVE", 1000, "OBSERVE BEFORE TRAIN")
tf.flags.DEFINE_integer("max_memory", 80000, "Max memory size")
tf.flags.DEFINE_float("reward_rate", 0.2, "reward rate")

#char-embedding parameters
tf.flags.DEFINE_string('data_dir',    'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
tf.flags.DEFINE_string('train_dir',   'cv',     'training directory (models and summaries are saved there periodically)')
tf.flags.DEFINE_string('load_model',   None,    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
tf.flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
tf.flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
tf.flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
tf.flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
tf.flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
tf.flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
tf.flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')

# optimization
tf.flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
tf.flags.DEFINE_float  ('learning_rate',       1.0,  'starting learning rate')
tf.flags.DEFINE_float  ('decay_when',          1.0,  'decay if validation perplexity does not improve by more than this much')
tf.flags.DEFINE_float  ('param_init',          0.05, 'initialize parameters at')
tf.flags.DEFINE_integer('num_unroll_steps',    35,   'number of timesteps to unroll for')
# flags.DEFINE_integer('batch_size',          20,   'number of sequences to train on in parallel')
tf.flags.DEFINE_integer('max_epochs',          25,   'number of full passes through the training data')
tf.flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
tf.flags.DEFINE_integer('max_word_length',     65,   'maximum word length')

# bookkeeping
tf.flags.DEFINE_integer('seed',           3435, 'random number generator seed')
tf.flags.DEFINE_integer('print_every',    5,    'how often to print current loss')
tf.flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = tf.flags.FLAGS
