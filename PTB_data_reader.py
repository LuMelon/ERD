from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
import pickle
import config

class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)

def load_data_fast():
  
    with open('data/char_tensors.txt', 'rb') as handle:
        char_tensors = pickle.load(handle)

    with open('data/word_tensors.txt', 'rb') as handle:
        word_tensors = pickle.load(handle)

    with open('data/char_vocab.txt', 'rb') as handle:
        char_vocab = pickle.load(handle)

    with open('data/word_vocab.txt', 'rb') as handle:
        word_vocab = pickle.load(handle)

    with open('data/word_len.txt', 'rb') as handle:
        x_len = pickle.load(handle)

    return word_vocab, char_vocab, word_tensors, char_tensors, x_len
    



def load_data(data_dir, max_word_length, char_vocab, idx2char=None, char2idx=None, idx2word=None, word2idx=None, eos='+'):

    # char_vocab = Vocab(char2idx, idx2char)
    # char_vocab.feed(' ')  # blank is at index 0 in char vocab
    # char_vocab.feed('{')  # start is at index 1 in char vocab
    # char_vocab.feed('}')  # end   is at index 2 in char vocab

    word_vocab = Vocab(word2idx, idx2word)
    word_vocab.feed('|')  # <unk> is at index 0 in word vocab

    actual_max_word_length = 0

    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)

    for fname in ('train', 'valid', 'test'):
        print('reading', fname)
        with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.replace('}', '').replace('{', '').replace('|', '')
                line = line.replace('<unk>', ' | ')
                if eos:
                    line = line.replace(eos, '')

                for word in line.split():
                    if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
                        word = word[:max_word_length-2]

                    word_tokens[fname].append(word_vocab.feed(word))

                    char_array = [char_vocab.feed(c) for c in '{' + word + '}']
                    char_tokens[fname].append(char_array)

                    actual_max_word_length = max(actual_max_word_length, len(char_array))

                if eos:
                    word_tokens[fname].append(word_vocab.feed(eos))

                    char_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                    char_tokens[fname].append(char_array)

    assert actual_max_word_length <= max_word_length

    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', word_vocab.size)
    print('size of char vocabulary:', char_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    x_len = {}
    for fname in ('train', 'valid', 'test'):
        assert len(char_tokens[fname]) == len(word_tokens[fname])
        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)
        x_len[fname] = np.zeros([len(char_tokens[fname])], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname] [i,:len(char_array)] = char_array
            x_len[fname][i] = len(char_array)

    return word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length, x_len


class DataReader:

    def __init__(self, word_tensor, char_tensor, x_len, batch_size, num_unroll_steps):

        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length

        max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        word_tensor = word_tensor[:reduced_length]
        char_tensor = char_tensor[:reduced_length, :]
        x_len = x_len[:reduced_length]

        ydata = np.zeros_like(word_tensor)
        ydata[:-1] = word_tensor[1:].copy()
        ydata[-1] = word_tensor[0].copy()

        x_batches = char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = ydata.reshape([batch_size, -1, num_unroll_steps])
        l_batches = x_len.reshape([batch_size, -1, num_unroll_steps]) 

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        l_batches = np.transpose(l_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        self._l_batches = list(l_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):

        for x, y, l in zip(self._x_batches, self._y_batches, self._l_batches):
            yield x, y, l


if __name__ == '__main__':

    _, _, wt, ct, _ = load_data('data', 65)
    print(wt.keys())

    count = 0
    for x, y in DataReader(wt['valid'], ct['valid'], 20, 35).iter():
        count += 1
        print(x, y)
        if count > 0:
            break
