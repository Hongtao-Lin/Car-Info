#!/usr/bin/python
# -*- coding:utf8 -*-

from keras.models import Sequential
from keras.layers import *
from theano import tensor as T
from theano import function
import numpy as np
from utils import GetWordVector, cosine_distance
import random
import sys
sys.setrecursionlimit(300)

ZERO = T.zeros(1)[0]
ONE = T.ones(1)[0]

# According to the paper
ALPHA = ONE * 0.4
BETA = ONE * 0.4

# lOAD PRE-TRAINED WORD VECTORS

# def load_word_vectors():
#     iwvf = open("word_vec_total.txt", "r")
#     wordvecs = []
#     line = iwvf.readline()
#     while line != "":
#         line = line.split()
#         if len(line) == 51:
#             line = np.array([float(x) for x in line[1:]])
#             wordvecs.append(line)
#         line = iwvf.readline()
#     iwvf.close()
#     np.save("pre_trained_word_vector.npy", np.array([wordvecs]))
# load_word_vectors()
# exit()
pre_weights = np.load("pre_trained_word_vector.npy")



def _squared_magnitude(x):
    return T.sqr(x).sum(axis=-1)

def _magnitude(x):
    return T.sqrt(T.maximum(_squared_magnitude(x), np.finfo(x.dtype).tiny))

def cosine(x, y):
    return T.clip((1 - (x * y).sum(axis=-1) / \
        (_magnitude(x) * _magnitude(y))) / 2, 0, 1)

def _obj_mce_s(syns, unre, rng):
    sum = ZERO
    for i in xrange(0, rng, 2):
        for j in xrange(0, rng/2):
            d_ii = cosine(syns[i], syns[i + 1])

            sum = sum + T.maximum(ZERO, ALPHA - cosine(syns[i], unre[j])\
                                 + d_ii)
            sum = sum + T.maximum(ZERO, ALPHA - cosine(syns[i + 1], unre[j])\
                                 + d_ii)
    return sum

def _obj_mce_a(ants, unre, rng):
    sum = ZERO
    for i in xrange(0, rng, 2):
        for j in xrange(0, rng/2):
            d_ii = cosine(ants[i], ants[i + 1])

            sum = sum + T.maximum(ZERO, BETA + cosine(ants[i], unre[j])\
                                 - d_ii)
            sum = sum + T.maximum(ZERO, BETA + cosine(ants[i + 1], unre[j])\
                                 - d_ii)
    return sum

def MCE(y_true, y_pred):
    """A custom loss function"""
    return cosine(y_true, y_pred)
    
class DNN:
    def __init__(self):
        model = Sequential()
        model.add(Embedding(115227, 50, input_length=75, weights=pre_weights))
        model.compile(loss=MCE, optimizer="adadelta")
        print "Build Network Completed..."
        self.model = model
        self.vocab = {"get_index":{}, "get_word":[]}

    def build_vocab(self, input_file):
        infile = open(input_file, 'r')
        lines = infile.readlines()
        idx = 0
        for line in lines:
            word = line.strip()
            self.vocab["get_word"].append(word)
            self.vocab["get_index"][word] = idx
            idx += 1

    def train(self, X, Y):
        pass

    def test(self):
        pass

def pair_distance(x, y, rng, flag = True, after_weights=None):
    total = 0
    for i in xrange(0, rng, 2):
        if flag:
            total += cosine_distance(pre_weights[0][x[i]], pre_weights[0][y[i]])
        else:
            total += cosine_distance(after_weights[x[i]], after_weights[y[i]])
    return total * 2 / rng 

if __name__ == '__main__':

    a = DNN()
    a.build_vocab("vocab")
    _input = []
    _input_v = []
    _output = []
    _output_v = []
    g = GetWordVector()
    g.load_all()

    for i in xrange(30):
        _input.append(a.vocab["get_index"][g.synpair[i][0]])
        _input_v.append(pre_weights[0][_input[i]])
        _output.append(a.vocab["get_index"][g.synpair[i][1]])
        _output_v.append(pre_weights[0][_output[i]])
    for i in xrange(30,45):
        sample_idx1 = random.randint(0, 115227-1)
        _input.append(sample_idx1)
        _output.append(sample_idx1)
        _input_v.append(pre_weights[0][sample_idx1])
        _output_v.append(pre_weights[0][sample_idx1])
    for i in xrange(45,75):
        _input.append(a.vocab["get_index"][g.antpair[i][0]])
        _input_v.append(pre_weights[0][_input[i]])
        _output.append(a.vocab["get_index"][g.antpair[i][1]])
        _output_v.append(pre_weights[0][_output[i]])

    print pair_distance(_input[0:30], _output[0:30], 30)
    print pair_distance(_input[30:45], _output[30:45], 15)
    print pair_distance(_input[45:75], _output[45:75], 30)
    for i in xrange(100):
        print "Epoch {0} ... ".format(i)
        a.model.train_on_batch(np.array([_input]), np.array([_output_v]))
        a.model.train_on_batch(np.array([_output]), np.array([_input_v]))
        after_weights = a.model.get_layer('embedding_1').get_weights()[0]
        # print after_weights
        print pair_distance(_input[0:30], _output[0:30], 30, False, after_weights),\
        pair_distance(_input[30:45], _output[30:45], 15, False, after_weights), \
        pair_distance(_input[45:75], _output[45:75], 30, False, after_weights)

