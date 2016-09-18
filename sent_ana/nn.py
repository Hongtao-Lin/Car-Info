#!/usr/bin/python
# -*- coding:utf8 -*-

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from theano import tensor as T
from theano import function
import numpy as np
from utils import GetWordVector, cosine_distance
import random




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
# 
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

def MCE_s(y_true, y_pred):
    """A custom loss function"""
    return cosine(y_true, y_pred)

def MCE_a(y_true, y_pred):
    """A custom loss function"""
    return -cosine(y_true, y_pred)    

sgd_faster = SGD(lr=0.4, decay=1e-6, momentum=0.9, nesterov=True)

class DNN:
    def __init__(self):
        share_emb_lay = Embedding(115227, 50, input_length=200, weights=pre_weights)
        model1 = Sequential()
        model1.add(share_emb_lay)
        model1.compile(loss=MCE_s, optimizer=sgd_faster)
        model2 = Sequential()
        model2.add(share_emb_lay)
        model2.compile(loss=MCE_a, optimizer="adadelta")
        print ("Build Network Completed...")
        self.model_s = model1
        self.model_a = model2
        self.emb_lay = share_emb_lay
        self.vocab = {"get_index":{}, "get_word":[]}

    def build_vocab(self, input_file):
        infile = open(input_file, 'r')
        save_vocab = open("vocab.dic", "w")
        lines = infile.readlines()
        idx = 0
        for line in lines:
            word = line.strip()
            self.vocab["get_word"].append(word)
            self.vocab["get_index"][word] = idx
            idx += 1
        save_vocab.write(str(self.vocab))
        save_vocab.close()

    def train(self, X, Y):
        pass

    def test(self):
        pass
    
    def draw(self, save_pic):
        plot(self.model_s, to_file=save_pic+'s.png')
        plot(self.model_a, to_file=save_pic+'a.png')

def pair_distance(x, y, rng, flag = True, after_weights=None):
    total = 0
    for i in xrange(0, rng, 2):
        if flag:
            total += cosine_distance(pre_weights[0][x[i]], pre_weights[0][y[i]])
        else:
            total += cosine_distance(after_weights[x[i]], after_weights[y[i]])
    return total * 2 / rng 

if __name__ == '__main__':
    log_file = open("log.txt", "w")
    a = DNN()
    #a.draw("model")
    #exit()
    a.build_vocab("vocab")
    exit()
    s_input = []
    s_input_v = []
    s_output = []
    s_output_v = []

    a_input = []
    a_input_v = []
    a_output = []
    a_output_v = []

    g = GetWordVector()
    g.load_all()
    print (len(g.synpair))
    print (len(g.antpair))
    for i in xrange(100):
        sample_syn = random.randint(0, 7522-1)
        sample_ant = random.randint(0, 1397-1)
        s_input.append(a.vocab["get_index"][g.synpair[sample_syn][0]])
        s_input_v.append(pre_weights[0][s_input[i]])
        s_output.append(a.vocab["get_index"][g.synpair[sample_syn][1]])
        s_output_v.append(pre_weights[0][s_output[i]])
        a_input.append(a.vocab["get_index"][g.antpair[sample_ant][0]])
        a_input_v.append(pre_weights[0][a_input[i]])
        a_output.append(a.vocab["get_index"][g.antpair[sample_ant][1]])
        a_output_v.append(pre_weights[0][a_output[i]])
    for i in xrange(100, 200):
        sample_idx1 = random.randint(0, 115227-1)
        sample_idx2 = random.randint(0, 115227-1)
        sample_ant = random.randint(0, 1397-1)
        s_input.append(sample_idx1)
        s_output.append(sample_idx2)
        s_input_v.append(pre_weights[0][sample_idx1])
        s_output_v.append(pre_weights[0][sample_idx2])
        a_input.append(sample_ant)
        a_output.append(sample_ant)
        a_input_v.append(pre_weights[0][sample_ant])
        a_output_v.append(pre_weights[0][sample_ant])

    # FINAL TEST
    s_test_input = []
    s_test_output = []
    a_test_input = []
    a_test_output = []
    for i in xrange(7522):
        s_test_input.append(a.vocab["get_index"][g.synpair[i][0]])
        s_test_output.append(a.vocab["get_index"][g.synpair[i][1]])
    for i in xrange(1397):
        a_test_input.append(a.vocab["get_index"][g.antpair[i][0]])
        a_test_output.append(a.vocab["get_index"][g.antpair[i][1]])

    print (pair_distance(s_test_input, s_test_output, 7522),\
          g.average_rand_samples_distance(), \
          pair_distance(a_test_input, a_test_output, 1397))
    log_file.write("{0} {1}\n".format(pair_distance(s_test_input, s_test_output, 7522) ,pair_distance(a_test_input, a_test_output, 1397)))
    
    for cnt in xrange(100):
        print "Epoch {0} ... ".format(cnt)
        for batch in xrange(40):
            a.model_s.train_on_batch(np.array([s_input]), np.array([s_output_v]))
            a.model_s.train_on_batch(np.array([s_output]), np.array([s_input_v]))
            a.model_a.train_on_batch(np.array([a_input]), np.array([a_output_v]))
            a.model_a.train_on_batch(np.array([a_output]), np.array([a_input_v]))
            for i in xrange(100,200):
                sample_idx1 = random.randint(0, 115227-1)
                sample_idx2 = random.randint(0, 115227-1)
                sample_ant = random.randint(0, 1397-1)
                s_input[i] = sample_idx1
                s_output[i] = sample_idx2
                s_input_v[i] = pre_weights[0][sample_idx1]
                s_output_v[i] = pre_weights[0][sample_idx2]
                a_input[i] = sample_ant
                a_output[i] = sample_ant
                a_input_v[i] = pre_weights[0][sample_ant]
                a_output_v[i] = pre_weights[0][sample_ant]
            for i in xrange(100):
                sample_syn = random.randint(0, 7522-1)
                sample_ant = random.randint(0, 1397-1)
                s_input[i] = a.vocab["get_index"][g.synpair[sample_syn][0]]
                s_input_v[i] = pre_weights[0][s_input[i]]
                s_output[i] = a.vocab["get_index"][g.synpair[sample_syn][1]]
                s_output_v[i] = pre_weights[0][s_output[i]]
                a_input[i] = a.vocab["get_index"][g.antpair[sample_ant][0]]
                a_input_v[i] = pre_weights[0][a_input[i]]
                a_output[i] = a.vocab["get_index"][g.antpair[sample_ant][1]]
                a_output_v[i] = pre_weights[0][a_output[i]]
        after_weights = a.emb_lay.get_weights()[0]
        # print after_weights
        # print pair_distance(s_input[0:30], s_output[0:30], 30, False, after_weights),\
        # pair_distance(s_input[30:60], a_output[30:60], 30, False, after_weights),\
        # pair_distance(a_input[0:30], a_output[0:30], 30, False, after_weights)
        print "On Test",\
          pair_distance(s_test_input, s_test_output, 7522, False, after_weights),\
          g.average_rand_samples_distance(after_weights), \
          pair_distance(a_test_input, a_test_output, 1397, False, after_weights)
        print ("")
        log_file.write("EPOCH {0} {1} {2} {3}\n\n".format(cnt,
            pair_distance(s_test_input, s_test_output, 7522, False, after_weights),\
          g.average_rand_samples_distance(after_weights), \
          pair_distance(a_test_input, a_test_output, 1397, False, after_weights)))


    # FINAL TEST
    np.save("new_word_vec_l0.4", after_weights)
    log_file.close()
