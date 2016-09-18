#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
from utils import cosine_distance
from utils import GetWordVector
 
pre = np.load("pre_trained_word_vector.npy")[0]
after = np.load("new_word_vec_l0.3.npy")
afterr = np.load("after_class.npy")[0]

xiao_p = pre[92]
xiao_a = after[92]
xiao_c = afterr[93]

da_p = pre[23]
da_a = after[23]
da_c = afterr[24]

xi_p = pre[922]
xi_a = after[922]
xi_c = after[923]

print "xi-xiao"
print cosine_distance(xi_p, xiao_p)
print cosine_distance(xi_a, xiao_a)
print cosine_distance(xi_c, xiao_c)

print "xiao-da"
print cosine_distance(xiao_p, da_p)
print cosine_distance(xiao_a, da_a)
print cosine_distance(xiao_c, da_c)

print "xi-da"
print cosine_distance(xi_p, da_p)
print cosine_distance(xi_a, da_a)
print cosine_distance(xi_c, da_c)

g = GetWordVector()
g.load_all()
g.average_antonym_distance()
g.average_synonym_distance()
g.average_rand_samples_distance()

vocab = eval(open("vocab.dic").read())

def pair_distance(pairs):
  total = 0.0
  for pair in pairs:
    a = pair[0]
    b = pair[1]
    a_id = vocab['get_index'][a]
    b_id = vocab['get_index'][b]
    total += cosine_distance(afterr[a_id+1], afterr[b_id+1])
  total /= len(pairs)
  return total

print pair_distance(g.synpair)
print g.average_rand_samples_distance(after)
print pair_distance(g.antpair)
