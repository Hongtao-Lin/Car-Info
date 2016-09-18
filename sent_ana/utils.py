#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import time
import scipy.spatial.distance
import random

word_vec_file = "word_vec_total.txt"
sym_file = "a.txt"
anto_file = "b.txt"
log_file = "get_vector_log.txt"
vocab_file = "vocab"

def cosine_distance(a, b):
    return scipy.spatial.distance.cosine(a, b)

class GetWordVector:
    """docstring for GetVec"""
    def __init__(self):
        self.wvf = word_vec_file
        self.sf = sym_file
        self.af = anto_file
        self.vf = vocab_file

        self.synpair = []
        self.antpair = []

        self.word2vec = {}
        self.log = open(log_file, "a")
        self.log.write("\n===" + time.strftime('%Y-%m-%d %H:%M:%S') + "===\n")

    def load_wordvec(self):
        iwvf = open(self.wvf, "r")
        # ovf = open(self.vf, "w")
        line = iwvf.readline()
        line = iwvf.readline()
        while line != "":
            line = line.split()
            if len(line) != 51:
                self.log.write("Dimenssion ERROR "+str(line)+'\n')
            else:
                word = line[0]
                # ovf.write(word+'\n')
                line = [float(x) for x in line[1:]]
                self.word2vec[word] = line

            line = iwvf.readline()
        iwvf.close()
        # ovf.close()
        self.log.write("Load Wrod Vectors Completed...\n")
        print ("Load Wrod Vectors Completed...\n")

    def load_synomym(self):
        isf = open(self.sf, "r")
        line = isf.readline()
        while line != "":
            line = line.split()
            if len(line) != 2:
                self.log.write("SYN Load ERROR " + ' '.join(line)+'\n')
            else:
                if [] not in self.get_vec(line):
                    self.synpair.append(tuple(line))
            line = isf.readline()
        isf.close()
        print ("Load SYN Completed...\n")
            
    def load_antomym(self):
        iaf = open(self.af, "r")
        line = iaf.readline()
        while line != "":
            line = line.split()
            if len(line) != 2:
                self.log.write("ANT Load ERROR "+ ' '.join(line)+'\n')
            else:
                if [] not in self.get_vec(line):
                    self.antpair.append(tuple(line))
            line = iaf.readline()
        iaf.close()
        print ("Load ANT Completed...")

    def load_all(self):
        self.load_wordvec()
        self.load_synomym()
        self.load_antomym()

    def get_vec(self, words):
        if type(words) == type("haha"):
            words = [words]
        result = []
        for word in words:
            wordvec = self.word2vec.get(word, [])
            if wordvec == []:
                # print "Get Vector ERROR: " + word
                self.log.write("Get Vector ERROR: " + word + '\n')
            result.append(wordvec)
        return result

    def average_synonym_distance(self):
        total_distance = 0.0
        total_number = len(self.synpair)
        for pair in self.synpair:
            vecs = self.get_vec(pair)
            if vecs[0] != [] and vecs[1] != []:
                total_distance += cosine_distance(vecs[0], vecs[1])
            else:
                total_number -= 1
        print total_number
        return total_distance / total_number

    def average_antonym_distance(self):
        total_distance = 0.0
        total_number = len(self.antpair)
        for pair in self.antpair:
            vecs = self.get_vec(pair)
            if vecs[0] != [] and vecs[1] != []:
                total_distance += cosine_distance(vecs[0], vecs[1])
            else:
                total_number -= 1
        print total_number
        return total_distance / total_number


    def average_rand_samples_distance(self, other_weight = None, num = 50000):
        dics = None
        if other_weight != None:
	    dics = other_weight
        else:
            dics = self.word2vec.values()
        total_distance = 0.0
        length = len(dics)
        for i in xrange(num):
            idx1 = random.randint(0, length-1)
            idx2 = random.randint(0, length-1)
            vec1 = dics[idx1]
            vec2 = dics[idx2]
            total_distance += cosine_distance(vec1, vec2)
        return total_distance / num


if __name__ == '__main__':
    g = GetWordVector()
    g.load_all()
    print g.average_synonym_distance()
    print g.average_antonym_distance()
    print g.average_rand_samples_distance()
