#!/usr/bin/python
# -*- coding:utf8 -*-

"""
Needs: numpy, keras, theano/tensorflow, HDFS5, h5py

For installation of h5py:
> sudo apt-get install libhdf5-dev
> sudo pip install h5py
"""

import jieba
import sys
import numpy as np
from random import shuffle
from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.optimizers import SGD

sense2vec = {
    "1": [1, 0, 0],
    "0": [0, 1, 0],
    "-1": [0, 0, 1]
}
vec2sense = ["1", "0", "-1"]

embedding_dims = 50
max_features = 115228
maxlen = 200
batch_size = 128
nb_filter = 250
filter_length = 3
hidden_dims = 300
nb_epoch = 1
dropout = 0.2

def build_vocab():
    f = open("vocab", "r")
    word2idx = {}
    idx2word = []
    for line in f.readlines():
        word = line[:-1]
        word2idx[word] = len(idx2word)
        idx2word.append(word)
    return word2idx, idx2word

def max_1d(X):
    return K.max(X, axis=1)


def get_data(txt_name="sentiment.txt", fnp="features.npy", tnp="targets.npy"):
    """
    Convert the data from the file to vector form.
    Some of the input sentence is too long to train
    so that I counted the length of sentences and
    its numbers, found that the sentences with more
    than 200 words only accounts for 140/54749, so
    I decided to ignore them.
    Finally I get features and targes with the shape
    of (54609, 200) (54609, 3), saving them to
    features.npy and targets.npy for convenience.
    """
    word2idx, idx2word = build_vocab()
    f = open(txt_name,"r")
    line = f.readline()
    targets = []
    features = []
    orignial = []
    maxlen = 0
    maxlen_list = []
    cnt = 0
    neg_cnt = 0
    pos_cnt = 0
    while line != "":
        word_seq = [0]*200
        line = line.split('\t')
        words = list(jieba.cut(line[1]))
        if len(words) <= 200:
            # print sense2vec[line[0]]
            try:
                targets.append(sense2vec[line[0]])
            except:
                targets.append(line[0])
            for i in xrange(len(words)):
                word_seq[i] = word2idx.get(words[i].encode('utf8'), 0)
            features.append(word_seq)
            orignial.append(' '.join(line))
        line = f.readline()
    f.close()
    features = np.array(features)
    targets = np.array(targets)
    print features.shape, targets.shape
    if fnp != None:
        np.save(fnp, features)
        np.save(tnp, targets)
    f = open("orignial.npy","w")
    for line in orignial:
        f.write(line+"\n")
    f.close()
    return features, targets, orignial

def load_data():
    f = open("orignial.npy", "r")
    lines = f.read().split("\n\n")
    return np.load("features.npy"), np.load("targets.npy"), lines

def build_net():
    """
    从数据中选取52000对作为训练数据，剩下的部分作为校验数据。
    值得注意的是这些数据只有优点和缺点，没有中性，并且这些数据噪声多。
    Train on 52000 samples, validate on 2609 samples.
    使用普通的CNN结构，随机初始化的embedding向量，效果：
    epoch1：loss: 0.3135 - acc: 0.8593 - val_loss: 0.2064 - val_acc: 0.9260
    epoch2：loss: 0.1306 - acc: 0.9553 - val_loss: 0.1542 - val_acc: 0.9494
    使用普通的CNN结构+预先训练的词向量，效果
    epoch1：loss: 0.2950 - acc: 0.8745 - val_loss: 0.2267 - val_acc: 0.9084
    epoch2：loss: 0.1485 - acc: 0.9463 - val_loss: 0.1534 - val_acc: 0.9475

    加入中性点之后，性能下降，不过由于数据噪声很多，所以还是有很多预测比标注
    更为合理的错误例子出现。
    epoch1: loss: 0.5526 - acc: 0.7747 - val_loss: 0.3716 - val_acc: 0.8643
    epoch2: loss: 0.1937 - acc: 0.9353 - val_loss: 0.2774 - val_acc: 0.9128
    """
    pre_weights = np.array([np.load("wordvec.npy")])
    pretrained_embedding = Embedding(max_features, embedding_dims, 
                                    input_length=maxlen, 
                                    weights=pre_weights, 
                                    trainable=False)
                                    # disable the update of par
    embedding = Embedding(max_features, embedding_dims, 
                          input_length=maxlen, init='uniform')
    model = Sequential()
    model.add(embedding)
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(Lambda(max_1d, output_shape=(nb_filter,)))
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print "Compile completed"
    return model
def from_json_to_txt():
    import json
    f = open('testset.json', "r")
    j = json.loads(f.read())
    o = open("testset.txt", "w")
    llll = []
    for i in xrange(len(j)):
        llll.append(str(j[i][0])+'\t'+j[i][4].encode('utf8')+'\n')
    # shuffle(llll)
    for l in llll:
        o.write(l)
    f.close()
    o.close()

def save_model(model, model_name):
    f = open(model_name,"w")
    f.write(model.to_json())
    f.close()
    model.save_weights(model_name+".weights")

def load_model(model_name):
    f = open(model_name, "r")
    json_string = f.read()
    model = model_from_json(json_string)
    f.close()
    model.load_weights(model_name+".weights")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print "Compile completed..."
    return model
                  
if __name__ == '__main__':

    #from_json_to_txt()

    features, targets, orignial = load_data()
    model = build_net()
    model.fit(features[:80000], targets[:80000],
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(features[80000:], targets[80000:]))

    model = load_model("withneutral_cnn.model")
    # np.save("after_class.npy", model.layers[0].get_weights())
    # Training
    # model.fit(features[:80000], targets[:80000],
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           validation_data=(features[80000:], targets[80000:]))
    # save_model(model, "withneutral_cnn.model")
    
    # Testing
    # correct = 0
    # total = len(features[80000:])
    # predicts = model.predict_on_batch(features[80000:])
    # for i in xrange(len(predicts)):
    #     p = np.argmax(predicts[i])
    #     y = np.argmax(targets[80000+i])
    #     if p != y:
    #         sys.stderr.write(vec2sense[p]+" "+orignial[i+80000]+"\n")
    #         sys.stderr.flush()
    #     else:
    #         correct += 1
    # print correct, "/", total, correct*1.0/total


    # Predict on test
    """
    features, id_, orignial = get_data(txt_name="testset.txt",
                                     fnp = None,
                                     tnp = None)
    predicts = model.predict_on_batch(features)
    correct = 0
    for i in xrange(len(predicts)):
        p = np.argmax(predicts[i])
        sys.stderr.write(vec2sense[p]+" "+orignial[i])
        sys.stderr.flush()
   """ 
