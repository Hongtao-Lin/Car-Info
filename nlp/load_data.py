 #coding=utf8

import os, re, sys, MySQLdb, random
import xlrd, xlwt, json, cPickle
import jieba
import numpy as np
import joblib
from scipy.sparse import *
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils import shuffle
from sklearn import naive_bayes
from sklearn.svm import LinearSVC, SVC
reload(sys)
sys.setdefaultencoding('utf8')

conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='1234',db='car',port=3306)
cur=conn.cursor()
conn.set_character_set('utf8')
cur.execute('SET NAMES utf8;')
cur.execute('SET CHARACTER SET utf8;')
cur.execute('SET character_set_connection=utf8;')

T=[]
sql=''
tags = ['space','power','operation','oilwear','comfort','appearance','decoration','costperformance', 'failure', 'maintenance']
dump_num = 5000000
dump_file = "data_10.pkl"
comment_num = 100000
limitation = 'label != "advantage" and label != "shortcoming" and label != "other"'

stat = {}
word_dic = {}
vec_len = 0
tag_dic = {}
i = 0
for tag in tags:
	tag_dic[tag] = i
	i += 1


stopwords = {}.fromkeys([ line.rstrip() for line in open('stopwords.txt') ])

def comment_to_info():
	# # shuffle comments.
	# print "shuffling comments..."
	# cur.execute('select * from car.comments')
	# data = list(cur.fetchall())
	# random.shuffle(data)
	# cur.execute("delete from car.comments;")
	# sql = "insert into car.comments values(%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s , %s, %s);"
	# for i in range(len(data)/1000):
	# 	cur.executemany(sql, data[1000*i:1000*(i+1)])
	# 	conn.commit()
	# cur.executemany(sql, data[1000*(i+1):])
	# conn.commit()

	print "Converting comments into info..."
	cur.execute('delete from car.info;')
	conn.commit()
	sql = 'select distinct * from comments limit %d' % comment_num
	# sql = 'select * from comments'
	count = cur.execute(sql)
	print "total count:", count
	data = cur.fetchall()
	num = 0
	prog = 0
	short_data = 0
	total_count = 0
	filter_len = 3
	for r in data:
		if num >= count*prog/100:
			print "processing %d%% ..." % prog
			prog += 1
		num += 1
		i = 6
		for tag in tags:
			total_count += 1
			if "'" in r[i]:
				r = list(r)
				print "special text: ", r[i]
				r[i] = ''.join(r[i].split("'"))

			if r[i].strip() == "":
				i += 1
				continue
			if len(r[i]) <= 3*filter_len:	#filter out sentence with less than 3 chinese char.
				short_data += 1
				i += 1
				continue
			web = r[4]
			sql = "insert into car.info values(%d, '%s','%s','%s','%s','%s','%s','%s', '%s', '%s');" % (0, tag, '', '0', r[1], r[2], r[4], r[5], r[i], r[3])
			i += 1
			try:
				cur.execute(sql)
			except MySQLdb.Error,e:
				print num
				print sql
				print "Mysql Error %d: %s" % (e.args[0], e.args[1])
		conn.commit()
	print "short reply (less than %d): %d" % (filter_len, short_data)

def init_dump():
	global dump_num
	print "init dump..."

	
	sql = 'select comment from car.info where %s limit %d' % (limitation, dump_num)
	total_count = cur.execute(sql)
	for r in cur.fetchall():
		seg_list = list(set(jieba.cut(r[0].decode('utf8'))))
		for seg in seg_list:
			if seg in stopwords:
				continue
			if seg in stat:
				stat[seg] += 1
			else:
				stat[seg] = 1
	filter_freq = 3
	for k in stat.keys():
		# filter out those low-freq.
		if stat[k] <= filter_freq:
			stat.pop(k)

	vec_len = len(stat.keys())
	i = 0
	for k in stat.keys():
		word_dic[k] = i 	# {"good", 12} (location)
		i += 1
	print "load total number: ", total_count
	print "init complete"

def dump_data():
	global dump_num
	print "dump data..."
	i = 0
	num = 0
	prog = 0
	sql = 'select id, comment, label from car.info where %s limit %d;' % (limitation, dump_num)
	total_num = cur.execute(sql)
	print "total number of data: %d" % total_num

	result = cur.fetchall()
	print "vector len: ", len(word_dic.keys())
	x = lil_matrix((total_num, len(word_dic.keys())), dtype=float)
	y = lil_matrix((total_num, 1), dtype=int)
	idx = lil_matrix((total_num, 1), dtype=int)
	# x = np.zeros([total_num, len(word_dic.keys())], dtype=int)
	# y = np.zeros([total_num], dtype=int)
	for r in result:
		if num >= total_num*prog/100:
			print "processing %d%% ..." % prog
			prog += 1
		num += 1 
		seg_list = jieba.cut(r[1].decode("utf-8"))
		for seg in seg_list:
			if seg in stopwords:
				continue
			if seg not in word_dic:
				continue
			# counting word occurance in sentence.
			x[i, word_dic[seg]] += 1
		idx[i, 0] = r[0]
		y[i, 0] = tag_dic[r[2]]
		i += 1
	with open(dump_file, "wb") as f:
		cPickle.dump(x, f)
		cPickle.dump(y, f)
		cPickle.dump(idx, f)
	print "Dump data complete"

def bayes_train(type, ratio):
	print "begin traning bayes..."
	with open(dump_file) as f:
		x = cPickle.load(f)
		y = cPickle.load(f)
		idx = cPickle.load(f)
		x = x.tocsr()
		y = np.array(y.todense()).ravel()
		idx = np.array(idx.todense()).ravel()
	print "load data complete"
	x = preprocessing.normalize(x)

	x, y, idx = shuffle(x, y, idx, random_state = 42)

	if type == "m":
		clf = naive_bayes.MultinomialNB()
	else:
		clf = naive_bayes.BernoulliNB()

	train_len = int(0.8*ratio*y.shape[0])
	test_len = ratio*y.shape[0] - train_len
	train_x = x[:train_len]
	test_x = x[train_len:ratio*y.shape[0]]
	train_y = y[:train_len]
	test_y = y[train_len:ratio*y.shape[0]]
	test_idx = idx[train_len:ratio*y.shape[0]]

	clf.fit(train_x, train_y)
	pred_y = clf.predict(test_x)

	acc = accuracy_score(test_y, pred_y)
	score = np.empty([3, len(tags)], dtype=float)
	score[0], score[1], score[2], support =  precision_recall_fscore_support(test_y, pred_y)
	macro = np.mean(score, axis=1)
	micro = np.mean(score*support, axis=1)/np.mean(support)

	f = open("error_log.txt", "w")
	for i in range(test_y.shape[0]):
		if test_y[i] == pred_y[i]:
			continue
		sql = "select url, comment from car.info where id = %d" % test_idx[i]
		cur.execute(sql)
		url, comment = cur.fetchone()
		info = "\npredict: %s\ntrue: %s\n%s\n%s\n" % (tags[pred_y[i]], tags[test_y[i]], url, comment)
		f.write(info)
	f.close()
	# with open("result.txt", "a") as f:
	# 	info = "\nBayes with %s and num %d: \n" % (type, train_len)
	# 	print info
	# 	f.write(info)
	# 	f.write(str(acc))
	print acc
	print score[0]
	print macro[0], micro[0]
	print score[1]
	print macro[1], micro[1]
	print score[2]
	print macro[2], micro[2]
	print support


## Num of var: C, loss, penalty, 
def svm_train(C, loss, penalty, ratio):
	print "begin traning svm..."
	with open(dump_file) as f:
		x = cPickle.load(f)
		y = cPickle.load(f)
		idx = cPickle.load(f)
		x = x.tocsr()
		y = np.array(y.todense()).ravel()
		idx = np.array(idx.todense()).ravel()
	print "load data complete"
	x = preprocessing.normalize(x)

	x, y, idx = shuffle(x, y, idx, random_state = 42)
	# x, y = shuffle(x, y, random_state = 42)

	# clf = OneVsOneClassifier(LinearSVC(C = C, loss = loss, penalty = penalty))
	clf = LinearSVC(C = C, loss = loss, penalty = penalty)
	# clf = SVC(C = C, kernel = "linear")

	train_len = int(0.8*ratio*y.shape[0])
	test_len = ratio*y.shape[0] - train_len
	train_x = x[:train_len]
	test_x = x[train_len:ratio*y.shape[0]]
	train_y = y[:train_len]
	test_y = y[train_len:ratio*y.shape[0]]
	test_idx = idx[train_len:ratio*y.shape[0]]

	clf.fit(train_x, train_y)
	pred_y = clf.predict(test_x)

	acc = accuracy_score(test_y, pred_y)
	score = np.empty([3, len(tags)], dtype=float)
	score[0], score[1], score[2], support =  precision_recall_fscore_support(test_y, pred_y)
	macro = np.mean(score, axis=1)
	micro = np.mean(score*support, axis=1)/np.mean(support)

	f = open("error_log.txt", "w")
	for i in range(test_y.shape[0]):
		if test_y[i] == pred_y[i]:
			continue
		sql = "select url, comment from car.info where id = %d" % test_idx[i]
		cur.execute(sql)
		url, comment = cur.fetchone()
		info = "\npredict: %s\ntrue: %s\n%s\n%s\n" % (tags[pred_y[i]], tags[test_y[i]], url, comment)
		f.write(info)
	f.close()
	# with open("result.txt", "a") as f:
	# 	info = "\nBayes with %s and num %d: \n" % (type, train_len)
	# 	print info
	# 	f.write(info)
	# 	f.write(str(acc))
	print acc
	print score[0]
	print macro[0], micro[0]
	print score[1]
	print macro[1], micro[1]
	print score[2]
	print macro[2], micro[2]
	print support



comment_to_info()
init_dump()
dump_data()
# bayes_train("m", 1)
# for type in ["m", "b"]:
# 	for ratio in [0.1, 0.2, 0.5, 1]:
# 		bayes_train(type, ratio)

svm_train(1, "squared_hinge", "l2", 1)
# for C in [1500, 5000]:
# 	for kernel in ["rbf"]:
# 		for ratio in [0.5]:
# 			svm_train(C, kernel, ratio)

cur.close()
conn.close()
