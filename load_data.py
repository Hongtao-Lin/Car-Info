 #coding=utf8

import os, re, sys, MySQLdb
import xlrd, xlwt, pickle, json
import jieba
import numpy as np
import joblib
from sklearn import naive_bayes
from sklearn.svm import SVC

reload(sys)
sys.setdefaultencoding('utf8')

conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='1234',db='mysql',port=3306)
cur=conn.cursor()
conn.set_character_set('utf8')
cur.execute('SET NAMES utf8;')
cur.execute('SET CHARACTER SET utf8;')
cur.execute('SET character_set_connection=utf8;')

T=[]
sql=''
tags = ['advantage','shortcoming','space','power','operation','oilwear','comfort','appearance','decoration','costperformance','failure','maintenance','other']
train_num = 40000
test_num = 10000
limitation = 'where label != "advantage" and label != "shortcoming" and label != "other"'

stat = {}
word_dic = {}
tag_dic = {}
vec_len = 0

stopwords = {}.fromkeys([ line.rstrip() for line in open('stopwords.txt') ])

def get_train_data():
	cur.execute('delete from car.info;')
	conn.commit()
	print "loading train data..."
	sql = 'select * from comments limit 0, %d' % train_num
	count = cur.execute(sql)
	train_data = cur.fetchall()
	num = 0
	prog = 0

	for r in train_data:
		if num >= count*prog/100:
			print "processing %d%% ..." % prog
			prog += 1
		num += 1
		i=5
		for tag in tags:
			if r[i].strip() == "":
				i += 1
				continue
			web = "autohome"
			sql = "insert into car.info values('%s','%s','%s','%s','%s','%s','%s', '%s', '%s');" % (tag, '', '0', r[1], r[2], web, r[4], r[i], r[3])
			i += 1
			try:
				cur.execute(sql)
			except MySQLdb.Error,e:
				print num
				print sql
				print "Mysql Error %d: %s" % (e.args[0], e.args[1])
		conn.commit()

def get_test_data():
	cur.execute('delete from car.test_info;')
	conn.commit()
	print "loading test data..."
	sql = 'select * from comments limit %d, %d' % (train_num, test_num)
	count = cur.execute(sql)
	test_data = cur.fetchall()
	num = 0
	prog = 0
	null_data = 0
	for r in test_data:
		if num >= count*prog/100:
			print "processing %d%% ..." % prog
			prog += 1
		num += 1
		i=5
		for tag in tags:
			if r[i].strip() == "":
				i += 1
				null_data += 1
				continue
			web = "autohome"
			sql="insert into car.test_info values('%s','%s','%s','%s','%s','%s','%s', '%s', '%s');" % (tag, '', '0', r[1], r[2], web, r[4], r[i], r[3])
			i += 1
			try:
				cur.execute(sql)
			except MySQLdb.Error,e:
				print num
				print sql
				print "Mysql Error %d: %s" % (e.args[0], e.args[1])
		conn.commit()
	print "null data is ", null_data
	print "count is ", count

def init_dump():
	global train_num
	print "init dump..."

	i = 0
	for tag in tags:
		tag_dic[tag] = i
		i += 1

	null_data = 0
	sql = 'select comment from car.info %s limit %d' % (limitation, train_num)
	total_count = cur.execute(sql)
	for r in cur.fetchall():
		seg_list = list(set(jieba.cut(r[0].decode('utf8'))))
		if seg_list == []:
			null_data += 1
			continue
		for seg in seg_list:
			if seg in stopwords:
				continue
			if seg in stat:
				stat[seg] += 1
			else:
				stat[seg] = 1
	filter_freq = 5
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

def dump_train_data():
	global train_num
	print "dump train data..."
	i = 0
	num = 0
	prog = 0
	sql = 'select comment, label from car.info %s limit 0, %d;' % (limitation, train_num)
	train_num = cur.execute(sql)
	print "total number of data: %d" % train_num

	result = cur.fetchall()
	print len(word_dic.keys())
	train_x = np.zeros([train_num, len(word_dic.keys())], dtype=int)
	train_y = np.zeros([train_num], dtype=int)
	null_seg = 0
	for r in result:
		if num >= train_num*prog/100:
			print "processing %d%% ..." % prog
			prog += 1
		num += 1 
		seg_list = jieba.cut(r[0].decode("utf-8"))
		if seg_list == []:
			null_seg += 1
			continue
		for seg in seg_list:
			if seg in stopwords:
				continue
			if seg not in word_dic:
				continue
			train_x[i][word_dic[seg]] += 1
		train_y[i] = tag_dic[r[1]]
		i += 1
	train_x = np.delete(train_x, np.s_[i:], 0)
	train_y = np.delete(train_y, np.s_[i:], 0)
	train = {}
	train["x"] = train_x.tolist()
	train["y"] = train_y.tolist()
	with open("train.json", "w") as f:
		json.dump(train, f)
		f.close()
	print "train data complete"

def dump_test_data():
	global test_num
	print "dump test data..."
	i = 0
	num = 0
	prog = 0
	sql = 'select comment, label from car.test_info %s limit %d, %d;' % (limitation, train_num, test_num)
	test_num = cur.execute(sql)
	print "total number of data: %d" % test_num
	result = cur.fetchall()
	test_x = np.zeros([test_num, len(word_dic.keys())], dtype=int)
	test_y = np.zeros([test_num], dtype=int)
	null_seg = 0
	for r in result:
		if num >= test_num*prog/100:
			print "processing %d%% ..." % prog
			prog += 1
		num += 1
		seg_list = jieba.cut(r[0].decode("utf-8"))
		if seg_list == []:
			null_seg += 1
			continue
		for seg in seg_list:
			if seg in stopwords:
				continue
			if seg not in word_dic:
				continue
			test_x[i][word_dic[seg]] += 1
		test_y[i] = tag_dic[r[1]]
		i += 1
	test_x = np.delete(test_x, np.s_[i:], 0)
	test_y = np.delete(test_y, np.s_[i:], 0)
	test = {}
	test["x"] = test_x.tolist()
	test["y"] = test_y.tolist()
	with open("test.json", "w") as f:
		json.dump(test, f)
		f.close()
	print "test data complete"

def bayes_train(type, ratio):
	print "begin traning bayes..."
	with open("train.json") as f1:
		train = json.load(f1)
	with open("test.json") as f2:
		test = json.load(f2)
	print "load data complete"
	if type == "b":
		clf = naive_bayes.BernoulliNB()
	else:
		clf = naive_bayes.MultinomialNB()
	train_len = int(ratio*len(train["y"]))
	test_len = int(0.5*len(test["y"]))
	train_x = train["x"][:train_len]
	train_y = train["y"][:train_len]
	test_x = test["x"][test_len:]
	test_y = test["y"][test_len:]
	clf.fit(train_x, train_y)
	pred_y = clf.predict(test_x)
	err_num = np.count_nonzero(pred_y - test_y)
	acc = float(len(test_x) - err_num) / len(test_x)
	with open("result.txt", "a") as f:
		info = "\nBayes with %s and num %d: \n" % (type, train_len)
		print info
		f.write(info)
		f.write(str(acc))
		f.close()
	print acc

def svm_train(C, kernel, ratio):
	print "begin traning svm..."
	with open("train.json") as f1:
		train = json.load(f1)
	with open("test.json") as f2:
		test = json.load(f2)
	print "load data complete"
	clf = SVC(C = C, kernel = kernel)

	train_len = int(ratio*len(train["y"]))
	test_len = int(ratio*len(test["y"]))
	train_x = train["x"][:train_len]
	train_y = train["y"][:train_len]
	test_x = test["x"][:test_len]
	test_y = test["y"][:test_len]
	clf.fit(train_x, train_y)
	pred_y = clf.predict(test_x)
	err_num = np.count_nonzero(pred_y - test_y)
	acc = float(test_len- err_num) / test_len
	with open("result.txt", "a") as f:
		info = "\nC = %d, kernel = %s with num %d \n" % (C, kernel, train_len)
		f.write(info)
		f.write(str(acc))
		f.close()
	print acc

# get_train_data()
# get_test_data()
# init_dump()
# dump_train_data()
# dump_test_data()
# bayes_train("m", 0.2)
for C in [1, 5]:
	for kernel in ["rbf"]:
		for ratio in [0.1, 0.3]:
			svm_train(C, kernel, ratio)

cur.close()
conn.close()
