 #coding=utf8

import os, re, sys, MySQLdb, random, copy, operator
import xlrd, xlwt, json, cPickle, h5py
import jieba
import numpy as np
import joblib
import cProfile

from scipy.sparse import *
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support

from nltk import bigrams
reload(sys)
sys.setdefaultencoding('utf8')

# global values
# tags = ['space','power','operation','oilwear','comfort','appearance','decoration','costperformance', 'configuration', 'failure', 'maintenance','neutral']
# tags = ['space','power','operation','oilwear','comfort','appearance','decoration','costperformance', 'configuration', 'failure', 'maintenance']
tags = ['space','power','operation','oilwear','comfort','appearance','decoration','costperformance', 'failure', 'maintenance', 'neutral']
all_tags = ['advantage','shortcoming','space','power','operation','oilwear','comfort','appearance','decoration','costperformance', 'configuration', 'failure', 'maintenance','other','neutral']
# limitation = 'label != "advantage" and label != "shortcoming" and label != "other" and label != "neutral"'
tag_dic = {}
word_dic = {}
word_idx = {}
vec_type = "segment"
info_data = "info4_2"	# in mysql
limitation = ""
stopwords = []

def init():
	global limitation, stopwords
	stopwords = [ unicode(line.rstrip('\n')) for line in open('model/stopwords.txt')]
	i = 0
	for tag in tags:
		tag_dic[tag] = i
		i += 1
	for tag in all_tags:
		if tag in tags:
			continue
		limitation += "label != '%s' and " % tag
	limitation = limitation[:-5]
init()

def init_model(load_file):
	global word_dic, vec_type
	with open(load_file, "r") as f:
		clf = cPickle.load(f)
		word_dic = cPickle.load(f)
		# vec_type = cPickle.load(f)
	for (k, v) in word_dic.items():
		word_idx[v] = k

	return clf

def init_mysql():
	conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='1234',db='car',port=3306)
	cur=conn.cursor()
	conn.set_character_set('utf8')
	cur.execute('SET NAMES utf8;')
	cur.execute('SET CHARACTER SET utf8;')
	cur.execute('SET character_set_connection=utf8;')
	return conn, cur

def get_seg_list(data, vec_type):
	data = data.upper().decode('utf8')
	if vec_type == "bigram":
		seg_list = [bigram[0]+bigram[1] for bigram in bigrams(data.decode('utf8'))]
	elif vec_type == "unigram":
		seg_list = [unigram for unigram in data.decode('utf8')]
	elif vec_type == "combined":
		seg_list = [bigram[0]+bigram[1] for bigram in bigrams(data.decode('utf8'))]
		seg_list += [unigram for unigram in data.decode('utf8')]
	elif vec_type == "combined2":
		seg_list = [bigram[0]+bigram[1] for bigram in bigrams(data.decode('utf8'))]
		seg_list += [unigram for unigram in data.decode('utf8')]
		seg_list += [segment for segment in jieba.cut(data.decode('utf8'))]
	else:
		seg_list = [segment for segment in jieba.cut(data.decode('utf8'))]
		# print seg_list
		# seg_list = jieba.cut_for_search(data.decode('utf8'))
	return seg_list

def is_number(seg):
	# return False
	try:
		float(seg)
		return True
	except:
		return False

# Whether phrase begins from `idx` contains a digit or date. 
def judge_number(idx, seg_list):

	seg = seg_list[idx]
	if '.' in seg:
		seg = 'FLOAT'
	else:
		# if is a DATE! pattern: 
		if idx+1 < len(seg_list):
			if seg_list[idx+1] == u"年":
				if idx+2 < len(seg_list)  and is_number(seg_list[idx+1]) and seg_list[idx+2] == u"月":
					idx += 2
					if idx+2 < len(seg_list)  and is_number(seg_list[idx+1]) and (seg_list[idx+2] == u"日" or seg_list[idx+2] == u"号"):
						idx += 2
					seg = 'DATE'
			elif seg_list[idx+1] == u"月":
				idx += 1
				if idx+2 < len(seg_list) and is_number(seg_list[idx+1]) and (seg_list[idx+2] == u"日" or seg_list[idx+2] == u"号"):
					idx += 2
					seg = 'DATE'
			elif seg_list[idx+1] == u"月份":
				idx += 1
				seg = 'DATE'
			elif seg_list[idx+1] == u"号":
				idx += 1
				seg = 'DATE'
			else:
				seg = 'INT'
		else:
			seg = 'INT'
	return seg, idx

# input a sentence, get a vector back.
# also, convert int, float into symbols.
def sen2vec(data, vec_type="segment"):
	global word_dic
	vec = np.zeros([len(word_dic.keys())], dtype=float)
	seg_list = get_seg_list(data, vec_type)	
	idx = 0
	while idx < len(seg_list):
		seg = seg_list[idx]
		if is_number(seg):
			if '.' in seg:
				seg = 'FLOAT'
			else:
				seg = 'INT'
			# seg, idx = judge_number(idx, seg_list)
		if seg in stopwords:
			idx += 1
			continue
		if seg not in word_dic:
			idx += 1
			continue
		# counting word occurance in sentence.
		vec[word_dic[seg]] += 1
		idx += 1
	vec = vec.reshape(1, -1)
	return vec

def set_word_dic(dump_num, vec_type="segment"):
	print "updating word dic..."
	conn, cur = init_mysql()

	sql = 'select comment from car.%s where %s limit %d' % (info_data, limitation, dump_num)
	total_count = cur.execute(sql)
	stat = {}
	for r in cur.fetchall():
		seg_list = get_seg_list(r[0], vec_type)
		idx = 0
		# print seg_list
		while idx < len(seg_list):
			seg = seg_list[idx]
			if is_number(seg):
				if '.' in seg:
					seg = 'FLOAT'
				else:
					seg = 'INT'
				# seg, idx = judge_number(idx, seg_list)
			if seg in stopwords:
				idx += 1
				continue
			if seg in stat:
				stat[seg] += 1
			else:
				stat[seg] = 1
			idx += 1
	filter_freq = 5
	for k in stat.keys():
		# filter out those low-freq.
		if stat[k] <= filter_freq:
			stat.pop(k)

	stat = sorted(stat.items(), key=operator.itemgetter(1)) # [('a',2),...]

	i = 0
	out = open("word_dic.out", "w")
	for k in stat:
		word_dic[k[0]] = i 	# {"good", idx} (location)
		i += 1
		out.write(k[0] + "\t" + str(k[1]) + "\n")
	out.close()
	print "load total number: ", total_count
	print "init complete"
	cur.close()
	conn.close()

# Tools for processing...

# Clean the original crawled data.
# The original data is from table `comment2` (FIXED in code)
# The destination table is the value in info_data.
def comment_to_info(comment_num):
	conn, cur = init_mysql()

	print "Converting comments into info..."

	sql = 'select count(*) from comments2'
	cur.execute(sql)
	count = int(cur.fetchone()[0])
	# in case comment_num larger than total count
	if count < comment_num or comment_num == -1:
		comment_num = count
	print "total count:", count
	num = 0
	prog = 0
	short_data = 0
	long_data = 0
	total_count = 0
	chunk_size = 50000
	filter_short_len = 4
	filter_long_len = 100
	for chunk in range((comment_num-1)/ chunk_size + 1):
		sql = 'select * from comments2 limit %d, %d' % (chunk*chunk_size, chunk_size)
		if chunk == (comment_num-1)/ chunk_size:
			sql = 'select * from comments2 limit %d, %d' % (chunk*chunk_size, comment_num % chunk_size)
		cur.execute(sql)
		for r in cur.fetchall():
			if num >= comment_num*prog/100:
				print "processing %d%% ..." % prog
				prog += 1
			num += 1
			i = 6
			r = list(r)
			for tag in all_tags:
				r[i] = r[i].strip()
				total_count += 1
				if "'" in r[i]:
					r = list(r)
					print "special text: ", r[i]
					r[i] = ''.join(r[i].split("'"))
				if "\\" in r[i]:
					r = list(r)
					print "special text: ", r[i]
					r[i] = ''.join(r[i].split("\\"))
				if r[i].strip() == "":
					i += 1
					continue
				if len(r[i]) <= 3*filter_short_len:	#filter out sentence with less than 3 chinese char.
					short_data += 1
					i += 1
					continue
				if len(r[i]) >= 3*filter_long_len:	#filter out sentence with less than 3 chinese char.
					long_data += 1
					i += 1
					continue
				sql = "insert into car.%s(label, showlabel, tendency, series, spec, web, url, comment, date) values('%s','%s','%s','%s','%s','%s','%s', '%s', '%s');" % (info_data, tag, '', '0', r[1], r[2], r[4], r[5], r[i], r[3])
				i += 1
				try:
					cur.execute(sql)
				except MySQLdb.Error,e:
					print num
					print sql
					print "Mysql Error %d: %s" % (e.args[0], e.args[1])
			conn.commit()

	print "short reply (less than %d): %d" % (filter_short_len, short_data)
	print "long reply (more than %d): %d" % (filter_long_len, long_data)

	cur.close()
	conn.close()

# Delete some invalid data from table (info_data).
def clean_info(comment_num):
	conn, cur = init_mysql()

	print "Cleaning texts in info..."

	sql = 'select count(*) from %s;' % info_data
	cur.execute(sql)
	count = int(cur.fetchone()[0])
	# in case comment_num larger than total count
	if count < comment_num or comment_num == -1:
		comment_num = count
	print "total count:", count
	num = 0
	prog = 0
	short_data = 0
	long_data = 0
	total_count = 0
	chunk_size = 50000
	filter_short_len = 4
	filter_long_len = 100
	rec = 0
	for chunk in range((comment_num-1)/ chunk_size + 1):
		sql = 'select * from %s limit %d, %d' % (info_data, chunk*chunk_size, chunk_size)
		if chunk == (comment_num-1)/ chunk_size:
			sql = 'select * from %s limit %d, %d' % (info_data, chunk*chunk_size, comment_num % chunk_size)
		cur.execute(sql)
		for r in cur.fetchall():
			if num >= comment_num*prog/100:
				print "processing %d%% ..." % prog
				prog += 1
			num += 1
			data = r[7].strip()
			if len(data) <= 3*filter_short_len or len(data) >= 3*filter_long_len:
				rec += 1
				sql = "delete from %s where id = %d" % (info_data, r[-1])
				cur.execute(sql)
				conn.commit()
	print "delete ", rec, " comments"
	cur.close()
	conn.close()

# Do shuffling in order to make the samples randomly distributed.
def shuffle_info():
	conn, cur = init_mysql()

	# shuffle comments.
	print "shuffling info..."

	cur.execute("create table car.c select * from info4_2 limit 1")
	cur.execute("delete from car.c")

	cur.execute("select count(*) from car.info4_2;")
	total_comment = int(cur.fetchone()[0])
	chunk_size = 50000
	print "total info is: ", total_comment

	for chunk in range( (total_comment-1) / chunk_size + 1):
		print "processing %d / %d" % (chunk, total_comment / chunk_size)
		sql = 'insert into car.c select * from car.info4_2 order by rand() limit %d, %d' % (chunk*chunk_size, chunk_size)
		if chunk == (total_comment-1) / chunk_size:
			sql = 'insert into car.c select * from car.info4_2 order by rand() limit %d, %d' % (chunk*chunk_size, total_comment % chunk_size)
		cur.execute(sql)
		conn.commit()

	# cur.execute("drop table car.info4_2")
	# cur.execute("rename table car.c as car.info4_2")

	conn.commit()

	cur.close()
	conn.close()

# Some so-called neutral comments (in website `xcar`), are actually NOT!
# Here I use the trained model to convert some of them back. 
def neutral_to_tag(load_file):
	global word_dic
	print "begin converting neutral to tag..."
	conn, cur = init_mysql()

	clf = init_model(load_file)

	sql = 'select count(*) from car.%s where web="xcar" and label="neutral" '% (info_data)
	cur.execute(sql)
	total_count = int(cur.fetchone()[0])
	cur.execute(sql)
	conn.commit()

	sql = 'select id, comment from car.%s where web="xcar" and label="neutral"' % (info_data)
	cur.execute(sql)
	datas = cur.fetchall()
	prog = 0
	cnt = 0
	f = open("neutral_to_tag.log", "w")
	print total_count
	for i in range(total_count):
		data = datas[i]
		if i >= total_count*prog/100:
			print "processing %d%% ..." % prog
			prog += 1
		test_x = sen2vec(data[-1], word_dic)
		pred_y = clf.predict(test_x)
		arr = clf.decision_function(test_x)[0]
		if np.max(arr) > 5 and len(data[-1].encode("utf8")) < 80:
			label = tags[pred_y]
			cnt += 1
			f.write(str(data[0]) + data[1] + label + str(len(data[-1].encode("utf8"))) + "\n")
			print data[0], data[1], label, np.max(arr)
			sql="update %s set label = '%s' where id = %d" % (info_data, label, data[0])
			cur.execute(sql)
			conn.commit()
	cur.close()
	conn.close()
	print cnt

# Dump data in table info_data into matrix. 
# Each entry in matrix corresponds to a sample, which is a bag-of-word vector.
# @param@ vec_type: how the sentence is converted into 'word' (may be segment, unigram, bigram etc.)		
def dump_data(dump_num, dump_file, vec_type="segment"):
	print "dump data..."
	conn, cur = init_mysql()
	i = 0
	num = 0
	prog = 0
	
	# get # of overall data
	sql = 'select count(*) from car.%s where %s;' % (info_data, limitation)
	cur.execute(sql)
	total_num = int(cur.fetchone()[0])
	if dump_num == -1:
		dump_num = total_num
	if total_num > dump_num:
		total_num = dump_num

	set_word_dic(dump_num, vec_type)

	print "total number of data: %d" % total_num
	print "vector len: ", len(word_dic.keys())
	# total_num = 10000
	x = lil_matrix((total_num, len(word_dic.keys())), dtype=float)
	y = lil_matrix((total_num, 1), dtype=int)
	idx = lil_matrix((total_num, 1), dtype=int)
	chunk = 0
	chunk_size = 50000
	for chunk in range((total_num-1)/ chunk_size + 1):
		print "ready to fetch data..."
		sql = 'select id, comment, label from car.%s where %s limit %d, %d;' % (info_data, limitation, chunk*chunk_size, chunk_size)
		if chunk == (total_num-1)/ chunk_size:
			sql = 'select id, comment, label from car.%s where %s limit %d, %d;' % (info_data, limitation, chunk*chunk_size, total_num % (chunk_size+1) )
		cur.execute(sql)
		for r in cur.fetchall():
			if num >= total_num*prog/100:
				print "processing %d%% ..." % prog
				prog += 1
			num += 1 
			x[i] = sen2vec(r[1], vec_type)
			idx[i, 0] = r[0]
			y[i, 0] = tag_dic[r[2]]
			i += 1

	with open(dump_file, "wb") as f:
		cPickle.dump(word_dic, f)
		cPickle.dump(vec_type, f)
		cPickle.dump(x.tocoo(), f)
		cPickle.dump(y.tocoo(), f)
		cPickle.dump(idx.tocoo(), f)
	cur.close()
	conn.close()
	print "Dump data complete"

# Select words from large vocab, reduce its size.
# feat_type may be chi-squared or mutual info.
def feat_selection(dump_file, feat_type, K_list = []):
	global word_dic

	if feat_type == "":
		return

	print "begin feature selection..."
	conn, cur = init_mysql()

	with open(dump_file) as f:
		word_dic = cPickle.load(f)
		vec_type = cPickle.load(f)
		x = cPickle.load(f)
		y = cPickle.load(f)
		idx = cPickle.load(f)
		x = x.tocsr()
		y = y.tocsr()
		idx = idx.tocsr()
	print "load data complete"

	train_len = int(0.8*y.shape[0])
	train_y = y[:train_len]
	train_x = x[:train_len]
	idx_new = None

	if feat_type == "chi2":
		print train_x.shape, np.array(train_y.todense()).ravel().shape
		for K in K_list:
			fname = dump_file.split(".")[0]+"_"+feat_type+"_"+str(K)+".pkl"
			if os.path.exists(fname):
				continue

			model = SelectKBest(chi2, k=K).fit(train_x, np.array(train_y.todense()).ravel())
			x1 = model.transform(x)
			idx_new = model.get_support(indices=True)
			s, p = model.get_params()["score_func"](train_x, np.array(train_y.todense()).ravel())
			# update word_dic
			i = 0
			word_dic1 = copy.deepcopy(word_dic)
			for k,v in word_dic1.items():
				if v not in idx_new:
					word_dic1.pop(k)
				else:
					word_dic1[k] = i
					i += 1

			# feat_dic = {}
			# for k in word_dic:
			# 	feat_dic[k] = s[word_dic[k]]

			# sorted_dic = sorted(feat_dic.items(), key=operator.itemgetter(1), reverse=True)
			# with open("res/feature_extraction_chi2.out", "w") as f:
			# 	for k,v in sorted_dic:
			# 		# print k, v
			# 		f.write(k + "\t" + str(v) + "\n")

			with open(fname, "wb") as f:
				cPickle.dump(word_dic1, f)
				cPickle.dump(vec_type, f)
				cPickle.dump(x1.tolil(), f)
				cPickle.dump(y.tolil(), f)
				cPickle.dump(idx.tolil(), f)

	elif feat_type == "mi":
		val = np.zeros(x.shape[1], dtype=float)
		# fit
		train_y1 = np.array(train_y.todense()).ravel()
		for i in range(x.shape[1]):
			print i
			mi = normalized_mutual_info_score(np.array(train_x[:,i].todense()).ravel(), train_y1)
			val[i] = mi
		for K in K_list:
			fname = dump_file.split(".")[0]+"_"+feat_type+"_"+str(K)+".pkl"
			# if os.path.exists(fname):
			# 	continue


			idx_new = (-val).argsort()[:K]
			# print val[idx_new]
			
			# transform
			x1 = x[:,idx_new]

			# update word_dic
			i = 0
			word_dic1 = copy.deepcopy(word_dic)
			for k,v in word_dic1.items():
				if v not in idx_new:
					word_dic1.pop(k)
				else:
					word_dic1[k] = i
					i += 1

			feat_dic = {}
			for k in word_dic:
				feat_dic[k] = val[word_dic[k]]

			sorted_dic = sorted(feat_dic.items(), key=operator.itemgetter(1), reverse=True)
			with open("res/feature_extraction_mi.out", "w") as f:
				for k,v in sorted_dic:
					f.write(k + "\t" + str(v) + "\n")


			# with open(fname, "wb") as f:
			# 	cPickle.dump(word_dic1, f)
			# 	cPickle.dump(vec_type, f)
			# 	cPickle.dump(x1.tolil(), f)
			# 	cPickle.dump(y.tolil(), f)
			# 	cPickle.dump(idx.tolil(), f)

	cur.close()
	conn.close()

# Tools for training...

def print_prf(test_y, pred_y):
	acc = accuracy_score(test_y, pred_y)
	score = np.empty([3, len(tags)], dtype=float)
	# precision, recall, f1
	score[0], score[1], score[2], support =  precision_recall_fscore_support(test_y, pred_y)
	macro = np.mean(score, axis=1)
	micro = np.mean(score*support, axis=1)/np.mean(support)

	print "##########"
	# print "ratio:", ratio
	print acc
	print score[0]
	print macro[0], micro[0]
	print score[1]
	print macro[1], micro[1]
	print score[2]
	print macro[2], micro[2]
	print support
	print "##########"
	return acc

def svm_train(load_file, C, ratio, add_info):
	print "begin traning svm..."
	conn, cur = init_mysql()

	with open(load_file) as f:
		word_dic = cPickle.load(f)
		vec_type = cPickle.load(f)
		x = cPickle.load(f).tolil()
		y = cPickle.load(f).tolil()
		idx = cPickle.load(f).tolil()
		# x = x.tocsr()
		# y = np.array(y.todense()).ravel()
		# idx = np.array(idx.todense()).ravel()
		y = y.todense().ravel().T
	print "load data complete"
	# x = preprocessing.normalize(x)

	clf = LinearSVC(C = C)

	train_len = int(0.8*ratio*y.shape[0])
	train_y = y[:train_len]
	train_x = x[:train_len]
	# train_x, train_y = shuffle(train_x, train_y, random_state = 42)

	clf.fit(train_x, train_y)

	dev_len = int(0.1*ratio*y.shape[0])
	dev_y = y[train_len:train_len+dev_len]
	dev_x = x[train_len:train_len+dev_len]


	test_len = int(0.1*ratio*y.shape[0])
	test_y = y[y.shape[0]-test_len:]
	test_x = x[y.shape[0]-test_len:]
	test_idx = idx[y.shape[0]-test_len:]
	pred_y = []
	print test_len

	file_name = "res/error_log_svm_" + load_file.split("data/data_")[-1]
	f = open(file_name, "w")

	pred_y = clf.predict(test_x)

	if ratio == 1:
		for i in xrange(test_len):
			if test_y[i] == pred_y[i]:
				continue
			try:
				sql = 'select comment, label, url from car.%s where id = %d' % (info_data, test_idx[i])
				cur.execute(sql)
				r = cur.fetchone()
				info = "\npredict: %s\ntrue: %s\n%s\n%s\n" % (tags[pred_y[i]], tags[test_y[i]], r[-1], r[1])
				f.write(info)
			except:
				a = 1
				# print "Writing error!\n", sql

	f.close()

	print "vec_type", vec_type
	print "train prf:"
	print_prf(train_y, clf.predict(train_x))
	print "dev prf:"
	print_prf(dev_y, clf.predict(dev_x))
	print "test prf:"
	acc = print_prf(test_y, pred_y)

	file_name = "model/svm_model_" + load_file.split("data/data_")[-1]
	print "dumpping model to ", file_name

	with open(file_name, "w") as f:
		cPickle.dump(clf, f)
		cPickle.dump(word_dic, f)
		cPickle.dump(vec_type, f)

	cur.close()
	conn.close()
	return acc

def lr_train(load_file, C, ratio, add_info):
	print "begin traning lr..."
	conn, cur = init_mysql()

	with open(load_file) as f:
		word_dic = cPickle.load(f)
		vec_type = cPickle.load(f)
		x = cPickle.load(f).tolil()
		y = cPickle.load(f).tolil()
		idx = cPickle.load(f).tolil()
		# x = x.tocsr()
		# y = np.array(y.todense()).ravel()
		# idx = np.array(idx.todense()).ravel()
	print "load data complete"
	x = preprocessing.normalize(x)

	clf = LogisticRegression(C = C)

	train_len = int(0.8*ratio*y.shape[0])
	train_y = y[:train_len]
	train_x = x[:train_len]
	train_x, train_y = shuffle(train_x, train_y, random_state = 42)

	clf.fit(train_x, train_y)

	dev_len = int(0.1*ratio*y.shape[0])
	dev_y = y[train_len:train_len+dev_len]
	dev_x = x[train_len:train_len+dev_len]


	test_len = int(0.1*ratio*y.shape[0])
	test_y = y[y.shape[0]-test_len:]
	test_x = x[y.shape[0]-test_len:]
	test_idx = idx[y.shape[0]-test_len:]
	pred_y = []
	print test_len

	file_name = "res/error_log_lr_" + load_file.split("data/data_")[-1]
	f = open(file_name, "w")

	pred_y = clf.predict(test_x)

	if ratio == 1:
		for i in xrange(test_len):
			if test_y[i] == pred_y[i]:
				continue
			try:
				sql = 'select comment, label, url from car.%s where id = %d' % (info_data, test_idx[i])
				cur.execute(sql)
				r = cur.fetchone()
				info = "\npredict: %s\ntrue: %s\n%s\n%s\n" % (tags[pred_y[i]], tags[test_y[i]], r[-1], r[1])
				f.write(info)
			except:
				a = 1
				# print "Writing error!\n", sql

	f.close()

	print "vec_type", vec_type
	print "train prf:"
	print_prf(train_y, clf.predict(train_x))
	print "dev prf:"
	print_prf(dev_y, clf.predict(dev_x))
	print "test prf:"
	acc = print_prf(test_y, pred_y)

	file_name = "model/lr_model_" + load_file.split("data/data_")[-1]
	print "dumpping model to ", file_name

	with open(file_name, "w") as f:
		cPickle.dump(clf, f)
		cPickle.dump(word_dic, f)
		cPickle.dump(vec_type, f)

	cur.close()
	conn.close()
	return acc

# Tools for predicting...

# get detailed information on SVM features.
def predict_sentence_detail(clf, data):
	global word_dic, vec_type
	test_x = sen2vec(data, vec_type)
	pred_y = clf.predict(test_x)
	arr = clf.decision_function(test_x)[0]
	print arr
	print "top 3 choices:"
	top = (-arr).argsort()[:3]
	# top = [tag_dic["neutral"]]
	header = '\t'
	for choice in top:
		header += '%s\t' % tags[choice]
	print header
	
	seg_list = jieba.cut(data.decode('utf8'))
	for seg in seg_list:
		if seg in stopwords:
			continue
		if seg not in word_dic:
			continue
		row = seg + '\t'
		for choice in top:
			row += "%.3f\t" % clf.coef_[[choice]][0][word_dic[seg]]
		print row
	total = "\t"
	for choice in top:
		total += "%.3f\t" % arr[choice]
	print total
	print data

	label = tags[pred_y]
	### for other
	if arr[top[0]] < 0.5:
		label = "other"

	return label

def predict_sentence(clf, data):
	global word_dic, vec_type
	test_x = sen2vec(data, vec_type)
	pred_y = clf.predict(test_x)
	arr = clf.decision_function(test_x)[0]
	top = (-arr).argsort()[:3]

	label = tags[pred_y]
	### for other
	if arr[top[0]] < 0.5:
		label = "neutral"

	return label


# Temp util

def sentence_to_bin(offset, dump_num, dump_file):
	conn, cur = init_mysql()
	i = 0
	num = 0
	prog = 0

	# get # of overall data
	sql = 'select count(*) from car.%s where %s;' % (info_data, limitation)
	cur.execute(sql)
	total_num = int(cur.fetchone()[0])
	print total_num
	if total_num > dump_num:
		total_num = dump_num
	print "total number of data: %d" % total_num

	chunk = 0
	chunk_size = 10000
	f = open(dump_file, "w")
	for chunk in range((total_num-1)/ chunk_size + 1):
		sql = 'select comment, label from car.%s where %s limit %d, %d;' % (info_data, limitation, chunk*chunk_size+offset, chunk_size)
		if chunk == (total_num-1)/ chunk_size:
			sql = 'select comment, label from car.%s where %s limit %d, %d;' % (info_data, limitation, chunk*chunk_size+offset, total_num % (chunk_size+1) )
		cur.execute(sql)
		for r in cur.fetchall():
			data = r[0]
			if num >= total_num*prog/100:
				print "processing %d%% ..." % prog
				prog += 1
			num += 1 
			data = " ".join(data.split("\n"))
			seg_list = get_seg_list(data, "segment")
			if len(seg_list) > 150:
				continue
			out = str(tag_dic[r[1]]) + " " + " ".join(seg_list)
			# print out
			i += 1
			f.write(out)
			f.write("\n")	
	f.close()
	cur.close()
	conn.close()

	# f = open("passage_data.out", "r")
	# f2 = open("../sent-conv-torch/data/passage.test", "w")
	# for line in f.readlines():
	# 	r = line.split("\t")
	# 	seg_list = get_seg_list(r[0], "segment")
	# 	out = r[1].strip() + " " + " ".join(seg_list)
	# 	f2.write(out)
	# 	f2.write("\n")	
	# f.close()

# For sentiment analysis.
def info_to_emo():
	conn, cur = init_mysql()

	# shuffle comments.
	print "Converting comments..."

	cur.execute("select count(*) from car.info2_cleaner;")
	total_comment = int(cur.fetchone()[0])
	chunk_size = 50000
	print "total info is: ", total_comment

	for chunk in range( (total_comment-1) / chunk_size + 1):
		print "processing %d / %d" % (chunk, total_comment / chunk_size)
		sql = 'select * from car.info2_cleaner order by rand() limit %d, %d' % (chunk*chunk_size, chunk_size)
		if chunk == (total_comment-1) / chunk_size:
			sql = 'select * from car.info2_cleaner order by rand() limit %d, %d' % (chunk*chunk_size, total_comment % chunk_size)
		cur.execute(sql)
		for r in cur.fetchall():
			r = list(r)
			label = r[0]
			tendency = None
			if label == "advantage":
				tendency = 1
			if label == "shortcoming":
				tendency = -1
			if tendency == None:
				continue
			r[2] = tendency

			sql='insert into info_emo(label, showlabel, tendency, series, spec, web, url, comment, date) values(%s , %s , %s , %s ,%s , %s , %s , %s ,%s);'
			# print r[:-1]
			cur.execute(sql, r[:-1])

	conn.commit()

	cur.close()
	conn.close()

def info_to_emo_dump():
	conn, cur = init_mysql()

	# shuffle comments.
	print "Converting comments..."

	cur.execute("select count(*) from car.info_emo;")
	total_comment = int(cur.fetchone()[0])
	# chunk_size = 50000
	print "total info is: ", total_comment

	# sql = 'select id, tendency, spec, label, comment, url from car.info where spec = "宝马3系2016款328ixDriveM运动型" or spec = "宝马3系2015款320Li超悦版时尚型" or spec = "  马3  2016款328Li豪华设计套装" or spec = "宝马3系2015款316i运动设计套装";'
	sql = 'select id, tendency, spec, comment, url from car.info_emo;'
	cur.execute(sql)
	r = cur.fetchall()
	j = json.dumps(r)

	out = open("sentiment.json", "w")
	out.write(j)

	cur.close()
	conn.close()

def sentiment_to_show():
	conn, cur = init_mysql()

	spec = ["宝马3系2016款328ixDriveM运动型", "宝马3系2015款320Li超悦版时尚型", "宝马3系2016款328Li豪华设计套装", '宝马3系2015款316i运动设计套装']
	node_list = ["space","power","operation","oilwear","comfort","appearance","decoration","costperformance","failure","maintenance"];

	for s in spec:
		sql = "select * from show_sentiments where spec = \"%s\"" % s
		cur.execute(sql)
		if cur.rowcount == 0:
			sql = "insert into show_sentiments(spec) values ('%s')" % s
			cur.execute(sql)
			conn.commit()

		for n in node_list:
			fids = []
			sql = 'select tendency from info2_cleaner where spec = "%s" and label = "%s" ' % (s, n)
			count = cur.execute(sql)
			tmp = cur.fetchall()
			count_list = [0, 0, 0] # bad, neutral, good
			for t in tmp:
				count_list[int(t[0])+1] += 1
			# print count_list

			info = "%s/%s/%s" % (count_list[2], count_list[1], count_list[0])
			sql = 'update show_sentiments set %s = "%s" where spec = "%s"' % (n, info, s)
			print sql
			cur.execute(sql)
			conn.commit()

	cur.close()
	conn.close()

def keyword_to_show():
	conn, cur = init_mysql()

	top = 10

	book = xlrd.open_workbook("keywords.xls")
	for i in range(book.nsheets):
		sh = book.sheet_by_index(i)
		spec = sh.name
		num = min(top, sh.nrows)
		for rx in range(1, num):
			r = sh.row(rx)
			keyword = r[1].value + r[2].value
			weight = int(r[3].value)
			sql = "insert into show_keyword(spec, keyword, weight) values('%s', '%s', %d)" % (spec, keyword, weight)
			print sql
			cur.execute(sql)
			conn.commit()

	cur.close()
	conn.close()

def shuffle_data(load_file):
	conn, cur = init_mysql()

	with open(load_file) as f:
		word_dic = cPickle.load(f)
		vec_type = cPickle.load(f)
		x = cPickle.load(f).tolil()
		y = cPickle.load(f).tolil()
		idx = cPickle.load(f).tolil()
	print "load data complete"

	x, y, idx = shuffle(x, y, idx, random_state = 42)

	with open(load_file, "wb") as f:
		cPickle.dump(word_dic, f)
		cPickle.dump(vec_type, f)
		cPickle.dump(x.tocoo(), f)
		cPickle.dump(y.tocoo(), f)
		cPickle.dump(idx.tocoo(), f)
	cur.close()
	conn.close()
	return

if __name__ == "__main__":
	# clean_info(-1)
	# shuffle_info()
	# shuffle_data("data/data_segment_11_new.pkl")
	dump_data(-1, "data/data_bigram_11_new.pkl", "bigram")
	# set_word_dic(1500000)
	# neutral_to_tag("model/svm_model_bigram_11_cleaner.pkl")
	# comment_to_info(-1)
	# sentence_to_bin(0, 1237620, "../sent-conv-torch/data/custom.train")
	# sentence_to_bin(1237620, 154702, "../sent-conv-torch/data/custom.dev")
	# sentence_to_bin(1392322, 154703, "../sent-conv-torch/data/custom.test")
	# dump_data(900000, "data_segment_large.pkl")
	pass
