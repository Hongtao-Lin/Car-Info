 #coding=utf8

import os, re, sys, MySQLdb, random, copy
import xlrd, xlwt, json, cPickle
import jieba
import numpy as np
import joblib

from scipy.sparse import *
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
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
stopwords = [ unicode(line.rstrip('\n')) for line in open('model/stopwords.txt')]
tag_dic = {}
word_dic = {}
word_idx = {}
vec_type = "segment"
info_data = "info2_cleaner"
limitation = ""


i = 0
for tag in tags:
	tag_dic[tag] = i
	i += 1
for tag in all_tags:
	if tag in tags:
		continue
	limitation += "label != '%s' and " % tag
limitation = limitation[:-5]


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

def sen2vec(data, vec_type="segment"):
	global word_dic
	vec = np.zeros([len(word_dic.keys())], dtype=float)
	seg_list = get_seg_list(data, vec_type)	
	idx = 0
	while idx < len(seg_list):
		seg = seg_list[idx]
		if is_number(seg):
			seg, idx = judge_number(idx, seg_list)
		if seg in stopwords:
			idx += 1
			continue
		if seg not in word_dic:
			idx += 1
			continue
		# counting word occurance in sentence.
		vec[word_dic[seg]] += 1
		idx += 1
	return vec

def set_word_dic(dump_num, vec_type):
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
				# print seg, idx
				seg, idx = judge_number(idx, seg_list)
			if seg in stopwords:
				idx += 1
				continue
			if seg in stat:
				stat[seg] += 1
			else:
				stat[seg] = 1
			idx += 1
	filter_freq = 3
	for k in stat.keys():
		# filter out those low-freq.
		if stat[k] <= filter_freq:
			stat.pop(k)

	i = 0
	for k in stat.keys():
		word_dic[k] = i 	# {"good", idx} (location)
		i += 1
	print "load total number: ", total_count
	print "init complete"
	cur.close()
	conn.close()

def shuffle_comments():

	conn, cur = init_mysql()

	# shuffle comments.
	print "shuffling comments..."

	cur.execute("create table car.c select * from comments limit 1")
	cur.execute("delete from car.c")

	cur.execute("select count(*) from car.comments;")
	total_comment = int(cur.fetchone()[0])
	chunk_size = 50000
	print "total comment is: ", total_comment

	for chunk in range( (total_comment-1) / chunk_size + 1):
		print "processing %d / %d" % (chunk, total_comment / chunk_size)
		sql = 'insert into car.c select * from car.comments order by rand() limit %d, %d' % (chunk*chunk_size, chunk_size)
		if chunk == (total_comment-1) / chunk_size:
			sql = 'insert into car.c select * from car.comments order by rand() limit %d, %d' % (chunk*chunk_size, total_comment % chunk_size)
		cur.execute(sql)
		conn.commit()

	cur.execute("drop table comments;")
	cur.execute("rename table c as comments;")
	conn.commit()

	cur.close()
	conn.close()

def shuffle_info():
	conn, cur = init_mysql()

	# shuffle comments.
	print "shuffling info..."

	cur.execute("create table car.c select * from info3 limit 1")
	cur.execute("delete from car.c")

	cur.execute("select count(*) from car.info3;")
	total_comment = int(cur.fetchone()[0])
	chunk_size = 50000
	print "total info is: ", total_comment

	for chunk in range( (total_comment-1) / chunk_size + 1):
		print "processing %d / %d" % (chunk, total_comment / chunk_size)
		sql = 'insert into car.c select * from car.info3 order by rand() limit %d, %d' % (chunk*chunk_size, chunk_size)
		if chunk == (total_comment-1) / chunk_size:
			sql = 'insert into car.c select * from car.info3 order by rand() limit %d, %d' % (chunk*chunk_size, total_comment % chunk_size)
		cur.execute(sql)
		conn.commit()

	conn.commit()

	cur.close()
	conn.close()

# Tools for processing...
def comment_to_info(comment_num):
	conn, cur = init_mysql()

	print "Converting comments into info..."
	# sql = 'delete from car.%s;' % info_data
	# cur.execute(sql)
	# conn.commit()
	sql = 'select count(*) from comments where series = "宝马3系"'
	cur.execute(sql)
	count = int(cur.fetchone()[0])
	if count > comment_num:
		count = comment_num
	print "total count:", count
	num = 0
	prog = 0
	short_data = 0
	long_data = 0
	total_count = 0
	chunk_size = 50000
	filter_short_len = 3
	filter_long_len = 100
	idx = 0
	for chunk in range((comment_num-1)/ chunk_size + 1):
		sql = 'select distinct * from comments limit %d, %d' % (chunk*chunk_size, chunk_size)
		if chunk == (comment_num-1)/ chunk_size:
			sql = 'select distinct * from comments limit %d, %d' % (chunk*chunk_size, comment_num % chunk_size)
		cur.execute(sql)
		for r in cur.fetchall():
			if num >= count*prog/100:
				print "processing %d%% ..." % prog
				prog += 1
			num += 1
			i = 6
			for tag in all_tags:
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
				web = r[4]
				idx += 1
				sql = "insert into car.%s values(%d, '%s','%s','%s','%s','%s','%s','%s', '%s', '%s');" % (info_data, idx, tag, '', '0', r[1], r[2], r[4], r[5], r[i], r[3])
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

def dump_data(dump_num, dump_file, vec_type="segment"):
	set_word_dic(dump_num, vec_type)
	print "dump data..."
	conn, cur = init_mysql()
	i = 0
	num = 0
	prog = 0

	# get # of overall data
	sql = 'select count(*) from car.%s where %s;' % (info_data, limitation)
	cur.execute(sql)
	total_num = int(cur.fetchone()[0])
	if total_num > dump_num:
		total_num = dump_num

	print "total number of data: %d" % total_num
	print "vector len: ", len(word_dic.keys())
	# total_num = 10000
	x = lil_matrix((total_num, len(word_dic.keys())), dtype=float)
	y = lil_matrix((total_num, 1), dtype=int)
	idx = lil_matrix((total_num, 1), dtype=int)
	chunk = 0
	chunk_size = 10000
	for chunk in range((total_num-1)/ chunk_size + 1):
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
		cPickle.dump(x, f)
		cPickle.dump(y, f)
		cPickle.dump(idx, f)
	cur.close()
	conn.close()
	print "Dump data complete"

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

			# update word_dic
			i = 0
			word_dic1 = copy.deepcopy(word_dic)
			for k,v in word_dic1.items():
				if v not in idx_new:
					word_dic1.pop(k)
				else:
					word_dic[k] = i
					i += 1

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
			mi = normalized_mutual_info_score(np.array(train_x[:,i].todense()).ravel(), train_y1)
			val[i] = mi
		for K in K_list:
			fname = dump_file.split(".")[0]+"_"+feat_type+"_"+str(K)+".pkl"
			if os.path.exists(fname):
				continue


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
					word_dic[k] = i
					i += 1

			with open(fname, "wb") as f:
				cPickle.dump(word_dic1, f)
				cPickle.dump(vec_type, f)
				cPickle.dump(x1.tolil(), f)
				cPickle.dump(y.tolil(), f)
				cPickle.dump(idx.tolil(), f)


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

def bayes_train(load_file, bayes_type, ratio):
	print "begin traning bayes..."
	conn, cur = init_mysql()

	with open(load_file) as f:
		word_dic = cPickle.load(f)
		vec_type = cPickle.load(f)
		x = cPickle.load(f)
		y = cPickle.load(f)
		idx = cPickle.load(f)
		x = x.tocsr()
		y = np.array(y.todense()).ravel()
		idx = np.array(idx.todense()).ravel()
	print "load data complete"
	# x = preprocessing.normalize(x)

	# x, y, idx = shuffle(x, y, idx, random_state = 42)

	if bayes_type == "m":
		clf = naive_bayes.MultinomialNB()
	else:
		clf = naive_bayes.BernoulliNB()

	train_len = int(0.8*ratio*y.shape[0])
	train_y = y[:train_len]
	train_x = x[:train_len]

	clf.fit(train_x, train_y)

	sql = 'select count(*) from car.info where %s' % limitation
	cur.execute(sql)
	total_count = int(cur.fetchone()[0])
	print total_count
	test_len = int(0.2*total_count)
	# test_len = 1000
	test_y = []
	pred_y = []
	# test_x = []

	sql = 'select comment, label from car.info where %s limit %d, %d' % (limitation, total_count-test_len, total_count)
	test_count = cur.execute(sql)
	filename = "res/error_log_%s_%s.txt" % (vec_type, bayes_type)
	f = open(filename, "w")

	for (data, label, url) in cur.fetchall():
		# print len(pred_y)
		test_x = sen2vec(data, vec_type)
		pred_y.append(clf.predict(test_x))
		test_y.append(tag_dic[label])
		if ratio == 1:
			if test_y[-1] == pred_y[-1]:
				continue
			try:
				info = "\npredict: %s\ntrue: %s\n%s\n%s\n" % (tags[pred_y[-1]], tags[test_y[-1]], url, data)
				f.write(info)
			except:
				print "Writing error!\n", sql

	acc = accuracy_score(test_y, pred_y)
	score = np.empty([3, len(tags)], dtype=float)
	score[0], score[1], score[2], support =  precision_recall_fscore_support(test_y, pred_y)
	macro = np.mean(score, axis=1)
	micro = np.mean(score*support, axis=1)/np.mean(support)
		
	print "##########"
	print "ratio:", ratio
	print "type:", bayes_type
	print acc
	print score[0]
	print macro[0], micro[0]
	print score[1]
	print macro[1], micro[1]
	print score[2]
	print macro[2], micro[2]
	print support
	print "##########"
	with open("model/bayes_model.pkl", "w") as f:
		cPickle.dump(clf, f)
		cPickle.dump(word_dic, f)
		cPickle.dump(vec_type, f)
	cur.close()
	conn.close()
	return acc

def svm_train(load_file, C, ratio, add_info):
	print "begin traning svm..."
	conn, cur = init_mysql()

	with open(load_file) as f:
		word_dic = cPickle.load(f)
		vec_type = cPickle.load(f)
		x = cPickle.load(f)
		y = cPickle.load(f)
		idx = cPickle.load(f)
		x = x.tocsr()
		y = np.array(y.todense()).ravel()
		idx = np.array(idx.todense()).ravel()
	print "load data complete"
	x = preprocessing.normalize(x)

	clf = LinearSVC(C = C)

	train_len = int(0.8*ratio*y.shape[0])
	train_y = y[:train_len]
	train_x = x[:train_len]
	train_x, train_y = shuffle(train_x, train_y, random_state = 42)

	clf.fit(train_x, train_y)

	dev_len = int(0.1*ratio*y.shape[0])
	dev_y = y[train_len:train_len+dev_len]
	dev_x = x[train_len:train_len+dev_len]


	test_len = int(0.1*ratio*y.shape[0])
	test_y = y[len(y)-test_len:]
	test_x = x[len(y)-test_len:]
	test_idx = idx[len(y)-test_len:]
	pred_y = []
	print test_len

	filename = "res/error_log_%s_%s_%s.txt" % (vec_type, "svm", add_info)
	f = open(filename, "w")

	pred_y = clf.predict(test_x)

	# if ratio == 1:
	# 	for i in xrange(test_len):
	# 		if test_y[i] == pred_y[i]:
	# 			continue
	# 		try:
	# 			sql = 'select comment, label, url from car.%s where id = %d' % (info_data, test_idx[i])
	# 			cur.execute(sql)
	# 			r = cur.fetchone()
	# 			info = "\npredict: %s\ntrue: %s\n%s\n%s\n" % (tags[pred_y[i]], tags[test_y[i]], r[-1], r[1])
	# 			f.write(info)
	# 		except:
	# 			# a = 1
	# 			# print "Writing error!\n", sql

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

# Tools for predicting...
def predict_sentence_detail(clf, data):
	global word_dic, vec_type
	test_x = sen2vec(data, vec_type).reshape(1,-1)
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
	test_x = sen2vec(data, vec_type).reshape(1,-1)
	pred_y = clf.predict(test_x)
	arr = clf.decision_function(test_x)[0]
	top = (-arr).argsort()[:3]

	label = tags[pred_y]
	### for other
	if arr[top[0]] < 0.5:
		label = "neutral"

	return label


def info_to_show():
	conn, cur = init_mysql()
	spec = ["宝马3系2016款328ixDriveM运动型", "宝马3系2015款320Li超悦版时尚型", "宝马3系2016款328Li豪华设计套装", '宝马3系2015款316i运动设计套装']
	web = ["yiche", "autohome", "pcauto", "sohu"]
	node_list = ["space","power","operation","oilwear","comfort","appearance","decoration","costperformance","failure","maintenance"];

	for s in spec:
		for n in node_list:
			fids = []
			print "YES!"
			sql = 'select distinct id from show_comment where spec="%s"and label = "%s" group by web order by date desc;' % (s, n)
			count = cur.execute(sql)
			tmp = cur.fetchall()
			for i in tmp:
				fids.append(int(i[0]))
			print fids
			if len(fids) == 1:
				t = fids[0]
				sql = "select id from show_comment where spec = '%s' and label = '%s' and id != %d order by date desc limit %d, %d" % (s, n, t, count, 5-count)
			elif fids == []:
				sql = "select id from show_comment where spec = '%s' and label = '%s' order by date desc limit %d, %d" % (s, n, 0, 5)
			else:
				sql = "select id from show_comment where spec = '%s' and label = '%s' and id not in %s order by date desc limit %d, %d" % (s, n, str(tuple(fids)), count, 5-count)
			cur.execute(sql)
			tmp = cur.fetchall()
			ids = []
			for i in tmp:
				ids.append(int(i[0]))
			print ids
			if fids != []:
				if len(fids) == 1:
					sql = "update show_comment set first_show = 2 where id = %d" % fids[0]
				else:
					sql = "update show_comment set first_show = 2 where id in %s" % str(tuple(fids))
				print sql
				cur.execute(sql)
				conn.commit()
			if ids != []:
				if len(ids) == 1:
					sql = "update show_comment set first_show = 1 where id = %d" % ids[0]
				else:
					sql = "update show_comment set first_show = 1 where id in %s" % str(tuple(ids))
				print sql
				cur.execute(sql)
				conn.commit()
	cur.close()
	conn.close()

def info_to_cut(dump_num):
	conn, cur = init_mysql()
	
	i = 0
	num = 0
	prog = 0

	# get # of overall data
	sql = 'select count(*) from car.%s where %s;' % (info_data, limitation)
	cur.execute(sql)
	total_num = int(cur.fetchone()[0])
	if total_num > dump_num:
		total_num = dump_num
	print "total number of data: %d" % total_num

	chunk = 0
	chunk_size = 10000
	f = open("res/segment.txt", "w")
	for chunk in range((total_num-1)/ chunk_size + 1):
		sql = 'select comment from car.%s where %s limit %d, %d;' % (info_data, limitation, chunk*chunk_size, chunk_size)
		if chunk == (total_num-1)/ chunk_size:
			sql = 'select comment from car.%s where %s limit %d, %d;' % (info_data, limitation, chunk*chunk_size, total_num % (chunk_size+1) )
		cur.execute(sql)
		for r in cur.fetchall():
			data = r[0]
			if num >= total_num*prog/100:
				print "processing %d%% ..." % prog
				prog += 1
			num += 1 
			sen_list = re.split("\)|\(|）|（|~|\?|？|！|!|，|。|,|\s", data)
			# sen_list = re.split("，|。|,|\s", data)
			data = " ".join(sen_list)
			# print data
			data = " ".join(jieba.cut(data))
			i += 1
			f.write(data)
			f.write("\n")	
	f.close()
	cur.close()
	conn.close()

def neutral_to_tag(load_file):
	global word_dic
	print "begin processing..."
	conn, cur = init_mysql()

	clf = init_model(load_file)

	sql = 'select count(*) from car.info2_cleaner where web="xcar" and label="neutral" '
	cur.execute(sql)
	total_count = int(cur.fetchone()[0])
	cur.execute(sql)
	conn.commit()

	sql = 'select id, comment from car.info2_cleaner where web="xcar" and label="neutral"'
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
		if np.max(arr) > 5 and len(data[-1].encode("utf8")) < 100:
			label = tags[pred_y]
			cnt += 1
			f.write(str(data[0]) + data[1] + label + str(len(data[-1].encode("utf8"))) + "\n")
			print data[0], data[1], label, len(data[-1].encode("utf8"))
			sql="update info2_cleaner set label = '%s' where id = %d" % (label, data[0])
			cur.execute(sql)
			conn.commit()
	cur.close()
	conn.close()
	print cnt

def sentence_to_bin(offset, dump_num, dump_file):
	conn, cur = init_mysql()
	info_data = "info2_cleaner"
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
			if len(seg_list) > 75:
				continue
			out = str(tag_dic[r[1]]) + " " + " ".join(seg_list)
			# print out
			i += 1
			f.write(out)
			f.write("\n")	
	f.close()
	cur.close()
	conn.close()

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
	chunk_size = 50000
	print "total info is: ", total_comment

	sql = 'select id, tendency, spec, label, comment, url from car.info where spec = "宝马3系2016款328ixDriveM运动型" or spec = "宝马3系2015款320Li超悦版时尚型" or spec = "  马3  2016款328Li豪华设计套装" or spec = "宝马3系2015款316i运动设计套装";'
	cur.execute(sql)
	r = cur.fetchall()
	j = json.dumps(r)

	out = open("testset.json", "w")
	out.write(j)

	cur.close()
	conn.close()

if __name__ == "__main__":

	# neutral_to_tag("model/svm_model_best2.pkl")
	sentence_to_bin(0, 160000, "../sent-conv-torch/data/custom.train")
	sentence_to_bin(160000, 20000, "../sent-conv-torch/data/custom.dev")
	sentence_to_bin(180000, 20000, "../sent-conv-torch/data/custom.test")
	# shuffle_comments()
	# dump_data(900000, "data_segment_large.pkl")
	# comment_to_info(10000)
	# info_to_emo()
	# info_to_emo_dump()