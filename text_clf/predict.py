#coding=utf8

import os, re, sys, MySQLdb, random
import json, cPickle
import jieba
import numpy as np
from scipy.sparse import *
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn import naive_bayes
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support

from car_processing_tool import *
 

reload(sys)
sys.setdefaultencoding('utf8')

limitation = "label = 'advantage' or label = 'shortcoming' or label = 'other' "
load_file = "model/svm_model_segment_11_cleaner.pkl"

# load classfier, (and word dic!)
print "loading model..."
clf = init_model(load_file)
print "load complete"

def get_top_words(top_num):
	out = open("TopWords.txt", "w")
	print "top %d words for each class:\n" % top_num
	for tag in tag_dic:
		print tag
		out.write(tag + "\n")
		# for (k, v) in word_idx.items():
		# 	if v == u"INT":
		# 		print v, clf.coef_[tag_dic[tag]][k]
		for idx in clf.coef_[tag_dic[tag]].argsort()[-top_num:][::-1]:
			info = ''.join(list(word_idx[idx])) + "\t" + str(clf.coef_[tag_dic[tag]][idx])
			print info
			out.write(info.encode("utf8"))
			out.write("\n")
		out.write("\n")
	out.close()

test_p = []
test_s = []
# test_s.append(u'我朋友借去开了一天也说的确是好车，这里呼吁大家一句借车还是得看人，因为很多时候因为借车会闹出不少状况的，这个不多解释！')
# test_s.append(u'天窗、遮阳帘和车内灯光控制')
# test_s.append(u'车子整个都很满意，但有些地方也有一些瑕疵！！！')
# test_p.append(u'但也不算重         缺点：发动机声音有点大 这可能和没有隔音棉有关')
# test_p.append(u'''关于操控，依旧沿用我的感性标准来衡量。在我心里，驾驭Q7这样的大家伙，一点也不困难。开它的时候，一点没有笨重的感觉，反倒有一些小型车的那种灵活的感觉。对于我的每一个转向操作，它都能非常快的执行，虽然做不到毫无拖沓，但那微乎其微的延迟可以忽略不计。Q7在低速行驶的时候，方向盘非常轻，夸张地说，用一根手指就能操作它。而一旦速度上来以后，方向盘明显变沉重，完全没有轻飘飘的感觉。
# 车在行驶中的稳定性，我觉得也算是操控中的一项。因为如果一个车开起来晃晃悠悠，一点儿安全感都没有的话，还何谈操控？Q7的稳定性我觉得不错，在一些紧急情况下，比如紧急避让前车的时候，Q7都很稳，当然前提是不慌乱。能做到这一点，我想是因为quattro四驱以及出色的底盘和悬挂。''')
# test_p.append(u'''现在谈谈定3系的原因，看车的时候是带老婆去4S店里看的，3系加长的后排真的太TMD大了，完全够用，而且也让老婆试驾了，动力、操控、灵活这些是宝马的特长没的说。5系毕竟要比3系大一圈，现在城市道路这么拥挤，停车位又这么紧张，而且价格也要差10万左右。平均一年算开2万公里，10万也差不多能加7年油了，买5系更多的面子上好看一点。所以说3系更多的是自已用来开的，5系则更多的给别人看的。不过跑长途应该是5系舒服一点，但3系也不差，这就是我选择3系的原因。  3系的几个优点：挂倒档右后视镜自动下翻；两个后门和后挡风玻璃还有遮阳帘；后排右侧座位还有老板键可以调整副驾驶座椅；后备箱容量挺大的；座椅很底，坐进去像跑车的感觉；雨天路滑出弯给大油明显屁股甩过去，方向也偏了，还好及时回正方向，不然碰到倒鸭子了。3系几个不足的地方：这个后备箱里没有逃身装置，这种安全功能有总比没有好；方向盘中间部分凸起太高，方向盘要调的很低才能看到仪表盘小屏幕的内容；播放U盘里的MP3能不能随机播放，里面有近700首歌，一不小心就从头开始放了.''')
# test.append(u'车辆减震效果不太好，转弯时感觉车子后面有些甩尾。')
# print tags

def sentence():
	for data in test_s:
		predict_sentence_detail(clf, data)

sentence()

def passage():
	for data in test_p:
		predict_passage_by_correlation(clf, data)

def predict_passage_by_single(clf, data, pred_dic, out):
	global word_dic, tags
	sen_list = []
	sen_list = re.split(u"：|:|；|;|\)|\(|）|（|~|\?|？|！|!|，|。|,|\s", data)
	for sen in sen_list:
		sen = sen.strip()
		if sen == "" :
			continue
		test_x = sen2vec(sen, vec_type)
		res = clf.decision_function(test_x)[0] 
		label = "other"
		if np.max(res) >= -10:
			pred_y = (-res).argsort()[0]
			label = tags[pred_y]
		# print label, sen
		info = "\n"
		info += label + "\n"
		info += sen + "\n"
		out.write(info)
		pred_dic[sen] = label
	return

def predict_passage_by_correlation(clf, data, pred_dic, out):
	global word_dic, tags
	sen_list = []
	sen_list = re.split(u"：|:|；|;|\)|\(|）|（|~|\?|？|！|!|，|。|,|\s", data)
	last_data = {"data":[], "label":-1}	# sentence, label
	for sen in sen_list:
		sen = sen.strip()
		if sen == "":
			continue
		test_x = sen2vec(sen, vec_type)
		res = clf.decision_function(test_x)[0]
		# print "in"
		top_y = (-res).argsort()[:2]
		score_y = [res[top_y[0]], res[top_y[1]]]
		# print sen, score_y
		if score_y[0] < -10:
			# other comment
			if last_data["label"] == -1:
				last_data["data"].append(sen)
				# print "out"
			else:
				# emit the sentence before
				# if last_data["label"] != -1:
				# 	print last_data["data"], tags[last_data["label"]], "\n"
				# else:
				# 	print last_data["data"], "other", "\n"
				# print sen
				# print 
				info = "\n"
				info += tags[last_data["label"]] + "\n"
				info += " ".join(last_data["data"]) + "\n"
				out.write(info)
				# print info
				for s in last_data["data"]:
					label = "other"
					if last_data["label"] != -1:
						label = tags[last_data["label"]]
					pred_dic[s] = label
				last_data["data"] = [sen]
				last_data["label"] = -1
				# print "out"

		else:
			if last_data["label"] == top_y[0]:
				last_data["data"].append(sen)
				# print "out"

			else:
				if last_data["label"] == top_y[1] and score_y[0] - score_y[1] < 1:
					last_data["data"].append(sen)
					# print "out"

					# last_data["data"] += sen
					# print last_data["data"]
				else:
					# if last_data["label"] != -1:
					# 	print last_data["data"], tags[last_data["label"]], "\n"
					# else:
					# 	print last_data["data"], "other", "\n"
					# print "out"
					# print sen
					info = "\n"
					info += tags[last_data["label"]] + "\n"
					info += " ".join(last_data["data"]) + "\n"
					out.write(info)
					# print info
					for s in last_data["data"]:
						label = "other"
						if last_data["label"] != -1:
							label = tags[last_data["label"]]
						pred_dic[s] = label
					last_data["data"] = [sen]
					last_data["label"] = top_y[0]
	# if last_data["label"] != -1:
	# 	print last_data["data"], tags[last_data["label"]], "\n"
	# else:
	# 	print last_data["data"], "other", "\n"
	# print "out"
	info = "\n"
	info += tags[last_data["label"]] + "\n"
	info += " ".join(last_data["data"]) + "\n"
	out.write(info)
	# print info
	for sen in last_data["data"]:
		label = "other"
		if last_data["label"] != -1:
			label = tags[last_data["label"]]
		pred_dic[sen] = label
	return

def evaluate_split(load_file):
	f = open(load_file, "r")
	ps = ''.join(f.readlines()).decode("utf8")
	pl = re.split("\*{3} \d* \*{3}", ps)

	or_pl = []
	test_dic = {}
	int_sen = []
	ref_sen = {}
	i = 0
	for p in pl:
		p_l = p.split("\r")
		or_p = ""
		for line in p_l:
			sen = " ".join(line.split("\t")[:-1])
			cl = line.split("\t")[-1] # may be multi-label!
			or_p += sen
			int_sen.append(sen)
			seg_list = re.split(u"：|:|；|;|\)|\(|）|（|~|\?|？|！|!|，|。|,|\s", sen)
			for seg in seg_list:
				test_dic[seg] = cl
				ref_sen[seg] = i
			i += 1
		or_pl.append(or_p)
	test_dic.pop('')

	out = open("res/passage_error.out", "w")
	pred_dic = {}
	pred_out = open("res/predict_passage.out", "w")
	for p in or_pl:
		# predict_passage_by_single(clf, p, pred_dic, pred_out)
		predict_passage_by_correlation(clf, p, pred_dic, pred_out)
	pred_out.close()
	print len(pred_dic), len(test_dic)
	acc = 0
	# precision, recall, f1, support
	test_y = []
	pred_y = []
	tag_dic["other"] = 11
	for k in test_dic.keys():
		try:
			test_y.append(tag_dic[test_dic[k].split(",")[0]])
			pred_y.append(tag_dic[pred_dic[k]])
		except:
			continue
	score =  precision_recall_fscore_support(test_y, pred_y)
	print score[0]
	print score[1]
	print score[2]
	print score[3]
	for k in test_dic.keys():
		if pred_dic[k] in test_dic[k]:
			acc += 1
		else:
			info = "\n"
			info += "predict: " + pred_dic[k] + "\n"
			info += "true: " + test_dic[k] + "\n"
			info += k + "\n"
			info += "ref: " + int_sen[ref_sen[k]].strip() + "\n"
			out.write(info)
	acc /= float(len(test_dic))
	print acc
	out.close()

def evaluate_split_sentence(load_file):
	f = open(load_file, "r")
	exclude = ["configuration"]
	# exclude = []
	test_dic = {}
	int_sen = []
	ref_sen = {}

	for line in f.readlines():
		line = line.decode("utf8")
		# print line
		if re.match("\*{3} \d* \*{3}", line):
			continue

		sen = " ".join(line.split("\t")[:-1])
		cl = line.split("\t")[-1].strip() # may be multi-label!
		cl = cl.split("|")[0]
		flag = False
		for tag in exclude:
			if tag in cl:
				if len(cl.split(",")) == 1:
					flag = True
				else:
					cl2 = cl.split(",")
					if tag == cl2[0]:
						cl = cl2[1]
					else:
						cl = cl2[0]
		if flag or cl == "":
			continue
		test_dic[sen] = cl

	out = open("res/passage_error.out", "w")
	pred_dic = {}
	pred_out = open("res/predict_passage.out", "w")
	for sen in test_dic:
		pred_y = predict_sentence(clf, sen)
		pred_dic[sen] = pred_y
		test_y = test_dic[sen]
		# for multi-label, ensure test_y is a single class.
		if "," in test_y:
			if pred_y not in test_y:
				test_y = test_y.split(",")[0]
			else:
				test_y = pred_y
		test_dic[sen] = test_y
	print len(pred_dic), len(test_dic)
	acc = 0
	# precision, recall, f1, support
	test_y = []
	pred_y = []
	tag_dic["other"] = 11
	for k in test_dic:
		try:
			test_y.append(tag_dic[test_dic[k]])
			pred_y.append(tag_dic[pred_dic[k]])
		except:
			continue
	score =  precision_recall_fscore_support(test_y, pred_y)
	print score[0]
	print score[1]
	print score[2]
	print score[3]
	for k in test_dic.keys():
		pred_out.write(k + "\t" + pred_dic[k] + "\n")
		if pred_dic[k] == test_dic[k]:
			acc += 1
		else:
			info = "\n"
			info += "predict: " + pred_dic[k] + "\n"
			info += "true: " + test_dic[k] + "\n"
			info += k + "\n"
			out.write(info)
	print acc
	acc /= float(len(test_dic))
	print acc
	out.close()
	pred_out.close()

def evaluate_split_sentence_bench(load_file):
	f = open(load_file, "r")
	exclude = ["configuration", "other"]
	# exclude = []
	test_dic = {}
	pred_dic = {}
	int_sen = []
	ref_sen = {}

	for line in f.readlines():
		line = line.decode("utf8")
		# print line
		if re.match("\*{3} \d* \*{3}", line):
			continue

		sen = (" ".join(line.split("\t")[:-1])).strip()
		cl = line.split("\t")[-1].strip() # may be multi-label!
		cl_t = cl.split("|")[0]
		if len(cl.split("|")) == 1:
			cl_p = "neutral"
		else:
			cl_p = cl.split("|")[1]
		flag = False
		for tag in exclude:
			if tag in cl_t:
				if len(cl_t.split(",")) == 1:
					flag = True
				else:
					cl2 = cl_t.split(",")
					if tag == cl2[0]:
						cl_t = cl2[1]
					else:
						cl_t = cl2[0]
		if flag or cl_t == "":
			continue
		test_dic[sen] = cl_t
		pred_dic[sen] = cl_p

	for sen in test_dic:
		test_y = test_dic[sen]
		pred_y = pred_dic[sen]
		# for multi-label, ensure test_y is a single class.
		if "," in test_y:
			if pred_y not in test_y:
				test_y = test_y.split(",")[0]
			else:
				test_y = pred_y
		test_dic[sen] = test_y
	print len(pred_dic), len(test_dic)

	with open("passage_data.out", "w") as f:
		for k in test_dic:
			info = str(k) + "\t" + str(tag_dic[test_dic[k]]) + "\n"
			f.write(info)


	acc = 0
	# precision, recall, f1, support
	test_y = []
	pred_y = []
	tag_dic["other"] = 11
	for k in test_dic:
		try:
			pred_y.append(tag_dic[pred_dic[k]])
			test_y.append(tag_dic[test_dic[k]])
		except:
			continue

	score =  precision_recall_fscore_support(test_y, pred_y)
	print score[0]
	print score[1]
	print score[2]
	print score[3]
	acc = 0
	for k in test_dic.keys():
		if pred_dic[k] == test_dic[k]:
			acc += 1
	print acc
	acc /= float(len(test_dic))
	print acc

# evaluate_split_sentence("test_passage.py")
# evaluate_split_sentence_bench("test_passage.py")

def clear_passage_tag(load_file):
	f = open(load_file, "r")
	out = open("test_passage_ori.txt", "w")
	split_tags = ['space','power','operation','oilwear','comfort','appearance','decoration','costperformance', 'failure', 'maintenance','neutral', 'configuration', 'other']
	i = 0
	split_tag_dic = {}
	for tag in tags:
		split_tag_dic[tag] = i
		i += 1
	for r in f.readlines():
		ori_s = r.split(" ")[0]
		if re.match("\*{3} \d* \*{3}", r):
			ori_s = r
		out.write(ori_s + "\n")
	f.close()
	out.close()

def visual_passage(f, out):
	f = open(f, "r")
	out = open(out, "w")
		
	exclude = ["configuration"]
	# exclude = []

	pre = '''
	<head>
		<title>Visualization of passage classes</title>
		<link rel="stylesheet" type="text/css" href="passage.css">
	</head>
	<body>
	'''
	out.write(pre)
	span = ""
	# get true, pred tags.
	for line in f.readlines():
		line = line.decode("utf8")
		# print line
		if re.match("\*{3} \d* \*{3}", line):
			if span != "":
				p = '<p>%s</p>' % span
				# print p
				out.write(p)
				span = ""
				# break
			continue

		sen = " ".join(line.split("\t")[:-1])
		cl = line.split("\t")[-1].strip() # may be multi-label!
		cl_t = cl.split("|")[0]
		if len(cl.split("|")) == 1:
			cl_p = "neutral"
		else:
			cl_p = cl.split("|")[1]
		cl_p = predict_sentence(clf, sen)
		print cl_p
		flag = False
		for tag in exclude:
			if tag in cl_t:
				if len(cl_t.split(",")) == 1:
					flag = True
				else:
					cl2 = cl_t.split(",")
					if tag == cl2[0]:
						cl_t = cl2[1]
					else:
						cl_t = cl2[0]
		if flag or cl_t == "":
			continue
		tag = cl_p.split(",")[0]
		# print tag, sen
		span += '<span class="%s">%s</span>' % (tag, sen)

	out.write("</body>")
		

	f.close()
	out.close()

# visual_passage("test_passage.py", "res/visual/passage3.html")