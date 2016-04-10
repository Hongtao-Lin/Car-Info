#coding=utf8

import os, re, sys, MySQLdb
import xlrd, xlwt, pickle
import jieba
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='1234',db='car',port=3306)

cur=conn.cursor()
conn.set_character_set('utf8')

pkl = open("bayes.pkl", 'rb')
tags = pickle.load(pkl)
word_dic = pickle.load(pkl)
py = pickle.load(pkl)
p = pickle.load(pkl)
limitation = 'label != "advantage" and label != "shortcoming" and label != "other"'

err_log = open("bayes_log.txt", "w")

# def get_class(data):
	
# 	# print "result is:", tags[res]
# 	return tags[res]

test_x = []
test_y = []
acc = 0
# tags = ['space','power','operation','oilwear','comfort','appearance','decoration','costperformance']

test_count = 0
num = 0
prog = 0
null_data = 0
# exclude some undefined comment, limit 10000.
sql = 'select comment, label from test_info where %s' % limitation
test_count = cur.execute(sql)
for (data, ans) in cur.fetchall():
	if num >= test_count*prog/100:
		print "progress: %d%%" % prog
		prog += 1
	num += 1

	seg_list = list(set(jieba.cut(data.decode("utf-8"))))
	if seg_list == []:
		null_data += 1
		continue
	vec = np.zeros([len(word_dic)], dtype=int)
	for seg in seg_list:
		if seg not in word_dic:
			# print "not in dic: ", seg.decode("utf8")
			continue
		vec[word_dic[seg]] = 1
	p_i = []
	for i in range(len(tags)):
		# p(x_j=?|y=i)
		# p_i.append(np.sum( np.log(abs(1 - vec - p[i])) ) + np.log(py[i]))
		p_i.append(np.sum( vec * np.log(p[i]) + (1-vec) * np.log(1-p[i]) ) + np.log(py[i]))

	# print "probs:", p_i
	res = np.argmax(np.array(p_i, dtype=float))
	res = tags[res]
	# res = get_class(data)
	# print res, ans
	if res == ans:
		acc += 1
	else:
		if data == "":
			null_data += 1
			continue
		err_log.write(res)
		err_log.write("\n")
		err_log.write(ans)
		err_log.write("\n")
		err_log.write(data.strip())
		err_log.write("\n")
		err_log.write(str(p_i))
		err_log.write("\n")
		err_log.write("\n")


print "correct_num: ", acc

print "null data: ", null_data
print "test_count: ", test_count
print "accuracy: ", acc / float(test_count - null_data)

# # get a test!
# test_x.append(u"外观:有亮点 内饰:做工细致 操控:驾驶舒适")

# for data in test_x:
# 	res = get_class(data)

cur.close()
conn.commit()
conn.close()
