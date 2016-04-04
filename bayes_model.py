#coding=utf8

import os, re, sys, MySQLdb
<<<<<<< HEAD
import xlrd, xlwt, pickle, json
=======
import xlrd, xlwt, pickle
>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
import jieba
import numpy as np


reload(sys)
sys.setdefaultencoding('utf8')

<<<<<<< HEAD
conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='1234',db='car',port=3306)
=======
conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='',db='car',port=3306)
>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
cur=conn.cursor()
conn.set_character_set('utf8')

cur.execute('SET NAMES utf8;')
cur.execute('SET CHARACTER SET utf8;')
cur.execute('SET character_set_connection=utf8;')
output = xlwt.Workbook()
tables = []
<<<<<<< HEAD
tags = ['space','power','operation','oilwear','comfort','appearance','decoration','costperformance', 'maintenance', 'failure']
limitation = 'label != "advantage" and label != "shortcoming" and label != "other"'
=======
tags = ['space','power','operation','oilwear','comfort','appearance','decoration','costperformance']
>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
tables.append(output.add_sheet("All", cell_overwrite_ok=True))
for i in range(len(tags)):
	tables.append(output.add_sheet(tags[i], cell_overwrite_ok=True))
stopwords = {}.fromkeys([ line.rstrip() for line in open('stopwords.txt') ])

# get all words into a vector
stat = {}
series = u"宝马3系"
<<<<<<< HEAD
sql = 'select comment from info where %s' % limitation
=======
sql = 'select info.comment from info where label != "advantage" and label != "shortcoming" and label != "maintenance" and label != "failure" and label != "other"'
>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
total_count = cur.execute(sql)
for r in cur.fetchall():
	seg_list = list(set(jieba.cut(r[0].decode('utf8'))))
	for seg in seg_list:
<<<<<<< HEAD
		# if seg in stopwords:
			# continue
=======
		if seg in stopwords:
			continue
>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
		if seg in stat:
			stat[seg] += 1
		else:
			stat[seg] = 1
j = 1
<<<<<<< HEAD
filter_freq = 5
=======
filter_freq = 3
>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
for k in stat.keys():
	# filter out those low-freq.
	if stat[k] <= filter_freq:
		stat.pop(k)

for word in stat:
	tables[0].write(j, 0, word)
	tables[0].write(j, 1, stat[word])
	j += 1

word_dic = {}
vec_len = len(stat.keys())
i = 0
for k in stat.keys():
	word_dic[k] = i
	i += 1

p = []
py = []
# p[i]<j> = p(x_j=1|y=i), py[i] = p(y=i)
for i in range(len(tags)):
	p.append(np.zeros([vec_len], dtype=float))
	py.append(0)
<<<<<<< HEAD
null_data = 0
i = 0

for tag in tags:
	local_stat = {}
	tmp_data = []
	sql='select comment from info where label="%s";' % tag
	print "tag is: ", tag
	count = cur.execute(sql)
	result = cur.fetchmany(count)
	py[i] = float(count + 1) / (total_count + len(tags))
	print "loading...", count
	for r in result:
		seg_list = list(set(jieba.cut(r[0].decode("utf-8"))))
		if seg_list == []:
			null_data += 1
			continue
		for seg in seg_list:
=======
i = 0
for tag in tags:
	local_stat = {}
	sql='select info.comment from info where label="%s";' % tag
	count = cur.execute(sql)
	result = cur.fetchmany(count)
	py[i] = float(count + 1) / (total_count + len(tags))

	print "loading...", count
	for r in result:
		seg_list = list(set(jieba.cut(r[0].decode("utf-8"))))
		for seg in seg_list:
			if seg in stopwords:
				continue
>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
			if seg in local_stat:
				local_stat[seg] += 1
			else:
				local_stat[seg] = 1
	for k in local_stat.keys(): # ("good", 2)
		if k not in word_dic:
			# print "not in dic: ", k.decode("utf8"), "count:", local_stat[k]
			continue
		p[i][word_dic[k]] = local_stat[k]
	p[i] += 1
<<<<<<< HEAD
	p[i] /= float(count + 2)
=======
	p[i] /= float(count + vec_len)
>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
	print py[i]
	print p[i]
	j = 1
	for word in local_stat:
		tables[i+1].write(j,0,word)
		tables[i+1].write(j,1,local_stat[word])				
		j += 1
	i += 1
<<<<<<< HEAD
# print json_data
print "null data is: %d\n" % null_data
=======

>>>>>>> bcc5bd2ec1929f4a740ab7eb1ff283a9ea642181
output.save('bayes.xls')

## dump data: word_dic, p, py
o = open('bayes.pkl', 'wb')
pickle.dump(tags, o)
pickle.dump(word_dic, o)
pickle.dump(py, o)
pickle.dump(p, o)
o.close()

cur.close()
conn.commit()
conn.close()