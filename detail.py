#coding=utf8

import MySQLdb
import sys
import jieba
import os
import re
import xlrd
import xlwt

reload(sys)
sys.setdefaultencoding('utf8')
T=[]
sql=''
#output = xlwt.Workbook()
L=['advantage','shortcoming','space','power','operation','oilwear','comfort','appearance','decoration','costperformance','failure','maintenance','other']

# try:
conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='',db='car',port=3306)
cur=conn.cursor()


conn.set_character_set('utf8')

cur.execute('SET NAMES utf8;')
cur.execute('SET CHARACTER SET utf8;')
cur.execute('SET character_set_connection=utf8;')

print "loading..."
count=cur.execute('select * from comments limit 8000, 2000')
print 'there has %s rows record' % count
result=cur.fetchmany(count)
num = 0
prog = 0

for r in result:
	if num >= count*prog:
		print "processing %.2f%% ...\n" % prog*100
		prog += 0.01
	num += 1
	i=5
	for l in L:
		T=[]
		T.append(l)
		T.append('')
		T.append(r[1])
		T.append(r[2])
		T.append(r[4])
		T.append(r[i])
		T.append(r[3])
		web = "autohome"
		sql="insert into test_info values('%s','%s','%s','%s','%s','%s','%s', '%s', '%s');" % (l, '', '0', r[1], r[2], web, r[4], r[i], r[3])
		try:
			cur.execute(sql)
		except MySQLdb.Error,e:
			print num
			print sql
			print "Mysql Error %d: %s" % (e.args[0], e.args[1])
		i=i+1
	conn.commit()

cur.close()
conn.close()
