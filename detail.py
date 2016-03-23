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
L=['advantage','shortcoming','space','power','operation','oilwear','comfort','appearance','decoration','costperformance','hitch','sustain','other']

try:
	conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='15980ptpt',db='mysql',port=3306)
    	cur=conn.cursor()

	
	conn.set_character_set('utf8')

	cur.execute('SET NAMES utf8;')
	cur.execute('SET CHARACTER SET utf8;')
	cur.execute('SET character_set_connection=utf8;')
	
	count=cur.execute('select * from comments')
   	print 'there has %s rows record' % count
	result=cur.fetchmany(count)
	sql='truncate table info;'
	cur.execute(sql)
    	for r in result:
		i=5
		j=0
		for l in L:
			T=[]
			T.append(l)
			T.append('')
			T.append(r[1])
			T.append(r[2])
			T.append(r[4])
			T.append(r[i])
			T.append(r[3])
			sql='insert into info values(%s,%s,%s,%s,%s,%s,%s);'
			cur.execute(sql,T)
			i=i+1
	    		
	cur.close()
	conn.commit()
    	conn.close()
except MySQLdb.Error,e:
     	print "Mysql Error %d: %s" % (e.args[0], e.args[1])

