#coding=utf8

#print ", ".join(seg_list)
import MySQLdb
import sys
import jieba
import os
import re
import xlrd
import xlwt
import jieba.posseg as pseg

reload(sys)
sys.setdefaultencoding('utf8')

stopwordsfile=open('stopwords.txt','r')
stopwords=stopwordsfile.read().split()
print stopwords[3].decode('utf8')
jieba.load_userdict('userdict.txt') 
seg_list = jieba.cut("他来到了网易杭研大厦".decode("utf-8"))
T=[]
S=[]
S1=[]
S2=[]
S3=[]
S4=[]
sql=''
output = xlwt.Workbook()
tables=[]
tables.append(output.add_sheet("test",cell_overwrite_ok=True))
tables[0].write(0,0,'test1')
count=0
row=1
volumn=1
i=0
try:
	conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='15980ptpt',db='mysql',port=3306)
    	cur=conn.cursor()

	
	conn.set_character_set('utf8')

	cur.execute('SET NAMES utf8;')
	cur.execute('SET CHARACTER SET utf8;')
	cur.execute('SET character_set_connection=utf8;')
	

	sql='select advantage,shortcoming,other from comments where other!="";'
	count=cur.execute(sql)
	result=cur.fetchmany(100)
	print 'OK',count

	for r in result:
		S+=r[0].split('。')
		S+=r[1].split('。')
		S+=r[2].split('。')
	for s in S:
		S1+=s.split('，')
	for s in S1:
		S2+=s.split('？')
	for s in S2:
		S3+=s.split('；')
	for s in S3:
		S4+=s.split('！')
	row=1
	volumn=6
	for s in S4:
		volumn=6
		if s!='':
			tables[0].write(row,volumn,s.decode('utf8'))
			seg_list = pseg.cut(s.decode("utf-8"))
			volumn=7
			for w in seg_list:
				tables[0].write(row,volumn,w.word)
				volumn+=1
				tables[0].write(row,volumn,w.flag)
				volumn+=1
			row+=1
				
	print len(S3)	
	cur.close()
	conn.commit()
    	conn.close()
except MySQLdb.Error,e:
     	print "Mysql Error %d: %s" % (e.args[0], e.args[1])

output.save('zhutijixing.xls')
