#coding=utf8

#print ", ".join(seg_list)
import MySQLdb
import sys
import jieba
import os
import re
import xlrd
import xlwt

reload(sys)
sys.setdefaultencoding('utf8')

stopwordsfile=open('stopwords.txt','r')
stopwords=stopwordsfile.read().split()
print stopwords[3].decode('utf8')

seg_list = jieba.cut("他来到了网易杭研大厦".decode("utf-8"))
stat={'token':1}
T=[]
sql=''
output = xlwt.Workbook()
L=['advantage','shortcoming','space','power','operation','oilwear','comfort','appearance','decoration','costperformance','failure','maintenance','other']
tables=[]
for i in range(13):
	tables.append(output.add_sheet(L[i],cell_overwrite_ok=True))
tables[0].write(0,0,'test1')
tables[3].write(0,0,'test2')
count=0
result=[]
useless=['出来','还要','哎哟','能比','随','小三是','般','+','总之','尼玛','是','也','上','鬼',',','略','！','。','，','、','有','我','就','很','和','太','还是','在','会','的']
r=''
i=0
try:
	conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='15980ptpt',db='mysql',port=3306)
    	cur=conn.cursor()

	
	conn.set_character_set('utf8')

	cur.execute('SET NAMES utf8;')
	cur.execute('SET CHARACTER SET utf8;')
	cur.execute('SET character_set_connection=utf8;')
	
	for l in L:
		sql='select comment from info where label=%s;'
		count=cur.execute(sql,l)
		result=cur.fetchmany(count)
		stat={'token':1}
		for r in result:
			seg_list = jieba.cut(r[0].decode("utf-8"))
			for seg in seg_list:
				if seg in stat:
					stat[seg]+=1
				else:
					stat[seg]=1
			#tables[i].write(j,1,r[0].decode('utf8'))
		j=1
		for word in stat:
			if word in useless or word in stopwords or stat[word]<20:
				continue
			else:
				tables[i].write(j,1,word)
				tables[i].write(j,2,stat[word])				
				j=j+1
		print count
		i=i+1
	
	cur.close()
	conn.commit()
    	conn.close()
except MySQLdb.Error,e:
     	print "Mysql Error %d: %s" % (e.args[0], e.args[1])

output.save('fenci.xlsx')
