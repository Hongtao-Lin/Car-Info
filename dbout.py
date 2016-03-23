#coding=utf-8
import urllib,urllib2,cookielib
import requests
from bs4 import BeautifulSoup
import sys, os
import json
import copy, collections, datetime, MySQLdb


reload(sys)
sys.setdefaultencoding('utf8')

filename = 'outcome.txt'
output = open(filename,'w') 
try:
	conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='15980ptpt',db='mysql',port=3306)
	cur=conn.cursor()
	conn.set_character_set('utf8')
	cur.execute('SET NAMES utf8;')
	cur.execute('SET CHARACTER SET utf8;')
	cur.execute('SET character_set_connection=utf8;')
		
	sql='select * from comments;'		
	cur.execute(sql)
	for (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) in cur:
		output.write(a.encode('gb2312')+g.encode('gbk')+i.encode('gb2312')+k.encode('gb2312')+n.encode('utf8')+o.encode('utf8')+p.encode('utf8'))
	cur.close()
	conn.commit()
	conn.close()
except MySQLdb.Error,e:
     		print "Mysql Error %d: %s" % (e.args[0], e.args[1])
