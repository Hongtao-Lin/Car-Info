#coding=utf8
import MySQLdb
import sys
reload(sys)
sys.setdefaultencoding('utf8')
 
try:
	conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='15980ptpt',db='mysql',port=3306)
    	cur=conn.cursor()

	sql = "drop table if exists comments;"
	cur.execute(sql)
	conn.set_character_set('utf8')

	cur.execute('SET NAMES utf8;')
	cur.execute('SET CHARACTER SET utf8;')
	cur.execute('SET character_set_connection=utf8;')

	sql = "create table if not exists comments (brand text, series text, spec text, dateNtime text, url text, advantage text, shortcoming text, space text , power text , operation text, oilwear text , comfort text , appearance text , decoration text, costperformance text, hitch text, sustain text, other text , upvote text, downvote text, respond text)DEFAULT CHARSET=utf8;"
	
	cur.execute(sql)
  	sql = "create table if not exists com2 (brand text, series text, spec text default null, date datetime)"
	cur.execute(sql)
	sql = "insert into comments values('testbrand','testseries','testspec','2016-01-01 21:12:12','www.baidu.com','good','bad','large','great','easy','high','terrible','ugly','simple','unworth','no','hard','no',11,12,'hehe')"
	cur.execute(sql)
	s='测试品牌'.decode('utf8')
	sql = "insert into comments values(%s,'testseries','testspec','2016-01-01 21:12:12','www.baidu.com','good','bad','large','great','easy','high','terrible','ugly','simple','unworth','no','hard','no',11,12,'hehe')"
	cur.execute(sql,s)

	sql = "create table if not exists info (label text, tendency text, series text,spec text,url text, comment text,date text)DEFAULT CHARSET=utf8;"
	cur.execute(sql)

    	cur.close()
	conn.commit()
    	conn.close()
except MySQLdb.Error,e:
     	print "Mysql Error %d: %s" % (e.args[0], e.args[1])
