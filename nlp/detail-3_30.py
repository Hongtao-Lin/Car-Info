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
ADdescriptionBase=['动力','操控','空间','外观']
ADPo=['好','满意','不错','喜欢','高','精准','漂亮','时尚','好看','舒适','舒服','高速','够用']
ADNe=[]

SHdescriptionBase=['内饰','座椅','自动','音响','配置']
SHPo=['']
SHNe=['胎噪','小','低','不够','差','不好','太小','少','太少','很大']

SPdescriptionBase=['空间','后排','够用','后备箱','前排','座椅']
SPPo=['够用','很大','好','不错','舒适','充足']
SPNe=['小','挤']

POdescriptionBase=['动力','模式','运动','加速','超车','油门']
POPo=['够用','不错','好','舒适','随叫随到','强劲']
PONe=[]

OPdescriptionBase=['操控','转向','方向盘','指向','悬挂']
OPPo=['精准','不错','满意','清晰','灵敏','没得说']
OPNe=['小','硬']

OIdescriptionBase=['油耗','磨合期']
OIPo=['满意','省油','低']
OINe=['偏高','耗油','高']

COMdescriptionBase=['座椅','舒适性','后排','隔音']
COPo=['舒服','舒适','好','不错']
CONe=['硬','累','噪音','颠簸','差','不好']

APdescriptionBase=['外观','大灯','设计']
APPo=['喜欢','天使','满意','好看','漂亮','犀利','时尚','霸气','很漂亮','大气']
APNe=['小']

DEdescriptionBase=['内饰','做工','用料','设计']
DEPo=['好','喜欢','不错','豪华','满意','耐看']
DENe=['差','简单','一般般']

COdescriptionBase=['性价比','价格','配置','价位']
COpo=['高','不错','优惠','好','喜欢','满意','便宜']
CONe=['后悔']

FAdescriptionBase=[]
FAPo=[]
FANe=[]

MAdescriptionBase=[]
MAPo=[]
MANe=[]

OTdescriptionBase=[]
OTPo=[]
OTNe=[]

Base=[ADdescriptionBase,SHdescriptionBase,SPdescriptionBase,POdescriptionBase,OPdescriptionBase,OIdescriptionBase,COdescriptionBase,APdescriptionBase,DEdescriptionBase,COdescriptionBase,FAdescriptionBase,MAdescriptionBase,OTdescriptionBase]
Po=[ADPo,SHPo,SPPo,POPo,OPPo,OIPo,COPo,APPo,DEPo,COPo,FAPo,MAPo,OTPo]
Ne=[ADNe,SHNe,SPNe,PONe,OPNe,OINe,CONe,APNe,DENe,CONe,FANe,MANe,OTNe]

stat={'token':1}
S=[]
seg_list = jieba.cut("他来到了网易杭研大厦".decode("utf-8"))

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
		i=6
		j=0
		for l in L:
			T=[]
			ten=''
			shlb=''
			T.append(l)
			S=[]
			seg_list = jieba.cut(r[i])
			for w in seg_list:
				S.append(w)
			for base in Base[j]:
				for ne in Ne[j]:
					if ne in S and base in S:
						if S.index(base)-S.index(ne)<15 and S.index(base)-S.index(ne)>-15:
							ten='-1'
							shlb=base+ne
							break
					else:
						continue
				for po in Po[j]:
					if po in S and base in S:
						if ten!='' and S.index(base)-S.index(po)<15 and S.index(base)-S.index(po)>-15:
							ten='+1'
							shlb=base+po
							break
					else:
						continue
			
			T.append(shlb)			
			T.append(ten)			
			T.append(r[1])
			T.append(r[2])
			T.append(r[4])
			T.append(r[5])
			T.append(r[i])
			T.append(r[3])
			sql='insert into info values(%s,%s,%s,%s,%s,%s,%s,%s,%s);'
			cur.execute(sql,T)
			i=i+1
			j=j+1
	    		
	cur.close()
	conn.commit()
    	conn.close()
except MySQLdb.Error,e:
     	print "Mysql Error %d: %s" % (e.args[0], e.args[1])

