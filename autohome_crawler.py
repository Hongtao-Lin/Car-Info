#-*- coding: UTF-8 -*-
#!/usr/bin/python
#coding=utf-8
import urllib,urllib2,cookielib
import requests
from bs4 import BeautifulSoup
import sys, os
import json
import copy, collections, datetime, MySQLdb


reload(sys)
sys.setdefaultencoding('utf8')



#conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='15980ptpt',db='mysql',port=3306)
#cur=conn.cursor()

comment_field = ["brand", "series", "spec", "dateNtime", "url", "advantage", "shortcoming", "space", "power", "operation", "oilwear", "comfort", "appearance", "decoration", "costperformance", "hitch", "sustain", "other", "upvote", "downvote", "respond"]
split_tag = ["【最满意的一点】", "【最不满意的一点】", "【空间】","【动力】","【操控】","【油耗】","【舒适性】","【外观】","【内饰】","【性价比】","【故障】","【保养】","【其他描述】"]

tran_tag={'【最满意的一点】':'advantage','【最不满意的一点】':'shortcoming', '【空间】':'space','【动力】': 'power', '【操控】':'operation', '【油耗】':'oilwear', '【舒适性】':'comfort', '【外观】':'appearance','【内饰】' :'decoration', '【性价比】':'costperformance','【故障】': 'hitch', '【保养】':'sustain','【其他描述】': 'other'}
comment_dict = collections.OrderedDict()	
for field in comment_field:
	comment_dict[field] = ""


reload(sys)  
sys.setdefaultencoding('utf8')   


def store_comment(record, raw_comment, file):
	raw_date=''
	# processing raw_comment
	for tag in split_tag:
		if len(raw_comment.split(tag))!=1 :
			record[tran_tag[tag]]+=raw_comment.split(tag)[1].split('【')[0]
			#print record[tran_tag[tag]]
	T=[]
	i=0
	for (key, value) in record.items():
		file.write("%s: %s\n" % (key, value))
		if i<21:
			T.append(value)
			i+=1
		
	try:
		conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='15980ptpt',db='mysql',port=3306)
	    	cur=conn.cursor()

		conn.set_character_set('utf8')

		cur.execute('SET NAMES utf8;')
		cur.execute('SET CHARACTER SET utf8;')
		cur.execute('SET character_set_connection=utf8;')

		sql='use mysql;'
		cur.execute(sql)
		#T=['a','b','a','2000-11-22 09:06:22','b','a','b','a','b','a','b','a','b','a','b','a','b','a','b','a','b']
		sql='insert into comments values(%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s , %s);'
		#T=["ttt","2000-11-22 09:06:22"]
		#sql='insert into test values(%s,%s);'

		cur.execute(sql,T)
		
	    	cur.close()
		conn.commit()
	    	conn.close()
	except MySQLdb.Error,e:
     		print "Mysql Error %d: %s" % (e.args[0], e.args[1])

	return

def crawl_basic(url_basic, car_id):
	r = requests.get(url_basic)
	
	configList = r.text.split("config = ")[1].split(";")[0]
	configList = json.loads(configList)
	car_name = "".join(configList["result"]["paramtypeitems"][0]["paramitems"][0]["valueitems"][0]["value"].split(" "))
	file_name = car_name.encode("utf8") + "_basic" + ".txt"
	file = open(file_name, 'w')

	# extract basic infomation.
	basic_config = {}

	for item in configList["result"]["paramtypeitems"][0]["paramitems"]:
		
		basic_config[item["name"]] = item["valueitems"][0]["value"]
		file.write("%%" + item["name"] + ":" + basic_config[item["name"]] + "\n") 

	file.close()
	return car_name

def crawl_comment(web_name, brand, series, spec_name, url_base):


	log = open("crawler_log.txt", "a")
	file_name = series + ".txt"
	file = open(file_name, "a")

	flag = 1

	url_comment = url_base
	page_num = 1
	while(flag):
		r = requests.get(url_comment)
		soup = BeautifulSoup(r.text, "lxml")
		page_next = soup.body.find_all("a", class_="page-item-next")
		# No next page.
		if (page_next != [] and page_next[0].get("href") != "###"):
			page_num += 1
			url_comment = url_base + "/index_" + str(page_num) + ".html"
			# print url_comment
		else:
			flag = 0
			
		for item in soup.body.find_all("div", class_="mouthcon"):

			record = copy.copy(comment_dict)
			record["brand"] = brand
			record["series"] = series
			record["spec"] = spec_name
			record["url"] = web_name

			# get the specific comment page.
			try:
				r = requests.get(item.select(".mouth-main .mouth-item .cont-title .title-name a")[0]["href"])
			except Exception as e:
				print e
                                log.write("Error when crawling page: " + url_comment + "\ 	n Error msg: " + e + "\n\n")

			soup = BeautifulSoup(r.text, "lxml")
			# all comments including add-ons.
			
			comments = soup.select(".mouth-main .mouth-item .text-con")

			# list of date (including real comment, add-ons.)
			date = soup.select(".mouth-main .mouth-item .title-name b")
			record["dateNtime"] = date[-1]
			print record["dateNtime"]

			# write additional info
			record["upvote"] = upvote = item.select(".mouth-main .mouth-remak label.supportNumber")[0].text
			record["respond"] = respond = item.select(".mouth-main .mouth-remak span.CommentNumber")[0].text
			print "upvote is " + upvote
			print "respond is " + respond
			store_comment(record, comments[-1].text.strip(), file)
			
			for i in range(len(comments[0:-1])-1):
				add_on = comments[i]
				# can further get data on comment_time, average_oil_per_mile etc.
				if add_on.select("dd.add-dl-text") == []:
					continue
				record["date"] = date[i] 
				store_comment(record, add_on.select("dd.add-dl-text")[0].text.strip(), file)
			

	file.close()
	log.close()

def main():

	series = u"宝马3系"
	brand = u"宝马"
	if not os.path.exists("data/"):
		os.makedirs("data/")
	os.chdir("data/")

	# get all car lists.
	url_series = "http://car.autohome.com.cn/config/series/66.html"
	r = requests.get(url_series)
	car_list_str = r.text.split("specIDs =")[1].split(";")[0]
	car_list = [int(s.strip()) for s in car_list_str[1:-1].split(",")]
	

	for car_id in car_list:		
		print "Processing", car_id, "..."
		url_basic = "http://car.autohome.com.cn/config/spec/" + str(car_id) + ".html"
		url_comment = "http://k.autohome.com.cn/spec/" + str(car_id)

		car_name = crawl_basic(url_basic, car_id)
		crawl_comment("autohome", brand, series, car_name, url_comment)

	#cur.close()
	#conn.close()

main()

