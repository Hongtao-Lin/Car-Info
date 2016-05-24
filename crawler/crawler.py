#!/usr/bin/env python
# -*- coding: utf8 -*-

import urllib,urllib2,cookielib
import requests
from bs4 import BeautifulSoup
import sys, os, time, signal, re
import json
import copy, collections, datetime, MySQLdb
import traceback

reload(sys) 
sys.setdefaultencoding('utf8')   

comment_field = ["brand", "series", "spec", "date", "web", "url", "advantage", "shortcoming", "space", "power", "operation", "oilwear", "comfort", "appearance", "decoration", "costperformance", "failure", "maintenance", "other", "upvote", "downvote", "respond"]
split_tag = ["【最满意的一点】", "【最不满意的一点】", '【最满意】', '【优点】','【缺点】']
split_tag += ["【空间】","【动力】","【操控】","【油耗】","【舒适性】","【外观】","【内饰】","【性价比】"]
split_tag += ["【故障】","【保养】","【其他描述】",'【其他】','【售前售后】','【综述】','【配置】']
split_tag += ['【总评】','【吐槽】', '【品质】', '【细节】', '【加速】', '【噪音】']

tran_tag = {'【最满意的一点】':'advantage','【最满意】':'advantage','【优点】':'advantage'}
tran_tag.update({'【缺点】':'shortcoming','【最不满意的一点】':'shortcoming','【最不满意】':'shortcoming'})
tran_tag.update({'【空间】':'space','【动力】': 'power', '【操控】':'operation', '【油耗】':'oilwear'})
tran_tag.update({'【舒适性】':'comfort', '【舒适度】':'comfort', '【舒适】':'comfort','【外观】':'appearance'})
tran_tag.update({'【内饰】' :'decoration', '【性价比】':'costperformance','【故障】': 'failure'})
tran_tag.update({'【保养】':'maintenance','【其他描述】': 'other','【其他】': 'other','【售前售后】': 'other','【综述】':'other', '【总评】':'other','【吐槽】':'other'})
tran_tag.update({'【配置】':'configuration', '【品质】':'costperformance', '【细节】':'other', '【加速】':'power'})
tran_tag.update({'【噪音】':'comfort'})
comment_dict = collections.OrderedDict()    
for field in comment_field:
	comment_dict[field] = ""

# get the pool from sql further.
url_pool = [
	# {"web":"sohu", "firm":u"华晨", "brand":u"宝马", "series":u"宝马3系", "url":"http://db.auto.sohu.com/huachenbmw/1232/dianping_1.html", "last_visit":time.clock(), "last_content":""},
	# {"web":"pcauto", "firm":u"华晨", "brand":u"宝马", "series":u"宝马3系", "url":"http://price.pcauto.com.cn/comment/sg424/t1/p1.html", "last_visit":time.clock(), "last_content":""},
	# {"web":"yiche", "firm":u"华晨", "brand":u"宝马", "series":u"宝马3系", "url":"http://car.bitauto.com/baoma3xi/koubei/tags/", "last_visit":time.clock(), "last_content":""},
	# {"web":"autohome", "firm":u"华晨", "brand":u"宝马", "series":u"宝马3系", "url":"http://k.autohome.com.cn/66/ge0/0-0-2/", "last_visit":time.clock(), "last_content":""}
	# {"web":"xcar", "firm":u"华晨", "brand":u"宝马", "series":u"宝马3系", "url":"http://newcar.xcar.com.cn/1472/review/0/0_2.htm", "last_visit":time.clock(), "last_content":""}
	# {"web":"netease", "firm":u"奔驰", "brand":u"北京奔驰", "series":u"奔驰GLC", "url":"http://product.auto.163.com/opinion_more/1990/1_1.html", "last_visit":time.clock(), "last_content":""}
	# {"web":"xgo", "firm":u"华晨", "brand":u"宝马", "series":u"宝马3系", "url":"http://www.xgo.com.cn/2710/list_s1_p1.html", "last_visit":time.clock(), "last_content":""}
]

max_comment_num = 1000000

def get_car_series():
	global url_pool

	# #autohome begin
	# web = "autohome"
	# url_basic = "http://www.autohome.com.cn/grade/carhtml/"
	# cnt = 0
	# for i in range(ord("A"), ord("Z")+1):
	# 	url_series = url_basic + chr(i) + ".html"
	# 	r = requests.get(url_series)
	# 	soup = BeautifulSoup(r.text, "lxml")
	# 	for item in soup.find_all("dl"):
	# 		firm = item.dt.div.text
	# 		for brand_item in item.select("div.h3-tit"):
	# 			brand = brand_item.text
	# 			for li in brand_item.next_sibling.next_sibling.select("li"):
	# 				if li.select("span.text-through") != []:
	# 					continue
	# 				if li.select("a") == []:
	# 					continue
	# 				series = li.h4.text
	# 				url = li.select("div")[-1].select("a")[-1]["href"].split("#")[0] + "ge0/0-0-2"
	# 				print "series: ", series, "url: ", url
	# 				cnt += 1
	# 				url_pool.append({"web":web, "firm":firm, "brand":brand, "series":series, "url":url, "last_visit":time.clock(), "last_content":""})
	# print "###########"
	# print "autohome cnt: ", cnt
	# print "###########"

	# #yiche begin
	# web = "yiche"
	# cnt = 0
	# url_basic = "http://api.car.bitauto.com/CarInfo/MasterBrandToSerialNew.aspx?type=2&pid=0&rt=master&serias=m&key=master"
	# r = requests.get(url_basic)
	# brand_list = json.loads(r.text.split('["master"]=')[-1])
	# for item in brand_list.values():
	#   firm = item["name"]
	#   url_series = "http://api.car.bitauto.com/CarInfo/MasterBrandToSerialNew.aspx?type=2&pid=%d&rt=serial&serias=m&key=serial&include=1" % int(item["id"])
	#   r = requests.get(url_series)
	#   series_list = json.loads(r.text.split('["serial"]=')[-1])
	#   for series_item in series_list.values():
	# 	  brand = series_item["goname"]
	# 	  series = series_item["showName"]
	# 	  url = "http://car.bitauto.com/%s/koubei/tags/综合/" % series_item["urlSpell"]
	# 	  print "series: ", series, "url: ", url
	# 	  url_pool.append({"web":web, "firm":firm, "brand":brand, "series":series, "url":url, "last_visit":time.clock(), "last_content":""})
	# 	  cnt += 1
	# print "###########"
	# print "yiche cnt: ", cnt
	# print "###########"

	# #pcauto begin
	# web = "pcauto"
	# cnt = 0
	# url_basic = "http://www.pcauto.com.cn/"
	# r = requests.get(url_basic)
	# soup = BeautifulSoup(r.text, "lxml")
	# for brand_item in soup.select("#brand_3_list dl dd"):
	# 	firm = brand_item.a.span.text.strip()
	# 	url_series = "http://price.pcauto.com.cn/interface/5_3/serial_json_chooser.jsp?brand=%d&callback=callback" % int(brand_item.attrs["data-value"])
	# 	r = requests.get(url_series)
	# 	series_list = json.loads(r.text.split("callback(")[-1][:-2]).values()[0]
	# 	for series_item in series_list:
	# 		if series_item["id"][0] == "+":
	# 			brand = series_item["text"]
	# 			continue
	# 		series = series_item["text"]
	# 		url = "http://price.pcauto.com.cn/comment/sg%d/t1/p1.html" % int(series_item["id"])
	# 		print "series: ", series, "url: ", url
	# 		url_pool.append({"web":web, "firm":firm, "brand":brand, "series":series, "url":url, "last_visit":time.clock(), "last_content":""})
	# 		cnt += 1
	# print "###########"
	# print "pcauto cnt: ", cnt
	# print "###########"

	# #xgo begin
	# web = "xgo"
	# cnt = 0
	# url_basic = "http://www.xgo.com.cn/brand.html"
	# r = requests.get(url_basic)
	# soup = BeautifulSoup(r.text, "lxml")
	# for firm_item in soup.select("div.main_nr"):
	# 	firm = firm_item.select("div.l a")[-1].text
	# 	for brand_item in firm_item.select("div.r div.car"):
	# 		brand = brand_item.text or firm
	# 		for series_item in brand_item.next_sibling.select("li"):
	# 			series = series_item.dl.dt.text
	# 			url = series_item.dl.dt.a["href"] + "list_s1_p1.html"
	# 			print "series: ", series, "url: ", url
	# 			url_pool.append({"web":web, "firm":firm, "brand":brand, "series":series, "url":url, "last_visit":time.clock(), "last_content":""})
	# 			cnt += 1         
	# print "###########"
	# print "xgo cnt: ", cnt
	# print "###########"

	# #sohu begin
	# web = "sohu"
	# cnt = 0
	# url_basic = "http://db.auto.sohu.com/index.shtml"
	# r = requests.get(url_basic)
	# soup = BeautifulSoup(r.text, "lxml")
	# for firm_item in soup.select("div.category_main"):
	#   firm = firm_item.select("p.car_brand")[0].text
	#   for brand_item in firm_item.select("div.meta_con"):
	# 	  brand = brand_item.div.a.text
	# 	  for series_item in brand_item.select("ul li"):
	# 		  series = series_item.select("a.name")[0].text
	# 		  # if no comments on this serial.
	# 		  if series_item.select("a.del") != []:
	# 			  continue
	# 		  url = series_item.select("dd")[1].select("a")[1]["href"][:-5] + "_1.html"
	# 		  print "series: ", series, "url: ", url
	# 		  url_pool.append({"web":web, "firm":firm, "brand":brand, "series":series, "url":url, "last_visit":time.clock(), "last_content":""})
	# 		  cnt += 1
	# print "###########"
	# print "sohu cnt: ", cnt
	# print "###########"
	
	#xcar begin
	web = "xcar"
	cnt = 0
	# note: firm -> brand -> series!
	url_basic = "http://newcar.xcar.com.cn/price"
	r = requests.get(url_basic)
	soup = BeautifulSoup(r.text, "lxml")
	for firm_item in soup.select("div.container > table > tbody > tr"):
	  firm = firm_item.select("div.column_tit")[0].text.strip()
	  print firm.encode("utf8")
	  for brand_item in firm_item.select("div.column_content"):
		  brand = brand_item.p.text.strip()
		  for series_item in brand_item.select(".item_list"):
			  series = series_item.a["title"].strip()
			  url = "http://newcar.xcar.com.cn" + series_item.a["href"] + "review/0.htm" 
			  print "series: ", series, "url: ", url
			  url_pool.append({"web":web, "firm":firm, "brand":brand, "series":series, "url":url, "last_visit":time.clock(), "last_content":""})
			  cnt += 1
	print "###########"
	print "xcar cnt: ", cnt
	print "###########"
	return url_pool

def store_comment(web_name, record, raw_comment):
	# processing raw_comment
	if web_name == "autohome":
		for tag in split_tag:
			if len(raw_comment.split(tag))>1 :
				record[tran_tag[tag]] = raw_comment.split(tag)[1].split('【')[0].strip()
	elif web_name == "xgo":
		for tag in split_tag:
			if len(raw_comment.split(tag))>1 :
				record[tran_tag[tag]] = raw_comment.split(tag)[1].split('\n')[0].strip()
	elif web_name == "pcauto":
		for tag in split_tag:
			if len(raw_comment.split(tag))>1 :
				record[tran_tag[tag]] = raw_comment.split(tag)[1].split('【')[0].strip()
	elif web_name == "yiche":
		for tag in split_tag:
			if len(raw_comment.split(tag))>1 :
				record[tran_tag[tag]] = raw_comment.split(tag)[1].split('【')[0].strip()
	elif web_name == "sohu":
		for tag in split_tag:
			if len(raw_comment.split(tag))>1 :
				record[tran_tag[tag]] = raw_comment.split(tag)[1].split('【')[0].strip()
	elif web_name == "xcar":
		record = record
	else:
		# just a temporary test!
		record["other"] = raw_comment
	# store it in sql.

	try:
		conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='1234',db='car',port=3306)
		cur=conn.cursor()
		conn.set_character_set('utf8')

		cur.execute('SET NAMES utf8;')
		cur.execute('SET CHARACTER SET utf8;')
		cur.execute('SET character_set_connection=utf8;')

		if web_name != "xcar":
			sql='insert into comments values(%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s ,%s , %s , %s , %s , %s, %s);'
			cur.execute(sql,list(record.values()))
		else:
			sql='insert into info_emo(label, showlabel, tendency, series, spec, web, url, comment, date) values(%s , %s , %s , %s ,%s , %s , %s , %s ,%s);'
			cur.execute(sql,list(record.values()))

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

def autohome_crawler(brand, series, url_base, last_visit, last_content, max_num):
	web_name = "autohome"
	cur_content = ""

	flag = 1
	# is_new = 1
	cnt = 0
	page_num = 1
	url_comment = url_base
	while(flag and max_num > 0):
		try:
			print "url:", url_comment
			r = requests.get(url_comment)
			soup = BeautifulSoup(r.text)
			for item in soup.body.find_all("div", class_="mouthcon"):
				# if not is_new:
				#   break
				record = copy.copy(comment_dict)
				record["brand"] = brand
				record["series"] = series
				record["spec"] = "".join(item.select("div.mouthcon-cont-left dl.choose-dl dd")[0].text.split())
				record["web"] = web_name
				record["url"] = item.select(".mouth-main .mouth-item .cont-title .title-name a")[0]["href"]
				# get the specific comment page.
				main_comment = item.select(".mouth-main .mouth-item .text-con")[0].text.strip()

				# write additional info
				
				record["upvote"] = upvote = item.select(".mouth-main .mouth-remak label.supportNumber")[0].text
				record["respond"] = respond = item.select(".mouth-main .mouth-remak span.CommentNumber")[0].text
				comment_date = item.select(".mouth-main .mouth-item .title-name b")[-1].text.strip()
				print "date: ", comment_date

				score = []
				for sc in item.select(".score-small"):
					score.append(int(sc.next_sibling.text))

				# if there are no add-ons, do not need to follow on.
				# No! Sometimes the main content is too long to exhibit them all.
				if item.select("dl.add-dl") != [] and item.select("div.con-mask") != []:
					r = requests.get(record["url"])
					print "add-on: ", record["url"]
					soup2 = BeautifulSoup(r.text, "lxml")
					more_comments = soup2.select(".mouth-main .mouth-item")
					main_comment = more_comments[-1].select("div.text-con")[0].text.strip()
					for item2 in more_comments[:-1]:
						if item2.select("dd.add-dl-text") == []:
							continue
						add_on_comment = item2.select("dd.add-dl-text")[0].text.strip()
						# print add_on_comment
						record["date"] = item2.select("div.title-name b")[0].text.strip() 
						cur_content = hash(add_on_comment)
						if cur_content == last_content:
							is_new = 0
							break
						store_comment(web_name, record, add_on_comment)

				record["date"] = comment_date
				cur_content = hash(main_comment)
				if cur_content == last_content:
					is_new = 0
					break
				cnt += 1
				store_comment(web_name, record, main_comment)
				max_num -= 1
				if max_num <= 0:
					break
			
			page_next = soup.body.find_all("a", class_="page-item-next")
			# Get next page.
			if (page_next != [] and page_next[0].get("href") != "###"):
				page_num += 1
				url_comment = url_base + "/index_" + str(page_num) + ".html"
			else:
				flag = 0

		except Exception as e:
			print e
			print "Error when crawling page: " + str(url_comment) + "\n"
			traceback.print_exc(file=sys.stdout)

	if cnt == 0:
		print "No comment in url: ", url_base
		print max_num, flag

def yiche_crawler(brand, series, url_base, last_visit, last_content, max_num):

	web_name = "yiche"
	flag = 1
	page_num = 1
	url_comments = url_base

	while(flag and max_num > 0):
		try:
			r = requests.get(url_comments)
			soup = BeautifulSoup(r.text, "lxml")
			# Find comments from a specific user.
			for item in soup.body.find_all("div", id="topiclistshow")[0].find_all("dl"):
				spec_name = series + "".join(item.select("p.carname")[0].text.split())
				url_comment_list = item.find_all("ul", class_="cont_list")[0].find_all("li")

				# One user may have several comments.
				for url_comment in url_comment_list:
					record = copy.copy(comment_dict)
					record["brand"] = brand
					record["series"] = series
					record["spec"] = spec_name
					record["web"] = web_name
					record["url"] = url_comment.a.get("href")
					r = requests.get(record["url"], timeout = 10)
					print url_comment.a.get("href")
					soup2 = BeautifulSoup(r.text, "lxml")
					# filter some special comments.
					if soup2.select("span.fapiao_tab") != []:
						continue

					comment = soup2.select("div#content_bit div.article-contents")[0]
					
					record["respond"] = respond = url_comment.select("div.rbox")[0].a.span.text[3:-1].strip()
					record["upvote"] = upvote = url_comment.select("div.rbox em")[-1].text[1:-1].strip()
					if soup2.select("#time") != []:
						record["date"] = date = soup2.select("#time")[0].text.strip()
					elif soup2.select("div.the_pages_tags_r") != []:
						record["date"] = date = soup2.select("div.the_pages_tags_r")[0].span.text.strip()
					else:
						print "Error: Date not available"
						continue
					print "date is: %s" % date 

					if comment.select("p.czjg_xq_cont") != []: # which means that comment is not valid.
						continue
					# pre-process the text for convenience.
					for tag in comment.find_all("strong"):
						if not tag.string:
							continue
						tag.string = u"【" + tag.string.strip() + u"】"
						# .text
					for p in comment.find_all("p"):
						if not p.string:
							continue
						p.string = p.string.strip()
					# Here we clean some data.
					if comment.h4:
						comment.h4.extract()
					if comment.pre:
						comment.pre.extract()
					if comment.find("div.con_nav2"):
						comment.find("duv.con_nav2").extract()
					store_comment(web_name, record, comment.text.strip())
					max_num -= 1
					if max_num <= 0:
						break;
			page_all = soup.body.select("#main_pager_down")
			# print page_all
			if (page_all != [] and page_all[0].find_all("a", class_="next_on") != []):
				page_num += 1
				url_comments = url_base + "page" + str(page_num)
				print "another_page, ", url_comments
			else:
				flag = 0

		except Exception as e:
			print url_comments
			print e
			traceback.print_exc(file=sys.stdout)

def pcauto_crawler(brand, series, url_base, last_visit, last_content, max_num):
	web_name = "pcauto"

	flag = 1
	page_num = 1
	url_comments = url_base
	while(flag and max_num > 0):
		try:
			r = requests.get(url_comments)
			soup = BeautifulSoup(r.text, "lxml")
				
			for item in soup.select("div.main_table"):
				record = copy.copy(comment_dict)
				record["brand"] = brand
				record["series"] = series
				record["spec"] = "".join(item.select("div.car span.td2")[0].text.split())
				print "spec: ", record["spec"]
				record["web"] = web_name
				record["url"] = item.select("div.info > p a")[0]["href"]
				print "url: ", record["url"]

				comment = item.select("div.table_text")[0]
				
				record["date"] = date = item.select("div.info p a")[0].text.strip()[:-2]
				record["upvote"] = upvote = item.select("a.good em")[0].text.strip()[1:-1]
				print "date is: %s" % date
				
				for tag in comment.find_all("strong"):
					tag.string = u"【" + tag.string[0:-1] + u"】"
				store_comment(web_name, record, comment.text.strip())
				
				for add_on in item.select("div.zjdp"):
					# note that this date is additional on original one!! e.g.: 2014-10-23 + 56 = 2014-11-23 + 26 (day!)
					add_date = item.select("span.sp2")[0].text.strip()[6:-3]
					# need to convert it!
					# record["date"] = ...
					print "additional_date: %s" % add_date
					store_comment(web_name, record, add_on.select("div.zjdp_text")[0].text.strip())
				
				max_num -= 1
				if max_num <= 0:
					break

			has_next_page = soup.select("#pcauto_page a.next") != []
			if has_next_page:
				page_num += 1
				url_comments = url_base.split("/p1")[0] + "/p" + str(page_num) + ".html"
				print "another_page, ", url_comments
			else:
				flag = 0

		except Exception as e:
			print e
			print "Error when crawling page: " + url_comments + "\n\n"
			traceback.print_exc(file=sys.stdout)

def xgo_crawler(brand, series, url_base, last_visit, last_content, max_num):
	web_name = "xgo"
	flag = 1
	page_num = 1
	url_comments = url_base

	spec_name = ""
	while(flag and max_num > 0):
		try:
			r = requests.get(url_comments)
			soup = BeautifulSoup(r.text, "lxml")
			
			# Find comments from a specific user.
			for item in soup.body.select("div.xgo_cars_dianping dl.info"):

				record = copy.copy(comment_dict)
				record["brand"] = brand
				record["series"] = series
				record["spec"] = spec_name
				record["web"] = web_name
				
				comment_div = item.select("dd.paragraph div.clearfix")
				if item.select("dd.title span.r") != []:
					record["date"] = item.select("dd.title span.r")[0].text[4:].encode("utf-8")
				print "date is: ", record["date"]
				content = ""
				for comment_item in comment_div:
					content += u"【" + comment_item.div.text[:-1] + u"】"
					content += comment_item.select("div.pingyu")[0].text.strip()
					content += "\n"
				
				info_list = tuple(i.text.encode("utf-8") for i in item.select("div.apply span.redc00"))

				record["url"] = "http://www.xgo.com.cn" + item.select("div.apply > a")[0]["href"]
				print "url is: ", record["url"]
				record["respond"] = info_list[0]
				record["upvote"] = info_list[1]
				record["downvote"] = info_list[2]

				store_comment(web_name, record, content.encode("utf-8"))
				max_num -= 1
				if max_num <= 0:
					break

			page_all = soup.body.find_all("div", class_="xgo_cars_page")
			if (page_all != [] and page_all[0].find_all("a", class_="next") != []):
				page_num += 1
				url_comments = url_base[:-6] + str(page_num) + ".html"
				print "another_page, ", url_comments
			else:
				flag = 0

		except Exception as e:
			print url_comments
			print e
			traceback.print_exc(file=sys.stdout)

def sohu_crawler(brand, series, url_base, last_visit, last_content, max_num):
	web_name = "sohu"
	flag = 1
	page_num = 1
	url_comments = url_base

	while(flag and max_num > 0):
		try:
			r = requests.get(url_comments)
			soup = BeautifulSoup(r.text, "lxml")
		
			# Find comments from a specific user.
			for item in soup.body.select("ul.pllist > li"):
				#print item
				spec_name = series + "".join(item.select("div.pltit")[0].h3.a.text.split())
				#print item.select("span.time")[0].text
				date = item.select("span.time")[0].text.split(u"发表于")[-1].strip()
				print "date is: %s" % date
				record = copy.copy(comment_dict)
				record["brand"] = brand
				record["series"] = series
				record["spec"] = spec_name
				record["web"] = web_name
				record["date"] = date
				record["url"] = url_comments + "#" + str(item.div["id"])
				print "url is: ", record["url"]

				content = ""
				comment_div = item.select("div.pltxt")[0]
				for tag in comment_div.select("p"):
					pair = tag.text.split()
					content += u"【" + pair[0] + u"】"
					content += pair[1] + "\n"
				store_comment(web_name, record, content.strip())
				max_num -= 1
				if max_num <= 0:
					break


			next_page = soup.body.select("div.pagelist_new li.bg_white")[0].previous_sibling.previous_sibling
			if next_page.attrs == {}: # no class named "unable"
				page_num += 1
				url_comments = url_base.split("dianping")[0] + "dianping_1_" + str(page_num) + ".html"
				print "another_page, ", url_comments
			else:
				flag = 0

		except Exception as e:
			print e
			print url_comments
			traceback.print_exc(file=sys.stdout)

def netease_crawler(brand, series, url_base, last_visit, last_content, max_num):
	web_name = "netease"
	spec_name = ""
	flag = 1
	page_num = 1
	url_comments = url_base
	last_content = ""

	while(flag and max_num > 0):
		try:
			r = requests.get(url_comments)
			soup = BeautifulSoup(r.text, "lxml")
			print url_comments
			
			# Find comments from a specific user.
			for item in soup.body.select("div.commentList-main > div.commentSingle"):
					# print url_comments
				date = item.select("span.postTime")[0].text[:-3]
				print "date is: %s" % date
				record = copy.copy(comment_dict)
				record["brand"] = brand
				record["series"] = series
				record["spec"] = spec_name
				record["web"] = web_name
				record["date"] = date
				content = ""
				comment_div = item.select("div.comBody")[0]
				record["url"] =  "http://product.auto.163.com" + comment_div.a["href"]
				print "url is: ", record["url"]

				r = requests.get(record["url"])
				soup2 = BeautifulSoup(r.text, "lxml")
				content = soup2.select("div.d3")[0].text.strip()
				# print "comment is: ", content

				record["respond"] = respond = item.select("li.reply")[0].text[3:-1]
				record["upvote"] = upvote = item.select("li.useful")[0].text[3:-1]
				record["downvote"] = downvote = item.select("li.unuseful")[0].text[3:-1]
				print "respond is %s" % respond
				print "upvote is %s" % upvote
				print "downvote is %s" % downvote

				if last_content != content.strip():
					last_content = content.strip()
					store_comment(web_name, record, content.strip())
					max_num -= 1
					if max_num <= 0:
						break
			next_page = soup.body.select("div.commentList-main div.comment-pages .active")[0].next_sibling.next_sibling
			if next_page.get("class") == None:
				page_num += 1
				url_comments = url_base.split("/1_")[0] + "/" + str(page_num) + "_1.html"
				print "another_page, ", url_comments
			else:
				flag = 0
		except Exception as e:
			print url_comments
			print e
			traceback.print_exc(file=sys.stdout)

def xcar_crawler(brand, series, url_base, last_visit, last_content, max_num):
	web_name = "xcar"
	flag = 1
	page_num = 1
	url_comments = url_base

	# Find comments from a specific user.
	last_url = ''
	last_spec = ''
	last_comment_all = ''
	last_url = ''
	last_date = ''
	last_valid_comments = []
	last_valid_labels = []
	last_valid_tendency = []
	tendency_list = {u"好评":1, u"中评":0, u"差评":-1}
	is_404 = False
	while(flag and max_num > 0):
		try:
			r = requests.get(url_comments)
			soup = BeautifulSoup(r.text, "lxml")
		
			for item in soup.body.select("div.review_comments_dl > dl"):
				record = collections.OrderedDict()
				cur_url = item.select("dt > a")[0]["href"]
				if (cur_url == last_url) and is_404:
					continue
				
				if cur_url != last_url:
					# emit the last set of comments
					if (not is_404) and last_url != '':
						record["label"] = ''
						record["showlabel"] = ''
						record["tendency"] = ''
						record["series"] = series
						record["spec"] = last_spec
						record["web"] = "xcar"
						record["url"] = last_url
						record["comment"] = ''
						record["date"] = last_date
						print last_url

						# filter out stamps
						comment_all = ''.join(re.split("\[.*本帖最后.*\]", comment_all))
						for i in range(len(last_valid_labels)):
							for j in re.split(u"；|;|~|\?|？|！|!|。|\s", last_valid_comments[i]):
								if len(j) == 0:
									continue
								comment_all = ''.join(comment_all.split(j))

						comment_all = re.split(u"；|;|~|\?|？|！|!|。|\s", comment_all)
						# print "splited..."
						for comment in comment_all:
							# omit those with explict tags: [外观] etc
							comment = re.sub(u"\[.{0,4}\]", "", comment)
							comment = re.sub(u"【.{0,4}】", "", comment)
							if len(comment) < 4:
								continue
							if re.match("^\w*$", comment):
								continue
							# print comment, len(comment)
							last_valid_labels.append("neutral")
							last_valid_comments.append(comment)
							last_valid_tendency.append(None)
						# print len(last_valid_labels), len(last_valid_comments)
						for i in range(len(last_valid_labels)):
							record["label"] = last_valid_labels[i]
							record["comment"] = last_valid_comments[i].decode("utf8")
							record["tendency"] = last_valid_tendency[i]
							store_comment(web_name, record, "")
							max_num -= 1
						if max_num <= 0:
							break

					# get all comments for this url
					r = requests.get(cur_url)
					soup2 = BeautifulSoup(r.text, "lxml")
					is_404 = (soup2.select("div.l_error") != [])
					if is_404:
						print "404 error!"
						last_url = cur_url
						continue
					
					comment_all = ''
					for div in soup2.select(".F_box_2")[1:]:
						# not the poser's comment.
						if div.table.tr.td.get("style").find("original.png") == -1:
							break
						line = div.select(".line .t_msgfont1")
						if line == []:
							continue
						line = line[0]
						try:
							line.style.extract()
							line.table.extract()
						except:
							hahah = 1
						for part in re.split(u"\n*", line.text.strip()):
							# MUST ignore! Or it will raise error!							
							text = part.strip().encode('iso8859-1', 'ignore').decode("gb2312", "ignore")
							comment_all += text + "\n"
					# print comment_all
					# print soup2.select("div.t_number")[0].next_sibling.next_sibling.text.encode('iso8859-1', 'ignore').decode("gb2312", "ignore")
					last_date = soup2.select("div.t_number")[0].next_sibling.next_sibling.text.encode('iso8859-1', 'ignore').decode("gb2312", "ignore").strip().split()[1]
					print last_date
					last_url = cur_url
					last_spec = ''	
					last_comment_all = comment_all
					last_valid_labels = []		
					last_valid_comments = []		
					last_valid_tendency = []		

				label = item.dt.em.a.text.encode('iso8859-1', 'ignore').decode("gb2312", "ignore")
				label = "【" + label + "】"
				label = tran_tag[label.encode("utf-8")]
				tendency = item.dt.text.encode('iso8859-1', 'ignore').decode("gb2312", "ignore").split(u"【")[1].split("】")[0]
				tendency = tendency_list[tendency]
				last_valid_labels.append(label)
				cur_comment = item.dd.text.split("\r\n")[-1].encode('iso8859-1', 'ignore').decode("gb2312", "ignore")
				last_valid_comments.append(cur_comment)
				last_valid_tendency.append(tendency)
				# print tendency, cur_comment
				
			next_page = soup.body.select(".pagers .on")
			if next_page != [] and next_page[0].next_sibling != None:
				page_num += 1
				url_comments = "http://newcar.xcar.com.cn" + next_page[0].next_sibling["href"]
				print "another_page, ", url_comments
			else:
				flag = 0

		except Exception as e:
			print e
			print url_comments
			traceback.print_exc(file=sys.stdout)
			flag = 0

def xcar_crawler_for_config(brand, series, url_base, last_visit, last_content, max_num):
	web_name = "xcar"
	flag = 1
	page_num = 1
	url_comments = url_base

	# Find comments from a specific user.
	last_url = ''
	last_spec = ''
	last_comment_all = ''
	last_url = ''
	last_date = ''
	last_valid_comments = []
	last_valid_labels = []
	is_404 = False
	while(flag and max_num > 0):
		try:
			r = requests.get(url_comments)
			soup = BeautifulSoup(r.text, "lxml")
		
			for item in soup.body.select("div.review_comments_dl > dl"):
				record = collections.OrderedDict()
				cur_url = item.select("dt > a")[0]["href"]
				label = item.dt.em.a.text.encode('iso8859-1', 'ignore').decode("gb2312", "ignore")
				label = "【" + label + "】"
				label = tran_tag[label.encode("utf-8")]
				if label != "configuration":
					continue
				cur_comment = item.dd.text.split("\r\n")[-1].encode('iso8859-1', 'ignore').decode("gb2312", "ignore")
				record["label"] = label
				record["showlabel"] = ''
				record["tendency"] = ''
				record["series"] = series
				record["spec"] = ''
				record["web"] = "xcar"
				record["url"] = cur_url
				record["comment"] = cur_comment
				record["date"] = ''
				store_comment(web_name, record, "")

			next_page = soup.body.select(".pagers .on")
			if next_page != [] and next_page[0].next_sibling != None:
				page_num += 1
				url_comments = "http://newcar.xcar.com.cn" + next_page[0].next_sibling["href"]
				print "another_page, ", url_comments
			else:
				flag = 0

		except Exception as e:
			print e
			print url_comments
			traceback.print_exc(file=sys.stdout)
			flag = 0

def signal_handler(signum, frame):
	global sig_stop
	sig_stop = True
	return

def main():
	global url_pool, sig_stop

	sig_stop = False
	signal.signal(signal.SIGINT, signal_handler)

	# get all series!
	if not os.path.exists("url_pool.json"):
		url_pool = get_car_series()
		with open("url_pool.json", "w") as f:
			json.dump(url_pool, f)
		f.close()
	else:
		f = open("url_pool.json", "r")
		url_pool = json.load(f)
		f.close()

	# get all car lists & crawl basic!
	# url_series = "http://car.autohome.com.cn/config/series/66.html"
	# r = requests.get(url_series)
	# car_list_str = r.text.split("specIDs =")[1].split(";")[0]
	# car_list = [int(s.strip()) for s in car_list_str[1:-1].split(",")]
	# for car_id in car_list:       
	#   url_basic = "http://car.autohome.com.cn/config/spec/" + str(car_id) + ".html"
	#   car_name = crawl_basic(url_basic, car_id)

	# sort the url by last_visit.

	url_cnt = 0
	for url_comment in url_pool:
		if not sig_stop:
			if url_comment["web"] == "xcar":
				print url_comment["series"]
				eval(url_comment["web"]+"_crawler")(url_comment["brand"], url_comment["series"], url_comment["url"], url_comment["last_visit"], url_comment["last_content"], max_comment_num)
				url_cnt += 1
		else:
			print url_cnt
			print "writing ..."
			with open("url_pool.json", "w") as f:
				json.dump(url_pool[url_cnt:], f)
			f.close()
			sys.exit()

main()

