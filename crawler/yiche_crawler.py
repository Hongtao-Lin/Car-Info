#-*- coding: UTF-8 -*-

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

comment_field = ["brand", "series", "spec", "date", "web", "url", "good", "bad", "space", "power", "operate", "oil", "comfort", "appearance", "decoration", "worth", "bugs", "sustain", "other", "upvote", "downvote", "respond"]

comment_dict = collections.OrderedDict()	
for field in comment_field:
	comment_dict[field] = ""

def store_comment(web_name, record, raw_comment, file):
	# split_tag = ["【最满意的一点】", "【最不满意的一点】", "【空间】"]
	# processing raw_comment

	# just a temporary test!
	record["other"] = raw_comment
	print raw_comment
	for (key, value) in record.items():
		file.write("%s: %s\n" % (key, value))

	# store it in sql.
	return

def yiche_crawler(brand, series, url_base):
	log = open("crawler_log.txt", "a")	
	file_name = series + ".txt"
	file = open(file_name, "a")

	web_name = u"易车网"
	flag = 1
	page_num = 1
	url_comments = url_base

	while(flag):
		r = requests.get(url_comments)
		soup = BeautifulSoup(r.text, "lxml")
		page_all = soup.body.find_all("span", id_="main_pager_down")
		if (page_all != [] and page_all[0].find_all("a", class_="next_on") != []):
			page_num += 1
			url_comments = url_base + "page" + str(page_num)
			print "another_page, ", page_num
		else:
			flag = 0
		
		# Find comments from a specific user.
		for item in soup.body.find_all("div", id="topiclistshow")[0].find_all("dl"):
			# try:
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
				
				r = requests.get(record["url"])
				print url_comment.a.get("href")
				soup = BeautifulSoup(r.text, "lxml")

				# filter some special comments.
				if soup.select("span.fapiao_tab") != []:
					continue

				comment = soup.select("div#content_bit div.article-contents")[0]
				
				record["respond"] = respond = url_comment.select("div.rbox")[0].a.span.text[3:-1].strip()
				record["upvote"] = upvote = url_comment.select("div.rbox em")[-1].text[1:-1].strip()
				record["date"] = date = soup.select("#time")[0].text.strip()
				print "respond is: %s" % respond
				print "upvote is: %s" % upvote
				print "date is: %s" % date 

				if comment.select("p.czjg_xq_cont") != []: # which means that comment is not valid.
					continue
				# pre-process the text for convenience.
				for tag in comment.find_all("strong"):
					if not tag.string:
						continue
					tag.string = "[" + tag.string.strip() + "]"
					# print tag.text
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

				store_comment(web_name, record, comment.text.strip(), file)

			# except Exception as e:
			# 	print url_comment.a.get("href")
			# 	print e
			# 	log.write("Error when crawling page: " + url_comment.a.get("href") + "\n\n")

	file.close()
	log.close()
	# print html

def main():
	# just a temperary name. Will change later.
	series = u"宝马3系"
	brand = u"宝马"
	dir = os.getcwd() + "/data/"
	if not os.path.exists("data/"):
		os.makedirs("data/")
	os.chdir("data/")
	# a car series
	url_series = "http://car.bitauto.com/baoma3xi/koubei/tags/%E7%BB%BC%E5%90%88/"
	
	r = requests.get(url_series)

	soup = BeautifulSoup(r.text, "lxml")

	url_comment = "http://car.bitauto.com/baoma3xi/koubei/tags/%E7%BB%BC%E5%90%88/"
	yiche_crawler(brand, series, url_comment)

	#cur.close()
	#conn.close()

main()
