
# from car_processing

def shuffle_comments():

	conn, cur = init_mysql()

	# shuffle comments.
	print "shuffling comments..."

	cur.execute("create table car.c select * from comments limit 1")
	cur.execute("delete from car.c")

	cur.execute("select count(*) from car.comments;")
	total_comment = int(cur.fetchone()[0])
	chunk_size = 50000
	print "total comment is: ", total_comment

	for chunk in range( (total_comment-1) / chunk_size + 1):
		print "processing %d / %d" % (chunk, total_comment / chunk_size)
		sql = 'insert into car.c select * from car.comments order by rand() limit %d, %d' % (chunk*chunk_size, chunk_size)
		if chunk == (total_comment-1) / chunk_size:
			sql = 'insert into car.c select * from car.comments order by rand() limit %d, %d' % (chunk*chunk_size, total_comment % chunk_size)
		cur.execute(sql)
		conn.commit()

	cur.execute("drop table comments;")
	cur.execute("rename table c as comments;")
	conn.commit()

	cur.close()
	conn.close()


def bayes_train(load_file, bayes_type, ratio):
	print "begin traning bayes..."
	conn, cur = init_mysql()

	with open(load_file) as f:
		word_dic = cPickle.load(f)
		vec_type = cPickle.load(f)
		x = cPickle.load(f)
		y = cPickle.load(f)
		idx = cPickle.load(f)
		x = x.tocsr()
		y = np.array(y.todense()).ravel()
		idx = np.array(idx.todense()).ravel()
	print "load data complete"
	# x = preprocessing.normalize(x)

	# x, y, idx = shuffle(x, y, idx, random_state = 42)

	if bayes_type == "m":
		clf = naive_bayes.MultinomialNB()
	else:
		clf = naive_bayes.BernoulliNB()

	train_len = int(0.8*ratio*y.shape[0])
	train_y = y[:train_len]
	train_x = x[:train_len]

	clf.fit(train_x, train_y)

	sql = 'select count(*) from car.info where %s' % limitation
	cur.execute(sql)
	total_count = int(cur.fetchone()[0])
	print total_count
	test_len = int(0.2*total_count)
	# test_len = 1000
	test_y = []
	pred_y = []
	# test_x = []

	sql = 'select comment, label from car.info where %s limit %d, %d' % (limitation, total_count-test_len, total_count)
	test_count = cur.execute(sql)
	filename = "res/error_log_%s_%s.txt" % (vec_type, bayes_type)
	f = open(filename, "w")

	for (data, label, url) in cur.fetchall():
		# print len(pred_y)
		test_x = sen2vec(data, vec_type)
		pred_y.append(clf.predict(test_x))
		test_y.append(tag_dic[label])
		if ratio == 1:
			if test_y[-1] == pred_y[-1]:
				continue
			try:
				info = "\npredict: %s\ntrue: %s\n%s\n%s\n" % (tags[pred_y[-1]], tags[test_y[-1]], url, data)
				f.write(info)
			except:
				print "Writing error!\n", sql

	acc = accuracy_score(test_y, pred_y)
	score = np.empty([3, len(tags)], dtype=float)
	score[0], score[1], score[2], support =  precision_recall_fscore_support(test_y, pred_y)
	macro = np.mean(score, axis=1)
	micro = np.mean(score*support, axis=1)/np.mean(support)
		
	print "##########"
	print "ratio:", ratio
	print "type:", bayes_type
	print acc
	print score[0]
	print macro[0], micro[0]
	print score[1]
	print macro[1], micro[1]
	print score[2]
	print macro[2], micro[2]
	print support
	print "##########"
	with open("model/bayes_model.pkl", "w") as f:
		cPickle.dump(clf, f)
		cPickle.dump(word_dic, f)
		cPickle.dump(vec_type, f)
	cur.close()
	conn.close()
	return acc

# clean data.
def neutral_to_tag(load_file):
	global word_dic
	print "begin converting neutral to tag..."
	conn, cur = init_mysql()

	clf = init_model(load_file)

	sql = 'select count(*) from car.info2_cleaner where web="xcar" and label="neutral" '
	cur.execute(sql)
	total_count = int(cur.fetchone()[0])
	cur.execute(sql)
	conn.commit()

	sql = 'select id, comment from car.info2_cleaner where web="xcar" and label="neutral"'
	cur.execute(sql)
	datas = cur.fetchall()
	prog = 0
	cnt = 0
	f = open("neutral_to_tag.log", "w")
	print total_count
	for i in range(total_count):
		data = datas[i]
		if i >= total_count*prog/100:
			print "processing %d%% ..." % prog
			prog += 1
		test_x = sen2vec(data[-1], word_dic)
		pred_y = clf.predict(test_x)
		arr = clf.decision_function(test_x)[0]
		if np.max(arr) > 5 and len(data[-1].encode("utf8")) < 100:
			label = tags[pred_y]
			cnt += 1
			f.write(str(data[0]) + data[1] + label + str(len(data[-1].encode("utf8"))) + "\n")
			print data[0], data[1], label, len(data[-1].encode("utf8"))
			sql="update info2_cleaner set label = '%s' where id = %d" % (label, data[0])
			cur.execute(sql)
			conn.commit()
	cur.close()
	conn.close()
	print cnt

# for showing sentences in web.

def info_to_show():
	conn, cur = init_mysql()
	spec = ["宝马3系2016款328ixDriveM运动型", "宝马3系2015款320Li超悦版时尚型", "宝马3系2016款328Li豪华设计套装", '宝马3系2015款316i运动设计套装']
	web = ["yiche", "autohome", "pcauto", "sohu"]
	node_list = ["space","power","operation","oilwear","comfort","appearance","decoration","costperformance","failure","maintenance"];

	# for s in spec:
	# 	sql = "insert into show_comment(label, showlabel, tendency, series, spec, web, url, comment, date, id) select distinct * from info2_cleaner where spec = '%s'" % s
	# 	print sql
	# 	cur.execute(sql)
	# 	conn.commit()
	for s in spec:
		for n in node_list:
			fids = []
			print "YES!"
			sql = 'select distinct id from show_comment where spec="%s"and label = "%s" group by web order by date desc;' % (s, n)
			count = cur.execute(sql)
			tmp = cur.fetchall()
			for i in tmp:
				fids.append(int(i[0]))
			print fids
			if len(fids) == 1:
				t = fids[0]
				sql = "select id from show_comment where spec = '%s' and label = '%s' and id != %d order by date desc limit %d, %d" % (s, n, t, count, 5-count)
			elif fids == []:
				sql = "select id from show_comment where spec = '%s' and label = '%s' order by date desc limit %d, %d" % (s, n, 0, 5)
			else:
				sql = "select id from show_comment where spec = '%s' and label = '%s' and id not in %s order by date desc limit %d, %d" % (s, n, str(tuple(fids)), count, 5-count)
			cur.execute(sql)
			tmp = cur.fetchall()
			ids = []
			for i in tmp:
				ids.append(int(i[0]))
			print ids
			if fids != []:
				if len(fids) == 1:
					sql = "update show_comment set first_show = 2 where id = %d" % fids[0]
				else:
					sql = "update show_comment set first_show = 2 where id in %s" % str(tuple(fids))
				print sql
				cur.execute(sql)
				conn.commit()
			if ids != []:
				if len(ids) == 1:
					sql = "update show_comment set first_show = 1 where id = %d" % ids[0]
				else:
					sql = "update show_comment set first_show = 1 where id in %s" % str(tuple(ids))
				print sql
				cur.execute(sql)
				conn.commit()
	cur.close()
	conn.close()

info_to_show()

# cut a passage by delimiter.
def info_to_cut(dump_num):
	conn, cur = init_mysql()
	
	i = 0
	num = 0
	prog = 0

	# get # of overall data
	sql = 'select count(*) from car.%s where %s;' % (info_data, limitation)
	cur.execute(sql)
	total_num = int(cur.fetchone()[0])
	if total_num > dump_num:
		total_num = dump_num
	print "total number of data: %d" % total_num

	chunk = 0
	chunk_size = 10000
	f = open("res/segment.txt", "w")
	for chunk in range((total_num-1)/ chunk_size + 1):
		sql = 'select comment from car.%s where %s limit %d, %d;' % (info_data, limitation, chunk*chunk_size, chunk_size)
		if chunk == (total_num-1)/ chunk_size:
			sql = 'select comment from car.%s where %s limit %d, %d;' % (info_data, limitation, chunk*chunk_size, total_num % (chunk_size+1) )
		cur.execute(sql)
		for r in cur.fetchall():
			data = r[0]
			if num >= total_num*prog/100:
				print "processing %d%% ..." % prog
				prog += 1
			num += 1 
			sen_list = re.split("\)|\(|）|（|~|\?|？|！|!|，|。|,|\s", data)
			# sen_list = re.split("，|。|,|\s", data)
			data = " ".join(sen_list)
			# print data
			data = " ".join(jieba.cut(data))
			i += 1
			f.write(data)
			f.write("\n")	
	f.close()
	cur.close()
	conn.close()

