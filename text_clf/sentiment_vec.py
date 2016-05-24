word_vec = {}

def load_vec():
	f = open("dict.out", "r")
	for r in f.readlines():
		word = r.split(" ")[0].decode("utf8")
		vec = r.split(" ")[1]
		y = []
		for v in vec.split(","):
			y.append(v)
		word_vec[word] = y
	f.close()

def get_vec(file_name):
	global word_vec
	f = open(file_name + ".txt", "r")
	out = open(file_name + ".out", "w")
	for r in f.readlines():
		w1 = r.split(" ")[0].decode("gb18030")
		w2 = r.split(" ")[1].decode("gb18030")
		out.write(word_vec[w1] + "\t" + word_vec[w2] + "\n")
	f.close()
	out.close()

load_vec()
get_vec("a")
get_vec("b")