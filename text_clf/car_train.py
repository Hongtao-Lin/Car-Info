from car_processing_tool import *
import matplotlib.pyplot as plt
	
dump_num = 200000
comment_num = 500000

# comment_to_info(comment_num)
# for vec_type in ["bigram", "combined", "combined2"]:
add_info = "11_cleaner"
for vec_type in ["segment"]:
	dump_file = "data/data_%s_%s.pkl" % (vec_type, add_info)
	load_file = dump_file
	# dump_data(dump_num, dump_file, vec_type)
	# K = [0]
	# for feat_type in ["chi2", "mi"]:
	for feat_type in ["chi2", "mi"]:
		if feat_type != "":
			K = [500, 1000, 2000, 5000, 8000, 10000, 12000, 15000]
			feat_selection(dump_file, feat_type, K)
			for k in K:
				load_file = "data/data_%s_%s_%s_%d.pkl" % (vec_type, add_info, feat_type, k)
			# x_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
				x_range = [1]
				svm_range = []
				for ratio in x_range:
					acc = svm_train(load_file, 1, ratio, add_info)
					svm_range.append(acc)
					info = ""
					with open("res/acc_log.txt", "a") as f:
						info += "\n vec_type: %s, add_info: %s, feat_type: %s, k: %d \n model: %s \n acc: %s \n" % (vec_type, add_info, feat_type, k, "svm_linear", acc)
						f.write(info)
		# plt.plot(x_range, svm_range, 'ro')
		# plt.show()
