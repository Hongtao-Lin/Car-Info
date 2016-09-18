import numpy as np
from sklearn.decomposition import PCA	
import matplotlib.pyplot as plt
pre = np.load("pre_trained_word_vector.npy")[0]
print pre.shape

after = np.load("new_word_vec_l0.3.npy")
print after.shape

pca = PCA(n_components=2)
pree = pca.fit_transform(pre)
afterr = pca.fit_transform(after)
print "xiao", pree[92], afterr[92]
print "da", pree[23], afterr[23]
print "duo", pree[42], afterr[42]
print "shao", pree[635], afterr[635]
print "hao", pree[184], afterr[184]
print "youxiu", pree[439], afterr[439]
print "haokan", pree[11183], afterr[11183]
print "piaoliang", pree[4000], afterr[4000]
print "choulou", pree[17852], afterr[17852]
print "nankan", pree[31910], afterr[31910]
xiao_pre = [pree[92][0], pree[92][1]]
da_pre = [pree[23][0], pree[23][1]]
duo_pre = [pree[42][0], pree[42][1]]
shao_pre = [pree[635][0], pree[635][1]]

xiao_after = [afterr[92][0], afterr[92][1]]
da_after = [afterr[23][0], afterr[23][1]]
duo_after = [afterr[42][0], afterr[42][1]]
shao_after = [afterr[635][0], afterr[635][1]]
