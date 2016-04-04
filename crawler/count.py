#-*- coding: UTF-8 -*-

import sys, os

file = open("data/宝马3系.txt", "r")
text = file.read()
print text.count("other:")
	
