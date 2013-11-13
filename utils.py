import numpy as np

def phase(text):
	print "----------------------"
	print "> %s" % text
	print "----------------------"

def output(text):
	print "---> %s" % text

def normalize(vector, normalize_to=1):
	total = np.sum(vector)
	for index in range(len(vector)):
		vector[index] = (vector[index]/float(total))*normalize_to