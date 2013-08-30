import numpy as np
from agglo import agglomerative_cluster
from agglo_em import cluster
import sys

fd = open("test_files/mat-file.txt", 'r')
matr = []
for line in fd:
	splits = line.split('\t')
	intsplits = [int(x) for x in splits]
	matr.append(intsplits)

array = np.array(matr, np.float16)

dist, seeds = cluster(array)
print dist
print "len seeds:" + str(len(seeds))
