import numpy as np
from amc_clustering import AMC_cluster as cluster
import sys
import networkx as nx

# fd = open("test_files/mat-file.txt", 'r')
# matr = []
# for line in fd:
# 	splits = line.split('\t')
# 	intsplits = [int(x) for x in splits]
# 	matr.append(intsplits)

# array = np.array(matr, np.float32)
# print cluster(array, [1,6])

karate_club = nx.read_gml('karate/karate.gml')
# print karate_club.nodes()
# print nx.to_numpy_matrix(karate_club)
# print len(nx.to_numpy_matrix(karate_club))
dist = cluster(nx.to_numpy_matrix(karate_club), [0,31])
print dist

c1 = []
c2 = []

for i in range(len(dist)):
	if np.argmax(dist[i]) == 0:
		c1.append(i)
	else:
		c2.append(i)

print c1
print "\n\n\n\n\n"
print c2


gold1 = [1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22]
gold1 = [x-1 for x in gold1]
gold2 = [i for i in range(34) if i not in gold1]

correct = 0
wrong = []
misclass = 0
for i in range(34):
	if i in gold1:
		if i in c1:
			correct +=1
		else:
			misclass += 1
			wrong.append(i)
	else:
		if i in c2:
			correct += 1
		else:
			misclass += 1
			wrong.append(i)

print "Correct: %d Wrong: %d" % (correct,misclass)
print "percent right: %f" % (float(correct)/34)
print wrong

