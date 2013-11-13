import numpy as np
from amc_clustering import AMC_cluster as cluster
import sys
import networkx as nx
from time import sleep

# fd = open("test_files/mat-file.txt", 'r')
# matr = []
# for line in fd:
# 	splits = line.split('\t')
# 	intsplits = [int(x) for x in splits]
# 	matr.append(intsplits)

# array = np.array(matr, np.float32)
# print cluster(array, [1,6])



books = nx.read_gml('polbooks/polbooks.gml')
# print books.nodes(data=True)
degrees = books.degree()
sorted_list = sorted(degrees.items(), key=lambda x:x[1], reverse=True)
# print sorted_list
bools = {'n':False, 'c':False, 'l':False}
n = False
c = False
l = False
seeds = []
labels = []
for item in sorted_list:
	if bools[books.node[item[0]]['value']] == False:
		seeds.append(item[0])
		labels.append(books.node[item[0]]['value'])
		bools[books.node[item[0]]['value']] = True
print seeds
print labels
# sys.exit(1)
dist = cluster(nx.to_numpy_matrix(books),seeds)
c = []
l = []
n = []

for i in range(len(dist)):
	if np.argmax(dist[i]) == 0:
		c.append(i)
	elif np.argmax(dist[i]) == 1:
		l.append(i)
	else:
		n.append(i)

gc = []
gl = []
gn = []
for node in books.nodes():
	if books.node[node]['value'] == 'c':
		gc.append(node)
	elif books.node[node]['value'] == 'l':
		gl.append(node)
	else:
		gn.append(node)

correct = 0
wrong = 0
wrongs = []
for node in books.nodes():
	if node == 1:
		print (node in c)
		print (node in gc)
		# sleep(5)
	if node in c:
		if node in gc:
			correct += 1
		else:
			wrong += 1
			wrongs.append(node)
	elif node in l:
		if node in gl:
			correct += 1
		else:
			wrong += 1
			wrongs.append(node)
	else:
		if node in gn:
			correct += 1
		else:
			wrong += 1
			wrongs.append(node)

print "Correct: %d. Wrong: %d. Percent: %f" % (correct, wrong, float(correct)/(correct+wrong))
print wrongs

# for node in wrongs:

# 	print '\n'
# 	print node
# 	print books.node[node]['value']
# 	if node in n:
# 		print 'n'
# 	elif node in l:
# 		print 'l'
# 	elif node in c:
# 		print 'c'
# 	print '\n'



sys.exit(1)


# --- KARATE CLUB ----- #

karate_club = nx.read_gml('karate/karate.gml')

dist = cluster(nx.to_numpy_matrix(karate_club), [0,31])
# print dist

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

