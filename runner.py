import numpy as np
from amc_clustering import AMC_cluster as cluster
import sys
import networkx as nx
from time import sleep
import random
#---- YOUTUBE --- #

# karate_club = nx.read_gml('karate/karate.gml')

# dist = cluster(nx.to_numpy_matrix(karate_club))
# print dist
# sys.exit(1)
# # seeds = []
# # for line in f:
# # 	# print line.split()
# # 	intlist = [int(x) for x in line.split()]
# # 	seeds.append(random.choice(intlist))
# # 	for node in intlist:
# # 		comms[node] = counter
# # 		nodes.add(node)
# # 	# nodes.add(intlist)
# # 	# sleep(5)
# # 	# comms[counter] = intlist


# graph = nx.read_edgelist('com-youtube.ungraph.txt')
# print "done reading in graph"
# seeds = cluster(nx.to_numpy_matrix(graph))
# print seeds
# sys.exit(1)
# # seeds = []
# # for i in range(len(comms)):
# 	# seeds.append(random.choice(comms[i]))

# dist = cluster(nx.to_numpy_matrix(graph))
# correct = 0
# for node in nodes:
# 	if np.argmax[dist[node]] == comms[node]:
# 		correct += 1

# print "Correct:%d. Percent: %f" % (correct, float(correct)/len(nodes))
# sys.exit(1)

# # fd = open("test_files/mat-file.txt", 'r')
# # matr = []
# # for line in fd:
# # 	splits = line.split('\t')
# # 	intsplits = [int(x) for x in splits]
# # 	matr.append(intsplits)

# # array = np.array(matr, np.float32)
# # print cluster(array, [1,6])



# # karate_club = nx.read_gml('karate/karate.gml')

# # dist = cluster(nx.to_numpy_matrix(karate_club), [,31])
# # # print dist

# # c1 = []
# # c2 = []

# # for i in range(len(dist)):
# # 	if np.argmax(dist[i]) == 0:
# # 		c1.append(i)
# # 	else:
# # 		c2.append(i)

# # print c1
# # print "\n\n\n\n\n"
# # print c2


# # gold1 = [1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22]
# # gold1 = [x-1 for x in gold1]
# # gold2 = [i for i in range(34) if i not in gold1]

# # correct = 0
# # wrong = []
# # misclass = 0
# # for i in range(34):
# # 	if i in gold1:
# # 		if i in c1:
# # 			correct +=1
# # 		else:
# # 			misclass += 1
# # 			wrong.append(i)
# # 	else:
# # 		if i in c2:
# # 			correct += 1
# # 		else:
# # 			misclass += 1
# # 			wrong.append(i)

# # print "Correct: %d Wrong: %d" % (correct,misclass)
# # print "percent right: %f" % (float(correct)/34)
# # print wrong
# # sys.exit(1)


# # football = nx.read_gml('football/football.gml')
# # degrees = football.degree()
# # sorted_list = sorted(degrees.items(), key=lambda x:x[1], reverse=True)
# # bools = {i:False for i in range(12)}
# # seeds = []
# # labels = []
# # for item in sorted_list:
# # 	if bools[football.node[item[0]]['value']] == False:
# # 		seeds.append(item[0])
# # 		labels.append(football.node[item[0]]['value'])
# # 		bools[football.node[item[0]]['value']] = True

# # print seeds
# # print labels
# # # sys.exit()
# # print nx.to_numpy_matrix(football)
# # sys.exit(1)
# # dist = cluster(nx.to_numpy_matrix(football),seeds)
# # clusterlabels = {}
# # for i in range(len(dist)):
# # 	clusterlabels[i] = labels[np.argmax(dist[i])]

# # goldclusterlabels = {}
# # for node in football.nodes():
# # 	goldclusterlabels[node] = football.node[node]['value']

# # correct = 0
# # wrong = 0
# # wrongs = []
# # for node in football.nodes():
# # 	if clusterlabels[node] == goldclusterlabels[node]:
# # 		correct += 1
# # 	else:
# # 		wrong += 1
# # 		wrongs.append(node)

# # print "Correct:%d, wrong:%d, percent:%f" % (correct, wrong, float(correct)/(wrong+correct))
# # print wrongs
# # for wrong in wrongs:
# # 	print clusterlabels[wrong]
# # 	print goldclusterlabels[wrong]
# # 	print dist[wrong]
# # 	print '\n\n'
# # sys.exit(1)




# --------POLBOOKS ----- #

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
print books.node[3]['value']
print books.node[8]['value']
print books.node[84]['value']
print books.node[65]['value']
print degrees[65]
# sys.exit(1)
for item in sorted_list:
	if bools[books.node[item[0]]['value']] == False:
		seeds.append(item[0])
		labels.append(books.node[item[0]]['value'])
		bools[books.node[item[0]]['value']] = True
print seeds
for seed in seeds:
	print degrees[seed]
# sys.exit(1)
# print labels
# sys.exit(1)
dist,seeds = cluster(nx.to_numpy_matrix(books),K=3)#seeds)
print "\n\n\n\n\n"
print seeds
print "\n"
for seed in seeds:
	print seed
	print degrees[seed]
# sys.exit(1)
c = []
l = []
n = []

for i in range(len(dist)):
	if np.argmax(dist[i]) == 0:
		# c.append(i)
		c.append(i)
	elif np.argmax(dist[i]) == 1:
		# l.append(i)
		n.append(i)
	else:
		# n.append(i)
		l.append(i)

print "C"
for node in c:
	print books.node[node]['value']

print "\nL"
for node in l:
	print books.node[node]['value']
print "\nN"
for node in n:
	print books.node[node]['value']

sys.exit(1)


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
print karate_club.degree()
# sys.exit(1)
arr = nx.to_numpy_matrix(karate_club)
for i in range(len(arr)):
	print i, np.sum(arr[i])
# sys.exit(1)
dist = cluster(nx.to_numpy_matrix(karate_club))#, [0,32])
# print dist
# sys.exit(1)

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

