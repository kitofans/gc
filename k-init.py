import io, os, sys
import math
from time import sleep
import numpy as np


def normalize(vector, normalize_to=1):
	total = np.sum(vector)
	for index in range(len(vector)):
		vector[index] = (vector[index]/float(total))*normalize_to


def check_end(index, N_mat, seeds, K):
	if K != None:
		if len(seeds) >= K:
			return True
	for seed in seeds:
		row = N_mat[seed]
		if row[index] > .01*np.average(row):
			return True
	return False

def find_seeds(N_mat, K):
	seeds = []
	# Note: This is a heuristic and by no means is optimized
	# --- 1: Find the index with the highest variance.
	index_of_min = np.argmin(N_mat)
	tuple_index = np.unravel_index(index_of_min, (len(N_mat),len(N_mat)))
	for index in tuple_index:
		seeds.append(index)

	while(True):
		summed_steps = N_mat[seeds[0]]
		for seed in seeds[1:]:
			summed_steps = np.add(summed_steps, N_mat[seed])
		indexes_of_min = summed_steps.argsort()[:3]
		count = -1
		for index_of_min in indexes_of_min:
			if check_end(index_of_min, N_mat, seeds, K):
				continue
			else:
				count = 1
				seeds.append(index_of_min)
		if count == -1:
			break

	# next_index = tuple_index[1]
	# while(True):
		# next_index = tuple_index[1]
		# next_min_index = np.argmin(N_mat[next_index])
		# if next_min_index in seeds:
			# break
		# else:
			# seeds.append(next_min_index)
		# next_index = next_min_index
	print "LEN SEEDS:" + str(len(seeds))
	return seeds



def initialize(array,seeds, cluster_prob, K):
	t = len(array)
	# -- 1: Create Q, the t by t probability matrix that represents transitioning from
	# ----- a transient state to a transient state. 
	print "> normalizing Q..."
	Q = np.array(array,copy=True) # Copy to leave the original array intact
	for index in range(t):
		normalize(Q[index], 1-cluster_prob)
		# if index in seeds:
			# normalize(Q[index], 1 - cluster_prob)
		# else:
			# normalize(Q[index])
	print "> finding N..."
	N = find_N(Q, t)

	if seeds == None:
		print "> finding seeds..."
		seeds = find_seeds(N, K)

	r = len(seeds)

	# -- 2: Create R, the t by r probability matrix that represents transitioning from
	# ----- a transient state to an absorption state.
	R = np.zeros((t,r),np.float16)

	# -- 3: Initialization puts the probability of going from a seed to it's 
	# ----- cluster at cluster_prob, and from any other node to a cluster at cluster_prob/r.
	print "> normalizing R..."
	for index in range(t):
		if index in seeds:
			R[index][seeds.index(index)] = cluster_prob
		else:
			R[index] = [cluster_prob/r for i in range(r)]
	
	return R, Q, r, t, N, seeds


# def diff(dists, old_dists):
# 	if old_dists == []:
# 		return 100
# 	else:
# 		total = 0
# 		for i in range(len(dists)):
# 			for j in range(len(dists[0])):
# 				toadd = abs(dists[i][j] - old_dists[i][j])
# 				total += toadd
# 				if toadd > .01:
# 					print "%d,%d went from %f to %f."% (i,j,dists[i][j], old_dists[i][j])
# 		return total


def test_array(array):
	for row in array:
		print row
		print sum(row)


def find_N(Q_mat, t_size):
	# -- 1: Note that N = (I - Q)^-1. Numpy makes this easy.
	I_mat = np.identity(t_size)
	to_invert = np.subtract(I_mat, Q_mat)
	return np.linalg.inv(to_invert)

def find_distributions_E(N_mat, R_mat):
	# -- 1: Note that the probability that a walker starting from transient state 
	# ----- i will be aborbed in absorbing state j is the (i,j)th entry of the 
	# ----- matrix B = NR.
	B_mat = N_mat.dot(R_mat)
	# print B_mat
	# sys.exit(1)
	return B_mat

def normalize_to_max(row):
	argmax = np.argmax(row)
	row = [0 for i in range(len(row))]
	row[argmax] = 1

def rewire_clusters_M(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob, fuzzy): # TO LOOK FOR: DOES THE IMPLEMENTATION TRY TO MAKE ALL CLUSTERS SAME SIZE (I.E IT SHOULD TAKE INTO ACCOUNT THAT SOME WILL BE SMALLER)
	# -- 1: First, we normalize the distribution_mat column wise to find the nodes
	# ----- that characterize the clusters best. 

	# print distribution_mat
	# print "\n\n"
	column_wise_mat = np.copy(np.transpose(distribution_mat))
	for row in column_wise_mat:
		if fuzzy:
			normalize(row)
		else:
			normalize_to_max(row)
	# print column_wise_mat
	# print distribution_mat
	# print "\n\n"

	# Note: Now the (j,i)th entry in column_wise_mat is the percent of random walkers who
	# ----- end up in absorption state j that began in transient state i. We want to rewire
	# ----- each cluster node in such a way that a random walk starting randomly at one of the nodes
	# ----- (with %chance of starting at i given by column_wise_mat[clusternode][i]) is most likely
	# ----- to end up at the given cluster node. This is given by rewiring R in such a way that 
	# ----- (column_wise_mat[j] * B)[j] is maximized. Now we know column_wise_mat[j], B is simply
	# ----- N * R and we know N, so we have one variable to solve for. And we know that order of 
	# ----- operations doesn't matter, so...

	# -- 2: Penalize each element by it's association to the cluster. The first normalized value is
	# ----- something like "distance", though it doesn't seem it... perhaps look at expected
	# ----- values of steps? It's being weighted here by "membership", which does look correct. 
	# print distribution_mat 
	# print "\n\n\n"
	if fuzzy:
		for j in range(r_size):
			for i in range(t_size):
				column_wise_mat[j][i] *= distribution_mat[i][j]
	
	# print distribution_mat

	# for row in column_wise_mat:
		# normalize(row)

	

	dist_times_N_mat = column_wise_mat.dot(N_mat)
	# transposed = np.transpose(dist_times_N_mat)
	# for index in range(t_size):
	argmaxes = np.argmax(dist_times_N_mat, axis=0)
	for index in range(t_size):
	# for argmax in argmaxes:
		R_mat[index] = [0 for i in range(r_size)]
		R_mat[index][argmaxes[index]] = cluster_prob

	# print argmaxes
	# print "\n\n"


	# print dist_times_N_mat
	# print "\n\n"
	# print R_mat
	# print "\n\n"
	# print dist_times_N_mat.dot(R_mat)
	# sys.exit(1)
	# for j in range(r_size):
		# pass

	# print distribution_mat


def diff(distone, disttwo):
	# print distone
	# print "\n\n"
	# print disttwo
	return np.sum(np.absolute(np.subtract(distone, disttwo)))

def diff_check(distone, disttwo, epsilon):
	difference = diff(distone, disttwo)
	if difference < epsilon:
		return True
	else:
		# print "Current difference: %f" % difference
		# print disttwo
		return False



def cluster(original_array, seeds=None, cluster_prob=.5, epsilon=.001, K=None, fuzzy=True):
	print "> initializing..."
	R_mat, Q_mat, r_size, t_size, N_mat,seeds = initialize(original_array,seeds,cluster_prob, K)
	# N_mat = find_N(Q_mat, t_size)
	distribution_mat = find_distributions_E(N_mat, R_mat)
	# print "ORIGINAL:"
	# print distribution_mat
	print "> iterating..."
	counter = 1
	while (True):
		print "> on iteration %d" % counter
		# if diff(cluster_distributions, old_cluster_distributions) < epsilon:
			# break
		# print R_mat
		# print "\n\n"
		rewire_clusters_M(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob, fuzzy)
		# print R_mat
		# maximization_step(array, cluster_distributions, cluster_prob, old_size, new_size)
		# print "NEW:"
		new_distribution_mat = find_distributions_E(N_mat, R_mat)
		# print new_distribution_mat
		# print "OLD:"
		# print distribution_mat
		
		# print "\n\n\n\n\n\n\n"
		# print new_distribution_mat
		# print "\n\n\n\n\n\n\n"
		if diff_check(distribution_mat, new_distribution_mat, epsilon):
			break
		distribution_mat = np.copy(new_distribution_mat)
		# print distribution_mat
		#sleep(1)
		counter += 1
		
	# print "Distributions:"
	# print distribution_mat
	return distribution_mat,seeds
