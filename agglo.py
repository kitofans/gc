import io, os, sys
import math
from time import sleep
import numpy as np


def normalize(vector, normalize_to=1):
	total = np.sum(vector)
	for index in range(len(vector)):
		vector[index] = (vector[index]/float(total))*normalize_to



def initialize(array, cluster_prob, K):
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

	r = t

	# -- 2: Create R, the t by r probability matrix that represents transitioning from
	# ----- a transient state to an absorption state.
	R = np.zeros((t,r),np.float16)

	# -- 3: Initialization puts the probability of going from each index to it's 
	# ----- cluster at cluster_prob.
	print "> normalizing R..."
	for index in range(t):
		R[index][index] = cluster_prob
		
	
	return R, Q, r, t, N


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


	

	dist_times_N_mat = column_wise_mat.dot(N_mat)
	argmaxes = np.argmax(dist_times_N_mat, axis=0)
	for index in range(t_size):
		R_mat[index] = [0 for i in range(r_size)]
		R_mat[index][argmaxes[index]] = cluster_prob


def diff(distone, disttwo):
	return np.sum(np.absolute(np.subtract(distone, disttwo)))

def diff_check(distone, disttwo, epsilon):
	difference = diff(distone, disttwo)
	if difference < epsilon:
		return True
	else:
		return False



def agglomerative_cluster(original_array, cluster_prob=.4, epsilon=.001, K=None, fuzzy=True):
	print "> initializing..."
	R_mat, Q_mat, r_size, t_size, N_mat = initialize(original_array,seeds,cluster_prob, K)
	distribution_mat = find_distributions_E(N_mat, R_mat)
	print "> iterating..."
	counter = 1
	while (True):
		print "> on iteration %d" % counter
		rewire_clusters_M(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob, fuzzy)
		new_distribution_mat = find_distributions_E(N_mat, R_mat)
		if diff_check(distribution_mat, new_distribution_mat, epsilon):
			break
		distribution_mat = np.copy(new_distribution_mat)
		counter += 1
		
	return distribution_mat,seeds
