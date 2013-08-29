import io, os, sys
import math
from time import sleep
import Pycluster
import numpy as np


def normalize(vector, normalize_to=1):
	total = np.sum(vector)
	for index in range(len(vector)):
		vector[index] = (vector[index]/float(total))*normalize_to



def initialize(array, cluster_prob):
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

	clusters = {x:[x] for x in range(r)}

	# -- 2: Create R, the t by r probability matrix that represents transitioning from
	# ----- a transient state to an absorption state.
	R = np.zeros((t,r),np.float16)

	# -- 3: Initialization puts the probability of going from each index to it's 
	# ----- cluster at cluster_prob.
	print "> normalizing R..."
	for index in range(t):
		R[index][index] = cluster_prob
		
	
	return R, Q, r, t, N, clusters


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


def find_cluster_ratio(index, B_mat, N_mat, t_size):
	starting_prob_dist = np.array([0 for i in range(len(B_mat))])
	starting_prob_dist[index] = 1
	ending_prob_dist = starting_prob_dist.dot(B_mat)
	total_prob = ending_prob_dist[index]
	H_mat = find_transient_prob(N_mat, t_size)
	ending_visit_dist = starting_prob_dist.dot(H_mat)
	total_steps = ending_visit_dist[index]
	return total_prob/(3*total_steps)



def find_distributions_E(N_mat, R_mat, t_size, init=False):
	# -- 1: Note that the probability that a walker starting from transient state 
	# ----- i will be aborbed in absorbing state j is the (i,j)th entry of the 
	# ----- matrix B = NR.
	B_mat = N_mat.dot(R_mat)
	if init:
		ratios = {}
		for i in range(len(R_mat)):
			ratios[i] = find_cluster_ratio(i, B_mat, N_mat, t_size)
		return B_mat, ratios
	else:
		return B_mat

def normalize_to_max(row):
	argmax = np.argmax(row)
	row = [0 for i in range(len(row))]
	row[argmax] = 1

def find_transient_prob(N_mat, t_size):
	identity = np.identity(t_size)
	n_minus_i = np.subtract(N_mat, identity)
	to_invert = np.diag(np.diag(N_mat))
	inverted = np.linalg.inv(to_invert)
	return n_minus_i.dot(inverted)



def cluster_eligibility(i, other_i, N_mat, distribution_mat, clusters, ratios, t_size):
	print "LOOKING AT %d, %d" % (i, other_i)
	orig_i = ratios[i]
	orig_other_i = ratios[other_i]

	if orig_i == 0 or orig_other_i == 0:
		return False, 0

	relavent_nodes = clusters[i] + clusters[other_i]
	# print relavent_nodes
	starting_prob_dist = np.array([0 for j in range(len(distribution_mat))], np.float16)
	for node in relavent_nodes:
		try:
			starting_prob_dist[node] = 1/float(len(relavent_nodes))
		except:
			print "node %d out of range" % node
			print distribution_mat
			sys.exit(1)

	# print starting_prob_dist


	ending_prob_dist = starting_prob_dist.dot(distribution_mat)
	# print "ending prob dist" + str(ending_prob_dist)
	total_prob = ending_prob_dist[i] + ending_prob_dist[other_i]
	H_mat = find_transient_prob(N_mat, t_size)
	ending_visit_dist = starting_prob_dist.dot(H_mat)
	total_steps = 0
	for node in relavent_nodes:
		total_steps += ending_visit_dist[node]
		print "Chance to visit node %d: %f" % (node, ending_visit_dist[node])

	total_steps = total_steps/len(relavent_nodes)

	merged_i_other_i = total_prob/(total_steps)
	# print "total prob: " + str(total_prob)
	# print "total steps: " + str(total_steps)
	# print "\n\n"

	if merged_i_other_i*2 > orig_i + orig_other_i:
		print "MERGING %d, %d" % (i, other_i)
		print orig_i
		print orig_other_i
		print merged_i_other_i
		return True, merged_i_other_i
	else:
		print "NOT MERGING"
		print orig_i
		print orig_other_i
		print merged_i_other_i
		return False, 0



def merge_clusters(ratios, clusters, i, other_i, R_mat, merged_ratio, cluster_prob):
	clusters[i] += clusters[other_i]
	clusters[other_i] = []
	ratios[i] = merged_ratio
	ratios[other_i] = 0
	for node in clusters[i]:
		R_mat[node][i] = cluster_prob
		R_mat[node][other_i] = 0


def rewire_clusters_M(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob, distance_cutoff, clusters, ratios): # TO LOOK FOR: DOES THE IMPLEMENTATION TRY TO MAKE ALL CLUSTERS SAME SIZE (I.E IT SHOULD TAKE INTO ACCOUNT THAT SOME WILL BE SMALLER)
	# -- 1: First, we normalize the distribution_mat column wise to find the nodes
	# ----- that characterize the clusters best. 

	# print distribution_mat
	# print "\n\n"
	count = 0
	column_wise_mat = np.copy(np.transpose(distribution_mat))
	# print column_wise_mat
	# labels, error, nfound = Pycluster.kcluster(column_wise_mat, 2)
	# print labels
	# print nfound
	# sys.exit(1)

	print column_wise_mat
	# sys.exit(1)
	for i in range(r_size):
		for other_i in range(i+1,r_size):
			print "looking at %d, %d" % (i, other_i)
			eligibility, merged_ratio = cluster_eligibility(i, other_i, N_mat, distribution_mat, clusters, ratios, t_size)
			if eligibility:
				merge_clusters(ratios,clusters,i,other_i, R_mat, merged_ratio, cluster_prob)
				count += 1

	return count


def diff(distone, disttwo):
	return np.sum(np.absolute(np.subtract(distone, disttwo)))

def diff_check(distone, disttwo, epsilon):
	difference = diff(distone, disttwo)
	if difference < epsilon:
		return True
	else:
		return False



def agglomerative_cluster(original_array, cluster_prob=.4, epsilon=.001, distance_cutoff=.2):
	print "> initializing..."
	R_mat, Q_mat, r_size, t_size, N_mat, clusters= initialize(original_array,cluster_prob)
	distribution_mat, ratios = find_distributions_E(N_mat, R_mat, t_size, init=True)
	print "> iterating..."
	counter = 1
	while (True):
		print "> on iteration %d" % counter
		print R_mat
		# sleep(3)
		num_rewired = rewire_clusters_M(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob, distance_cutoff, clusters, ratios)
		new_distribution_mat = find_distributions_E(N_mat, R_mat, t_size)
		distribution_mat = np.copy(new_distribution_mat)
		if num_rewired == 0:
			break
		counter += 1
		
	return clusters
