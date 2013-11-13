import io, os, sys
import math
from time import sleep
import numpy as np
from utils import *


def seed_initialize(N,t, cluster_prob, seeds):
	r = len(seeds)

	# -- 1: Create R, the t by r probability matrix that represents transitioning from
	# ----- a transient state to an absorption state.
	R = np.zeros((t,r),np.float32)

	# -- 2: Initialization puts the probability of going from a seed to it's 
	# ----- cluster at cluster_prob, and from any other node to a cluster at cluster_prob/r.

	# NOTE: Should it initialize as cluster_prob/r or 0 or something else?
	output("Normalizing R")
	for index in range(t):
		if index in seeds:
			R[index][seeds.index(index)] = cluster_prob
		else:
			R[index] = [cluster_prob/r for i in range(r)]
	
	return seeds, R, r


def find_N(Q_mat, t_size):
	# -- 1: Note that N = (I - Q)^-1. Numpy makes this easy.
	I_mat = np.identity(t_size)
	to_invert = np.subtract(I_mat, Q_mat)
	return np.linalg.inv(to_invert)

def array_initialize(array,cluster_prob):
	t = len(array)
	# -- 1: Create Q, the t by t probability matrix that represents transitioning from
	# ----- a transient state to a transient state. 

	output('Normalizing Q')
	Q = np.array(array,copy=True) # Copy to leave the original array intact
	for index in range(t):
		normalize(Q[index], 1-cluster_prob)
	
	output('Finding N')
	N = find_N(Q, t)

	return Q, N, t

def find_distributions_E(N_mat, R_mat):
	# -- 1: Note that the probability that a walker starting from transient state 
	# ----- i will be aborbed in absorbing state j is the (i,j)th entry of the 
	# ----- matrix B = NR.
	B_mat = N_mat.dot(R_mat)
	print B_mat
	return B_mat

def rewire_clusters_M_BINARY(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob): 
	# print distribution_mat

	#-- 1: First, we create the matrix V such that V_ij = I(a_j = c_i).
	V_mat = np.zeros((r_size,t_size),np.float32)
	for i in range(t_size):
		V_mat[np.argmax(distribution_mat[i])][i] = 1
	print V_mat

	# sleep(10)






	# -- 
	







	# -- And normalize

	for row in V_mat:
		normalize(row)#, normalize_to=cluster_prob)

	# -- 2: Now, we want to connect each node j to argmax_i(VN)Transpose_ji
	# print V_mat
	# print R_mat
	# sleep(10)
	VNTrans = np.transpose(V_mat.dot(N_mat))
	for j in range(t_size):
		argmax = np.argmax(VNTrans[j])
		R_mat[j] = [0 for i in range(r_size)]
		R_mat[j][argmax] = cluster_prob



	print R_mat
	return R_mat
	# print R_mat == R_mat
	# sleep(5)


	# -- 1: First, we normalize the distribution_mat column wise to find the nodes
	# # ----- that characterize the clusters best. 

	# # print distribution_mat
	# # print "\n\n"
	# for i in range(len(distribution_mat)):
	# 	row = distribution_mat[i]
	# 	new_row = normalize_to_max(row)
	# 	distribution_mat[i] = new_row
		
		
	# # print distribution_mat
	# # sleep(5)
	# column_wise_mat = np.copy(np.transpose(distribution_mat))
	
	# # print column_wise_mat
	# # print distribution_mat
	# # print "\n\n"

	# # Note: Now the (j,i)th entry in column_wise_mat is the percent of random walkers who
	# # ----- end up in absorption state j that began in transient state i. We want to rewire
	# # ----- each cluster node in such a way that a random walk starting randomly at one of the nodes
	# # ----- (with %chance of starting at i given by column_wise_mat[clusternode][i]) is most likely
	# # ----- to end up at the given cluster node. This is given by rewiring R in such a way that 
	# # ----- (column_wise_mat[j] * B)[j] is maximized. Now we know column_wise_mat[j], B is simply
	# # ----- N * R and we know N, so we have one variable to solve for. And we know that order of 
	# # ----- operations doesn't matter, so...

	# # -- 2: Penalize each element by it's association to the cluster. The first normalized value is
	# # ----- something like "distance", though it doesn't seem it... perhaps look at expected
	# # ----- values of steps? It's being weighted here by "membership", which does look correct. 
	# # print distribution_mat 
	# # print "\n\n\n"
	# # if fuzzy:
	# # 	for j in range(r_size):
	# # 		for i in range(t_size):
	# # 			column_wise_mat[j][i] *= distribution_mat[i][j]
	
	# # print distribution_mat

	# # for row in column_wise_mat:
	# 	# normalize(row)

	

	# dist_times_N_mat = column_wise_mat.dot(N_mat)
	# # transposed = np.transpose(dist_times_N_mat)
	# # for index in range(t_size):
	# argmaxes = np.argmax(dist_times_N_mat, axis=0)
	# for index in range(t_size):
	# # for argmax in argmaxes:
	# 	R_mat[index] = [0 for i in range(r_size)]
	# 	R_mat[index][argmaxes[index]] = cluster_prob

	# # print argmaxes

def rewire_clusters_M(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob): 
	# print distribution_mat

	# -- 1: First, we create the matrix V such that V_ij = I(a_j = c_i).
	# V_mat = np.zeros((r_size,t_size),np.float32)
	# for i in range(t_size):
	# 	V_mat[np.argmax(distribution_mat[i])][i] = 1
	# print V_mat

	# sleep(10)






	# -- 1: First, we create the matrix V such that V_ij = 
	print R_mat
	R_mat = np.copy(distribution_mat)

	# -- Penalization
	for row in R_mat:
		for i in range(len(row)):
			row[i] = row[i]**3

	for row in R_mat:
		normalize(row, normalize_to=cluster_prob)

	print R_mat
	# sleep(5)
	return R_mat
	V_mat = np.copy(distribution_mat)
	# R_mat = V_mat
	







	# -- And normalize

	for row in V_mat:
		normalize(row)#, normalize_to=cluster_prob)

	# -- 2: Now, we want to connect each node j to argmax_i(VN)Transpose_ji
	# print V_mat
	# print R_mat
	# sleep(10)
	VNTrans = np.transpose(V_mat.dot(N_mat))
	for j in range(t_size):
		argmax = np.argmax(VNTrans[j])
		R_mat[j] = [0 for i in range(r_size)]
		R_mat[j][argmax] = cluster_prob



	print R_mat
	# print R_mat == R_mat
	# sleep(5)


	# -- 1: First, we normalize the distribution_mat column wise to find the nodes
	# # ----- that characterize the clusters best. 

	# # print distribution_mat
	# # print "\n\n"
	# for i in range(len(distribution_mat)):
	# 	row = distribution_mat[i]
	# 	new_row = normalize_to_max(row)
	# 	distribution_mat[i] = new_row
		
		
	# # print distribution_mat
	# # sleep(5)
	# column_wise_mat = np.copy(np.transpose(distribution_mat))
	
	# # print column_wise_mat
	# # print distribution_mat
	# # print "\n\n"

	# # Note: Now the (j,i)th entry in column_wise_mat is the percent of random walkers who
	# # ----- end up in absorption state j that began in transient state i. We want to rewire
	# # ----- each cluster node in such a way that a random walk starting randomly at one of the nodes
	# # ----- (with %chance of starting at i given by column_wise_mat[clusternode][i]) is most likely
	# # ----- to end up at the given cluster node. This is given by rewiring R in such a way that 
	# # ----- (column_wise_mat[j] * B)[j] is maximized. Now we know column_wise_mat[j], B is simply
	# # ----- N * R and we know N, so we have one variable to solve for. And we know that order of 
	# # ----- operations doesn't matter, so...

	# # -- 2: Penalize each element by it's association to the cluster. The first normalized value is
	# # ----- something like "distance", though it doesn't seem it... perhaps look at expected
	# # ----- values of steps? It's being weighted here by "membership", which does look correct. 
	# # print distribution_mat 
	# # print "\n\n\n"
	# # if fuzzy:
	# # 	for j in range(r_size):
	# # 		for i in range(t_size):
	# # 			column_wise_mat[j][i] *= distribution_mat[i][j]
	
	# # print distribution_mat

	# # for row in column_wise_mat:
	# 	# normalize(row)

	

	# dist_times_N_mat = column_wise_mat.dot(N_mat)
	# # transposed = np.transpose(dist_times_N_mat)
	# # for index in range(t_size):
	# argmaxes = np.argmax(dist_times_N_mat, axis=0)
	# for index in range(t_size):
	# # for argmax in argmaxes:
	# 	R_mat[index] = [0 for i in range(r_size)]
	# 	R_mat[index][argmaxes[index]] = cluster_prob

	# # print argmaxes

def evaluate(distribution_mat, R_mat, t_size):
	total_sum = 0
	for i in range(t_size):
		cluster_i = np.argmax(R_mat[i])
		total_sum += distribution_mat[i][cluster_i]
	return total_sum


def AMC_cluster(original_array, seeds, cluster_prob=.3, epsilon=.001, K=None):
	phase("Initializing")

	Q_mat, N_mat, t_size = array_initialize(original_array, cluster_prob)
	seeds, R_mat, r_size = seed_initialize(N_mat,t_size, cluster_prob, seeds)

	output('First E step')
	distribution_mat = find_distributions_E(N_mat, R_mat)
	# return distribution_mat
	curr_eval = evaluate(distribution_mat, R_mat, t_size)
	phase ("Iteration")
	counter = 0
	while(True):
		output('M Step')
		test_R = np.copy(R_mat)

		R_mat = rewire_clusters_M(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob)
		print R_mat
		print R_mat == test_R
		output('E Step')
		new_distribution_mat = find_distributions_E(N_mat, R_mat)
		print new_distribution_mat == distribution_mat
		new_eval = evaluate(new_distribution_mat, R_mat, t_size)
		if new_eval - curr_eval < epsilon:
			print new_eval - curr_eval
			break
		else:
			print "diff = %f" % (new_eval - curr_eval)
			curr_eval = new_eval
		distribution_mat = new_distribution_mat
		counter += 1


	print "COUNTER AT:%d" % counter
	print R_mat
	return distribution_mat



