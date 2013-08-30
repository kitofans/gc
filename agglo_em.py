import io, os, sys
import math
from time import sleep
import numpy as np
from clustering import cluster as cluster_distributions_E

def normalize(vector, normalize_to=1):
	total = np.sum(vector)
	for index in range(len(vector)):
		vector[index] = (vector[index]/float(total))*normalize_to


def check_end(index, N_mat, seeds):
	for seed in seeds:
		row = N_mat[seed]
		if row[index] > .5*np.average(row):
			return True
	return False

def find_seeds(N_mat):
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
			if check_end(index_of_min, N_mat, seeds):
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



def array_initialize(array,cluster_prob):
	t = len(array)

	print "t size: %d" % t
	# -- 1: Create Q, the t by t probability matrix that represents transitioning from
	# ----- a transient state to a transient state. 
	print "> normalizing Q..."
	Q = np.array(array,copy=True) # Copy to leave the original array intact
	for index in range(t):
		normalize(Q[index], 1-cluster_prob)
	
	print "> finding N..."
	N = find_N(Q, t)

	return Q, N, t

def seed_initialize(N,t, cluster_prob):
	seeds = find_seeds(N)
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
	
	return seeds, R, r


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

def rewire_clusters_M(distribution_mat, r_size, N_mat, t_size, R_mat, cluster_prob): # TO LOOK FOR: DOES THE IMPLEMENTATION TRY TO MAKE ALL CLUSTERS SAME SIZE (I.E IT SHOULD TAKE INTO ACCOUNT THAT SOME WILL BE SMALLER)
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


def merge_clusters(seed_index, other_seed_index, R_mat, seeds, cluster_prob):
	R_mat[seeds[other_seed_index]][other_seed_index] = 0
	R_mat[seeds[other_seed_index]][seed_index] = cluster_prob




def coalesce_seeds_M(B_mat, r_size, t_size, seeds, cluster_prob, R_mat):
	distances = []
	D_mat = 50*np.identity(r_size)
	B_transpose = np.copy(np.transpose(B_mat))
	for seed_index in range(r_size):
		for other_seed_index in range(seed_index + 1, r_size):
			euclidean_distance = np.linalg.norm(B_transpose[seed_index] - B_transpose[other_seed_index])
			print "Seed indexes: %d, %d. Euclidean distance: %f" % (seed_index, other_seed_index, euclidean_distance)
			D_mat[seed_index][other_seed_index] = euclidean_distance
			D_mat[other_seed_index][seed_index] = euclidean_distance
			distances.append(euclidean_distance)

	average = np.average(distances)
	stdev = np.std(distances)

	to_remove =[] #INDEXES INTO SEEDS
	num_coalesced = 0
	while(True):
		min_tuple = np.unravel_index(np.argmin(D_mat),D_mat.shape)
		if D_mat[min_tuple[0]][min_tuple[1]] < (average - 2*stdev):
			print "MERGING: %d and %d!" % min_tuple
			num_coalesced += 1
			merge_clusters(min_tuple[0],min_tuple[1],R_mat,seeds,cluster_prob)
			to_remove.append(min_tuple[1])
			#D_mat[min_tuple[0]] = [100 for x in range(r_size)]#np.delete(D_mat,max(min_tuple),0)
			D_mat[min_tuple[1]] = [100 for x in range(r_size)]# = np.delete(D_mat,max(min_tuple),1)
			new_D_mat = np.transpose(D_mat)
			#new_D_mat[min_tuple[0]] = [100 for x in range(r_size)]#np.delete(D_mat,max(min_tuple),0)
			new_D_mat[min_tuple[1]] = [100 for x in range(r_size)]# = np.delete(D_mat,max(min_tuple),1)
			D_mat = np.transpose(new_D_mat)
			# D_mat = np.delete(D_mat,min(min_tuple),0)
			# D_mat = np.delete(D_mat,min(min_tuple),1)
		else:
			break

	# for seed_index in range(r_size):# only merge the best merge fit!!!!
	# 	row = D_mat[seed_index]
	# 	index_of_min = np.argmin(row)
	# 	if row[index_of_min] < (average - stdev):
	# 		print "MERGING: %d and %d!" % (seed_index, index_of_min)
	# 		num_coalesced += 1
	# 		merge_clusters(seed_index, index_of_min, R_mat, seeds, cluster_prob)
	# 		to_remove.append(index_of_min)
		# for other_seed_index in range(seed_index + 1, r_size):
		# 	if D_mat[seed_index][other_seed_index] < (average - stdev):
		# 		print "MERGING: %d and %d!" % (seed_index, other_seed_index)
		# 		num_coalesced += 1
		# 		merge_clusters(seed_index, other_seed_index, R_mat, seeds, cluster_prob)
		# 		to_remove.append(other_seed_index)

	sorted_remove = sorted(to_remove, key=lambda number: -1*number)
	print "BEFORE:"
	print R_mat
	print "\n\n"
	print sorted_remove
	for remove in sorted_remove:
		seeds.pop(remove)
		R_mat = np.delete(R_mat, remove, 1)
	print "AFTER:"
	print R_mat

	r_size = len(seeds)
	for index in range(t_size):
		if index not in seeds:
			R_mat[index] = [cluster_prob/r_size for i in range(r_size)]

	print "RETURNED:"
	print R_mat


	return R_mat, r_size, num_coalesced







def cluster(original_array, seeds=None, cluster_prob=.5, epsilon=.001, K=None):
	print "> initializing..."
	Q_mat, N_mat, t_size = array_initialize(original_array, cluster_prob)
	seeds, R_mat, r_size = seed_initialize(N_mat,t_size, cluster_prob)
	print seeds
	print "ORIG:"
	print R_mat
	
	while(True):
		B_mat = cluster_distributions_E(N_mat, R_mat, r_size, t_size, cluster_prob, epsilon)
		print "AFTER CDE:"
		print R_mat
		R_mat, r_size, num_coalesced = coalesce_seeds_M(B_mat, r_size, t_size,seeds, cluster_prob, R_mat)
		if num_coalesced == 0:
			break

		
	# print "Distributions:"
	# print distribution_mat
	return B_mat,seeds
