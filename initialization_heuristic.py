import numpy as np
import sys
import time




def findseeds():
	pass



def first_seeds(score_array):
	for i in range(len(score_array)):
		score_array[i][i] = 100000
		print score_array[i]
	# sys.exit(1)
	seeds = []

	index_of_min = np.argmin(score_array)
	tuple_index = np.unravel_index(index_of_min, (len(score_array),len(score_array)))
	for index in tuple_index:
		seeds.append(index)

	return seeds


# def find_next(seeds,score_array):
# 	scores = []
# 	for i in range(len(score_array)):
# 		find_score(N,)
# 		scores.append(score)

# 	return np.argmin(scores)

def find_score(N,elem,seeds,degrees):
	seeds += [elem]
	print seeds
	# time.sleep(1)
	# score = 0.0
	diffscore = 0.0
	for i in range(len(seeds)):
		seed = seeds[i]
		for other_seed in seeds[i:]:
			diffscore += float(abs(degrees[seed] - degrees[other_seed]))
	timespentscore = 0
	for i in range(len(seeds)):
		seed = seeds[i]
		for other_seed in seeds:
			if seed == other_seed:
				continue
			timespentscore += N[seed][other_seed]

	avgdegsum = 0
	for seed in seeds:
		avgdegsum += degrees[seed]
	avgdeg = avgdegsum/len(seeds)

	score = (1/avgdeg) * (diffscore) * timespentscore
	print avgdeg
	print diffscore
	print timespentscore
	print score
	# time.sleep(2)
	return score



	#we want the score over the choice of seeds, i.e min sum of differences of degree (but max degree) and min time spent from one to another.


def find_next(seeds, N, degrees):
	scores = []
	for i in range(len(N)):
		scores.append(find_score(N,i,seeds,degrees))
	return np.argmin(scores)


def find_seeds(N, original_array, K):
	score_array = np.zeros((len(N),len(N)))
	seeds =[]
	degrees = []
	for i in range(len(original_array)):
		degrees.append(np.sum(original_array[i]))


	# print score_array
	# sys.exit(1)
	for i in range(N.shape[0]):
		for j in range(N.shape[1]):
			score_array[i][j] = find_score(N,j,[i], degrees)


	# print score_array
	# sys.exit1)
	seeds = first_seeds(score_array)
	added = 0

	while len(seeds) <K:
		next = find_next(seeds,N,degrees)
		seeds.append(next)
		# added += 1

	print seeds
	# sys.exit(1)
	return seeds
