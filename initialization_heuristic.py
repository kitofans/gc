import numpy as np
import sys
import time
from scipy import stats



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

def find_score(N,elem,old_seeds,degrees):
	seeds = old_seeds + [elem]
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
	# print avgdeg
	# print diffscore
	# print timespentscore
	# print score
	# time.sleep(2)
	return score


def find_scores_Z(N,old_seeds,degrees):
	scores = []
	diffs = []
	timespents = []
	avgdegs = []
	for i in range(len(N)):
		seeds = old_seeds + [i]
		diffscore = 0.0
		timespentscore = 0.0
		for j in range(len(seeds)):
			seed = seeds[j]
			for other_seed in seeds[j:]:
				diffscore += float(abs(degrees[seed] - degrees[other_seed]))
			for other_seed in seeds:
				if seed == other_seed:
					continue
				timespentscore += N[seed][other_seed]
		diffs.append(diffscore)
		timespents.append(timespentscore)

		avgdegsum = 0
		for seed in seeds:
			avgdegsum += degrees[seed]
		avgdeg = avgdegsum/len(seeds)
		avgdegs.append(avgdeg)

	print timespents[76]
	print timespents[58]
	diffs = stats.zscore(diffs)
	timespents = stats.zscore(timespents)
	avgdegs = stats.zscore(avgdegs)

	for i in range(len(diffs)):
		scores.append(diffs[i] + timespents[i] - avgdegs[i])
	return scores



	#we want the score over the choice of seeds, i.e min sum of differences of degree (but max degree) and min time spent from one to another.


def find_next(seeds, N, degrees):
	# ------ ZSCORE TEST ------ #
	scores = find_scores_Z(N,seeds,degrees)
	for i in range(len(scores)):
		if i in seeds:
			scores[i] = 1000000
	print scores[58]
	print scores[76]
	# sys.exit(1)
	return np.argmin(scores)
	scores = []
	for i in range(len(N)):
		if i in seeds:
			scores.append(1000000)
		scores.append(find_score(N,i,seeds,degrees))
	return np.argmin(scores)







def find_seeds(N, original_array, K):
	score_array = np.zeros((len(N),len(N)))
	seeds =[]
	degrees = []
	for i in range(len(original_array)):
		degrees.append(np.sum(original_array[i]))



	# ------ ZSCORE TESTING ------ #
	for i in range(N.shape[0]):
		# for j in range(N.shape[1]):
			score_array[i] = find_scores_Z(N,[i],degrees)



	# # print score_array
	# # sys.exit(1)
	# for i in range(N.shape[0]):
	# 	for j in range(N.shape[1]):
	# 		score_array[i][j] = find_score(N,j,[i], degrees)


	# print score_array
	# sys.exit1)
	seeds = first_seeds(score_array)
	print len(seeds)
	print seeds
	# sys.exit(1)
	added = 0

	while len(seeds) <K:
		next = find_next(seeds,N,degrees)
		# print next
		# print seeds
		seeds.append(next)
		# time.sleep(1)
		# print seeds
		# time.sleep(1)
		# added += 1

	print seeds
	# print len(seeds)
	# print K
	# sys.exit(1)
	return seeds
