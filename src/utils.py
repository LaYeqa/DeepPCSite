import numpy as np

def pretty_print(text):
	"""Prints text with a topbar and bottom bar."""
	print("===============================================")
	print(text)
	print("===============================================\n")


def remove_all(lst, chars):
	"""Removes all occurrences of the characters in chars from list."""
	for char in chars:
		while char in lst:
			lst.remove(char)
	return lst


def uniform_weight(train_labels):
	weights = []
	[weights.append(1) for i in range(len(train_labels))]
	return weights


def save_numpy(arrays, names):
	assert(len(arrays) == len(names))
	for i in range(len(arrays)):
		np.save('../model/input/%s.npy' %names[i], arrays[i])



