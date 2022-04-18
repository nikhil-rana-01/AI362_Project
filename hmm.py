import numpy as np
import pickle
import re
import copy
from numpy.random import choice
import sys

# resources:
# https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
# https://en.wikipedia.org/wiki/Viterbi_algorithm

class HMM:

	# description: 	constructor
	#
	# param: 		n_hidden_states (int), number of hidden states
	# param: 		a (2d array of floats), transition probabilities
	# param: 		b (dict), emission probabilities
	# param: 		pi (array of floats), initial probabilities
	def __init__(self, n_hidden_states, a=None, b=None, pi=None):
		
		self.n_hidden_states = n_hidden_states
		self.a = a
		self.b = b
		self.pi = pi

		# assign initial probabilities randomly if none are given
		if a is None:
			self.a = np.ones((n_hidden_states, n_hidden_states))

			for i in range(n_hidden_states):
				self.a[i] = np.random.dirichlet(np.ones(n_hidden_states), size=1)

		if pi is None:
			self.pi = np.random.dirichlet(np.ones(n_hidden_states), size=1)[0]