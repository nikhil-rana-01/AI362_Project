import pickle
from hidden_markov import HMM

hmm_parameters = pickle.load(open('data/pickle/hmm_parameters.p', 'rb'))
transition_prob = hmm_parameters[0]
emission_prob = hmm_parameters[1]
initial_prob = hmm_parameters[2]
model = HMM(n_hidden_states=8, a=transition_prob, b=emission_prob, pi=initial_prob)

n = int(input('Number of words to generate: '))
model.generate(n)