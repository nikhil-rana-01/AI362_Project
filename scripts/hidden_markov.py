import numpy as np
import pickle
import re
import copy
from numpy.random import choice
import sys



class HMM:
    def __init__(self, n_hidden_states, a=None, b=None, pi=None):

        self.n_hidden_states = n_hidden_states
        self.a = a
        self.b = b
        self.pi = pi

        if a is None:
            self.a = np.ones((n_hidden_states, n_hidden_states))

            for i in range(n_hidden_states):
                self.a[i] = np.random.dirichlet(np.ones(n_hidden_states), size=1)

        if pi is None:
            self.pi = np.random.dirichlet(np.ones(n_hidden_states), size=1)[0]
    def train(self, observations, max_iterations=100):

        print('Counting unique words... ', end='')
        sys.stdout.flush()
        uniques = self.get_uniques(observations)
        print('done')
        sys.stdout.flush()

        temp = np.ones((self.n_hidden_states, len(uniques)))

        for i in range(self.n_hidden_states):
            temp[i] = np.random.dirichlet(np.ones(len(uniques)), size=1)

        self.b = []
        for i in range(self.n_hidden_states):
            self.b.append(dict(zip(uniques, temp[i])))

        del temp

        self.old_a = np.ones((len(self.a), len(self.a[0])))
        self.old_b = np.ones((len(self.b), len(self.b[0])))
        self.old_pi = np.ones(len(self.pi))

        current_iteration = 0

        print('Training model with max_iterations={}...'.format(max_iterations))

        while current_iteration < max_iterations and not self.converged():
            current_iteration += 1
            print('Iteration', current_iteration)

            self.xx = []  
            self.gg = []  

            for observation in observations:
                sentence = re.split('\s+', observation)
                self.forward(sentence)
                self.backward(sentence)
                self.xi(sentence)
                self.xx.append(self.x)
                self.gamma(sentence)
                self.gg.append(self.g)

            
            self.save()

            
            self.update(observations)

        print('Training complete, ', end='')
        if current_iteration == max_iterations:
            print('maximum iterations reached')
        else:
            print('convergence reached')

        
        hmm_parameters = (self.a, self.b, self.pi)
        pickle.dump(hmm_parameters, open('data/pickle/hmm_parameters.p', 'wb'))
    def get_uniques(self, observations):

        uniques = []

        
        for sentence in observations:
            words = re.split('\s+', sentence)
            for word in words:
                if word not in uniques:
                    uniques.append(word)

        return uniques

   
    def converged(self):
        if not np.allclose(self.a, self.old_a):
            return False

        temp_b = []
        temp_old_b = []
        for i in range(len(self.b)):
            temp_b.append(list(self.b[i].values()))
            temp_old_b.append(list(self.old_b[i].values()))

        if not np.allclose(temp_b, temp_old_b):
            return False

        if not np.allclose(self.pi, self.old_pi):
            return False

        return True

   
    def forward(self, observations):

        
        self.alpha = np.zeros((self.n_hidden_states, len(observations)))

        
        for i in range(self.n_hidden_states):
            self.alpha[i, 0] = self.pi[i] * self.b[i][observations[0]]

        
        self.alpha[:, 0] = self.normalize(self.alpha[:, 0])

        
        for t in range(len(observations) - 1):
            for i in range(self.n_hidden_states):
                for j in range(self.n_hidden_states):
                    self.alpha[i, t + 1] += self.alpha[j, t] * self.a[j, i]
                self.alpha[i, t + 1] = self.b[i][observations[t + 1]] * self.alpha[i, t + 1]

            
            self.alpha[:, t + 1] = self.normalize(self.alpha[:, t + 1])

    
    def backward(self, observations):
        self.beta = np.zeros((self.n_hidden_states, len(observations)))
        self.beta[:, -1] = 1

        for t in range(len(observations) - 2, -1, -1):
            for i in range(self.n_hidden_states):
                for j in range(self.n_hidden_states):
                    self.beta[i, t] += self.beta[j, t + 1] * self.a[i, j] * self.b[j][observations[t + 1]]

    
    def gamma(self, observations):

        self.g = np.zeros((self.n_hidden_states, len(observations)))

        for t in range(len(observations)):
            d = 0
            for i in range(self.n_hidden_states):
                self.g[i, t] = self.alpha[i, t] * self.beta[i, t]
                d += self.g[i, t]

            
            if d == 0:
                self.g[:, t] = np.zeros(self.g[:, t].shape)
            else:
                self.g[:, t] = self.g[:, t] / d


    def xi(self, observations):

        self.x = np.zeros((self.n_hidden_states, self.n_hidden_states, len(observations)))

        for t in range(len(observations) - 1):
            d = 0
            for i in range(self.n_hidden_states):
                for j in range(self.n_hidden_states):
                    self.x[i, j, t] = self.alpha[i, t] * self.a[i, j] * self.beta[j, t + 1] * self.b[j][
                        observations[t + 1]]
                    d += self.x[i, j, t]

            
            if d == 0:
                self.x[:, :, t] = np.zeros(self.x[:, :, t].shape)
            else:
                self.x[:, :, t] = self.x[:, :, t] / d

    def save(self):
        self.old_a = copy.deepcopy(self.a)
        self.old_b = copy.deepcopy(self.b)
        self.old_pi = copy.deepcopy(self.pi)
    def update(self, observations):

        
        for i in range(self.n_hidden_states):
            temp = 0
            for t in range(len(observations)):
                temp += self.gg[t][i][0]
            self.pi[i] = temp
        self.pi = self.normalize(self.pi)

        for i in range(self.n_hidden_states):
            d = 0
            for t in range(len(observations)):
                d += np.sum(self.gg[t][i][:-1])

            for j in range(self.n_hidden_states):
                n = 0
                for t in range(len(observations)):
                    n += np.sum(self.xx[t][i][j][:-1])
                self.a[i][j] = n / d


        for i in range(self.n_hidden_states):
            temp = dict.fromkeys(self.b[0], 0)
            d = 0
            for t1 in range(len(observations)):
                current_sentence = re.split('\s+', observations[t1])
                d += np.sum(self.gg[t1][i])
                for t2 in range(len(current_sentence)):
                    current_word = current_sentence[t2]
                    temp.update({current_word: temp[current_word] + self.gg[t1][i][t2]})

            for key in temp.keys():
                temp.update({key: temp[key] / d})

            self.b[i].update(temp)

    def normalize(self, dist):
        # avoid division by zero
        if np.count_nonzero(dist) == 0:
            return dist
        x = 1 / np.sum(dist)
        return [x * p for p in dist]

    def generate(self, n):
        if self.a is None or self.b is None or self.pi is None:
            print('You must train the model before you can generate text')
            return

        if n < 1:
            print('You must enter a number greater than 0')
            return

        state = choice(range(self.n_hidden_states), p=self.pi)

        s = ''

        for i in range(n):
            s += choice(list(self.b[state].keys()), p=list(self.b[state].values())) + ' '
            state = choice(range(self.n_hidden_states), p=self.a[state])

        print(s)

    def predict(self, sequence, n):
        if self.a is None or self.b is None or self.pi is None:
            print('You must train the model before you can generate text')
            return

        if n < 1:
            print('You must enter a number greater than 0')
            return

        s = re.split('\s+', sequence)

        t1 = np.zeros((self.n_hidden_states, len(s)))
        t2 = np.zeros((self.n_hidden_states, len(s)))

        for i in range(self.n_hidden_states):
            t1[i, 0] = self.pi[i] * self.safe_emission(i, s[0])
            t2[i, 0] = 0

        for i in range(1, len(s)):
            for j in range(self.n_hidden_states):
                t = t1[:, i - 1] * self.a[:][j] * self.safe_emission(j, s[i])
                t1[j][i] = np.max(t)
                t2[j][i] = np.argmax(t)

        z = np.zeros(len(s))
        x = np.zeros(len(s))

        z[-1] = np.argmax(t1[:, -1])
        x[-1] = int(z[-1])

        for i in range(len(s) - 1, 0, -1):
            z[i - 1] = t2[int(z[i]), i]
            x[i - 1] = z[i - 1]
        state = x[-1]

        output = ''

        for i in range(n):
            state = choice(range(self.n_hidden_states), p=self.a[int(state)])
            output += choice(list(self.b[int(state)].keys()), p=list(self.b[int(state)].values())) + ' '

        print(output)
    def safe_emission(self, state, word):
        if word in self.b[state].keys():
            return self.b[state][word]

        return 0
