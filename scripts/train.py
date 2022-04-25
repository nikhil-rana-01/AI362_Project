import pickle
import nltk
from hidden_markov import HMM
import sys
import string

print('Cleaning text... ', end='')
sys.stdout.flush()
f = open('data/text/amazon_reviews.txt')
text = ' '.join(f.read().splitlines())
f.close()

sentences = nltk.sent_tokenize(text)

lines_to_remove = []
for i in range(len(sentences)):

	sentences[i] = sentences[i].translate(str.maketrans('', '', string.punctuation)).lower()
	if sentences[i] == '':
		lines_to_remove = [i] + lines_to_remove

for i in lines_to_remove:
	del sentences[i]
print('done')
sys.stdout.flush()

model = HMM(n_hidden_states=8)
model.train(sentences)