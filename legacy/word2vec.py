import gensim, logging
import csv, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
class MySentences(object):
	def __init__(self, filename):
		self.filename = filename

	def __iter__(self):
		with open(self.filename, encoding="utf8") as csvfile:
			porter = PorterStemmer()
			n = 0
			for row in csv.reader(csvfile):
				n=n+1
				print(float(n)/339420)
				yield [porter.stem(i) for i in word_tokenize(row[1])]

#sentences = MySentences('../dataset.csv') # a memory-friendly iterator
#
#model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=0, workers=8)
#print("model done")
#model.save("word2vec-small.model")
#model.wv.save("patents-small.wv")
#print("done saving model")
#model = gensim.models.Word2Vec.load("word2vec.model")


import numpy as np
wv = gensim.models.KeyedVectors.load("patents-small.wv", mmap='r')
train_arr = np.empty((100,))
train_labels = np.array([])
ret = []
ret_labels = []
def get_sentence_vector(sentence):
	ret = []
	for word in sentence:
		if word.islower():
			try:
				ret.append(wv[porter.stem(word)])
			except:
				pass
	return np.mean(np.vstack(ret), axis=0)

with open('../dataset.csv', encoding="utf8") as csvfile:
	porter = PorterStemmer()
	for row in csv.reader(csvfile):
		try:
			mean_arr = get_sentence_vector(word_tokenize(row[1]))
		except:
			continue
		ret.append(mean_arr)
		ret_labels.append(row[0])
print(np.vstack(ret).shape)
try:
	np.save("training_arr_small.npy", np.vstack(ret))
	np.save("training_arr_labels_small.npy", np.vstack(ret_labels))
except:
	import code
	code.interact(local=locals())

len(train_labels)
#import tensorflow as tf
#train_dataset = tf.data.Dataset.from_tensor_slices((train_arr, train_labels))
