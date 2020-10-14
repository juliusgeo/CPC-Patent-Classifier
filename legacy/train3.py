import numpy as np
import gensim, logging
import csv, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd
import tensorflow as tf
import os
import pickle

embedding_dim = 300
num_words = 200

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)


import random

wv = gensim.models.KeyedVectors.load("patents-small.wv", mmap='r')
porter = PorterStemmer()

placeholder_vec = [0 for _ in range(embedding_dim)]

def get_sentence_vector(sentence):
	ret = []
	for word in sentence:
		if word.islower():
			try:
				ret.append(wv[porter.stem(word)])
			except:
				pass
	while len(ret) < num_words:
		ret.append(placeholder_vec)
	return np.vstack(ret[:num_words])

with open("class_descriptions.pickle", 'rb') as f:
    label_dict = pickle.load(f)
for i in label_dict:
	label_dict[i] = get_sentence_vector(word_tokenize(label_dict[i].lower()))

label_dict_keys = set(label_dict.keys())
def lstm_data_generator():
	current_dataframe = pd.read_csv('../dataset.csv',sep=',', header = None).to_numpy()#, chunksize=100000)
	indices = np.random.choice(300000, 100000)
	for row in current_dataframe[indices]:
		label, description=row[0], row[1]
		lstm_input_patent = get_sentence_vector(word_tokenize(description.lower()))
		label_vectors = [z.strip().strip('\'\"')[:4] for z in label.split(',')]
		label_vectors = set([i for i in label_vectors if i in label_dict_keys])
		non_true_vectors = [label_dict[i] for i in random.sample(label_dict_keys, 2*len(label_vectors)) if i not in label_vectors]
		for l in (label_dict[i] for i in label_vectors):
			yield ({'input_1':lstm_input_patent, 'input_2':l}, {'output_binary':[1]})
		for l in non_true_vectors:
			yield ({'input_1':lstm_input_patent, 'input_2':l}, {'output_binary':[0]})
	#np.save("state.npy", np.array([0]))

def lstm_data_generator_test():
	#last_state = np.load("state.npy")[0]
	current_dataframe = pd.read_csv('../test_dataset.csv',sep=',', header = None).to_numpy()#, chunksize=100000)
	indices = np.random.choice(7215, 6000)
	cur_batch = []
	for row in current_dataframe[indices]:
		label, description=row[0], row[1]
		lstm_input_patent = get_sentence_vector(word_tokenize(description.lower()))
		meaned = np.mean(lstm_input_patent, axis=0)
		label_vectors = [convert_label_to_vector(z.strip().strip('\'\"')) for z in label.split(',')]
		label_vectors = set([i for i in label_vectors if i in label_dict_keys])
		non_true_vectors = [label_dict[i] for i in random.sample(label_dict_keys, 2*len(label_vectors)) if i not in label_vectors]
		for l in (label_dict[i] for i in label_vectors):
			yield ({'input_1':lstm_input_patent, 'input_2':l}, {'output_binary':[1]})
		for l in non_true_vectors:
			yield ({'input_1':lstm_input_patent, 'input_2':l}, {'output_binary':[0]})


lstm_dataset = tf.data.Dataset.from_generator(lstm_data_generator, ({'input_1':tf.float64, 'input_2':tf.float64}, {'output_binary':tf.float64}), ({'input_1':tf.TensorShape([num_words, embedding_dim]), 'input_2':tf.TensorShape([num_words, embedding_dim])}, {'output_binary':tf.TensorShape([1])}))
lstm_dataset = lstm_dataset.batch(16, drop_remainder=True).repeat().prefetch(100)
test = tf.data.Dataset.from_generator(lstm_data_generator_test,  ({'input_1':tf.float64, 'input_2':tf.float64}, {'output_binary':tf.float64}), ({'input_1':tf.TensorShape([num_words, embedding_dim]), 'input_2':tf.TensorShape([num_words, embedding_dim])}, {'output_binary':tf.TensorShape([1])}))
test = test.batch(16, drop_remainder=True).repeat()


input_lstm = tf.keras.Input(shape=(num_words, embedding_dim), name='input_1')
input_label = tf.keras.Input(shape=(num_words, embedding_dim), name='input_2')

patent_mask = tf.keras.layers.Masking(mask_value=0., input_shape=(num_words, embedding_dim))(input_lstm)
label_mask = tf.keras.layers.Masking(mask_value=0., input_shape=(num_words, embedding_dim))(input_label)

patent_lstm = tf.keras.layers.LSTM(embedding_dim, input_shape=(num_words, embedding_dim), activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, unroll=False, use_bias=True
)(patent_mask)
label_lstm = tf.keras.layers.LSTM(embedding_dim, input_shape=(num_words, embedding_dim), activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, unroll=False, use_bias=True
)(label_mask)
dense_patent_1 = tf.keras.layers.Dense(embedding_dim, activation='relu')(patent_lstm)
dense_label_1 =  tf.keras.layers.Dense(embedding_dim, activation='relu')(label_lstm)
dense_patent_2 = tf.keras.layers.Dense(embedding_dim, activation='relu')(dense_patent_1)
dense_label_2 =  tf.keras.layers.Dense(embedding_dim, activation='relu')(dense_label_1)
dense_patent_3 = tf.keras.layers.Dense(embedding_dim, activation='relu')(dense_patent_2)
dense_label_3 =  tf.keras.layers.Dense(embedding_dim, activation='relu')(dense_label_2)
dense_patent_4 = tf.keras.layers.Dropout(.1)(dense_patent_3)
dense_label_4=  tf.keras.layers.Dropout(.1)(dense_label_3)
concat = tf.keras.layers.Concatenate(axis=1)([dense_patent_4, dense_label_4])
dense_1 = tf.keras.layers.Dense(int(1.8*embedding_dim), activation='relu')(concat)
output_binary = tf.keras.layers.Dense(1, activation="sigmoid", name='output_binary')(dense_1)


#lstm_enforce_1 = tf.keras.layers.Dense(200, activation='relu')(patent_lstm)
#lstm_enforce_2 = tf.keras.layers.Dense(1000, name='output_2')(lstm_enforce_1)
#model = tf.keras.Model(inputs={'input_1':input_lstm, 'input_2':input_label}, outputs={'output_1':output_binary, 'output_2':lstm_enforce_2})
model = tf.keras.Model(inputs={'input_1':input_lstm, 'input_2':input_label}, outputs=[output_binary])

#saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
#del model
try:
	model.load_weights('./doublelstmcheckpoint1.h5')
except:
	pass

model.summary()
opt = tf.optimizers.Adam()
model.compile(loss=['binary_crossentropy'],
			  optimizer='adam',
			  metrics=['accuracy'], experimental_run_tf_function=False)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./doublelstmcheckpoint1.h5',
												 save_weights_only=False,
												 verbose=1)

history = model.fit(lstm_dataset, epochs=200,
			   steps_per_epoch=500, callbacks=[cp_callback])