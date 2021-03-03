from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)

#############
#SETUP MODEL#
#############
import numpy as np
import gensim, logging
import csv, string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd
import tensorflow as tf
import os
import pickle

num_words_abstract = 400
num_words_claims = 500
num_words_label_description = num_words_abstract+num_words_claims
directory_prefix = "../../"
label_depth = 4
import random
#import gensim.downloader as api
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords


import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1")
#embed = hub.load("https://tfhub.dev/google/Wiki-words-500/2")

embeddings = embed(["cat is on the mat", "dog is in the fog"])
embedding_dim = embeddings.shape[1]
#print(embeddings)
def get_sentence_vector(words, num_words):
    words = word_tokenize(remove_stopwords(words))
    #words = list(filter(lambda w: len(w)>2, words))
    ret = embed(words)
    ret = tf.pad(ret, tf.constant([[0, max(0, num_words-ret.shape[0]),], [0, 0]]), "CONSTANT")
    ret = ret[:num_words]
    return ret
 
with open(directory_prefix+"/class_descriptions/class_descriptions_from_patents.pickle", 'rb') as f:
    label_dict = pickle.load(f)
label_dict = {k[:label_depth]: get_sentence_vector(val.lower(), num_words_label_description) for k, val in label_dict.items()}
#label_dict = {k[:label_depth]: get_sentence_vector(val.lower(), num_words_description) for k, val in label_dict.items()}

label_dict_keys = set(label_dict.keys())
description_shape = tf.TensorShape([num_words_abstract, embedding_dim])
claims_shape = tf.TensorShape([num_words_claims, embedding_dim])
label_shape = tf.TensorShape([num_words_label_description, embedding_dim])

layer_size = 256
layer_size2 = 1024

input_abstract = tf.keras.Input(shape=(num_words_abstract, embedding_dim), name='input_abstract')
input_claims = tf.keras.Input(shape=(num_words_claims, embedding_dim), name='input_claims')
input_label = tf.keras.Input(shape=(num_words_label_description, embedding_dim), name='input_label')

abstract_mask = tf.keras.layers.Masking(mask_value=0., input_shape=(num_words_abstract, embedding_dim))(input_abstract)
claims_mask = tf.keras.layers.Masking(mask_value=0., input_shape=(num_words_claims, embedding_dim))(input_claims)
label_mask = tf.keras.layers.Masking(mask_value=0., input_shape=(num_words_label_description, embedding_dim))(input_label)

abstractLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(abstract_mask)
claimsLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(claims_mask)
labelLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True))(label_mask)

abstractLSTM = tf.keras.layers.Dense(layer_size)(abstractLSTM)
claimsLSTM = tf.keras.layers.Dense(layer_size)(claimsLSTM)
labelLSTM = tf.keras.layers.Dense(layer_size)(labelLSTM)


abstract = tf.keras.layers.GlobalAveragePooling1D()(abstractLSTM)
claims = tf.keras.layers.GlobalAveragePooling1D()(claimsLSTM)
label = tf.keras.layers.GlobalAveragePooling1D()(labelLSTM)

concat = tf.keras.layers.Concatenate(axis=1)([abstract,claims,label])

dense = tf.keras.layers.Dense(int(layer_size2))(concat)
dense = tf.keras.layers.Dense(int(layer_size2), activation='relu')(dense)
dense = tf.keras.layers.Dense(int(layer_size2), activation='relu')(dense)
dense = tf.keras.layers.Dense(int(layer_size2), activation='relu')(dense)

output_binary = tf.keras.layers.Dense(1, activation="sigmoid", name='output_binary')(dense)
model = tf.keras.Model(inputs={'input_abstract':input_abstract, 'input_claims':input_claims,  'input_label':input_label}, outputs=[output_binary])
print(model)

#saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
#del model

CHECKPOINT_PATH = directory_prefix+"checkpoints/lstm3head_label4.h5"
model.load_weights(CHECKPOINT_PATH)
model.summary()
opt = tf.optimizers.Adam(1e-5)
model.compile(loss=['binary_crossentropy'],optimizer=opt,metrics=['accuracy'], experimental_run_tf_function=False)

label_len = len(label_dict_keys)
left = [label_dict[i] for i in label_dict_keys]
left_vectors = np.array(list(label_dict_keys))
description_shape = tf.TensorShape([num_words_abstract, embedding_dim])
claims_shape = tf.TensorShape([num_words_claims, embedding_dim])
label_shape = tf.TensorShape([num_words_label_description, embedding_dim])

@app.route('/getprediction', methods=["POST"])
@cross_origin()
def get_prediction():
    print(request.json)

    model.load_weights(CHECKPOINT_PATH)
    abstract = request.json["abstract"]
    claims = request.json["claims"]
    lstm_input_patent = get_sentence_vector(abstract.lower(), num_words_abstract)
    lstm_input_claims = get_sentence_vector(claims.lower(), num_words_claims)
    num_tests = 1
    def cur_gen():
        for i in left:
            yield ({'input_abstract':lstm_input_patent, 'input_claims':lstm_input_claims, 'input_label':i})
    dataset = tf.data.Dataset.from_generator(cur_gen, ({'input_abstract':tf.float64, 'input_claims':tf.float64, 'input_label':tf.float64}), ({'input_abstract':description_shape,'input_claims':claims_shape, 'input_label':label_shape}))
    dataset = dataset.batch(label_len).prefetch(50)
    prediction = model.predict(dataset, steps=num_tests, verbose=1).reshape(label_len)
    indices = np.flip(np.argsort(prediction, axis=0))
    print(np.flip(np.sort(prediction, axis=0))[:3])
    e = left_vectors[indices]
    return jsonify(e.tolist())

@app.route('/trainup', methods=["POST"])
@cross_origin()
def train_up():

    label, abstract, claims=request.json["label"], request.json["abstract"], request.json["claims"]
    def lstm_data_generator():
        label_vectors = [label]
        label_vectors = set([i for i in label_vectors if i in label_dict_keys])
        lstm_input_patent = get_sentence_vector(abstract.lower(), num_words_abstract)
        lstm_input_claims = get_sentence_vector(claims.lower(), num_words_claims)
        for l, k in zip((label_dict[i] for i in label_vectors), label_vectors):
            yield ({'input_abstract':lstm_input_patent, 'input_claims':lstm_input_claims, 'input_label':l}, {'output_binary':[1]})
        return
    description_shape = tf.TensorShape([num_words_abstract, embedding_dim])
    claims_shape = tf.TensorShape([num_words_claims, embedding_dim])
    label_shape = tf.TensorShape([num_words_label_description, embedding_dim])
    lstm_dataset = tf.data.Dataset.from_generator(lstm_data_generator, ({'input_abstract':tf.float64, 'input_claims':tf.float64, 'input_label':tf.float64}, {'output_binary':tf.float64}), ({'input_abstract':description_shape,'input_claims':claims_shape, 'input_label':label_shape}, {'output_binary':tf.TensorShape([1])}))
    lstm_dataset = lstm_dataset.repeat().batch(64, drop_remainder=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=directory_prefix+"checkpoints/LSTMWithoutAttention-ThreeHeadsLabelDepth4.h5",
                                                             save_weights_only=False,
                                                                                                              verbose=1)
    model.fit(lstm_dataset, epochs=5, steps_per_epoch=5, callbacks=[cp_callback])
    return jsonify({"message":"model succeeded"})

@app.route('/traindown', methods=["POST"])
@cross_origin()
def train_down():

    label, abstract, claims=request.json["label"], request.json["abstract"], request.json["claims"]
    def lstm_data_generator():
        label_vectors = [label]
        label_vectors = set([i for i in label_vectors if i in label_dict_keys])
        lstm_input_patent = get_sentence_vector(abstract.lower(), num_words_abstract)
        lstm_input_claims = get_sentence_vector(claims.lower(), num_words_claims)
        for l, k in zip((label_dict[i] for i in label_vectors), label_vectors):
            yield ({'input_abstract':lstm_input_patent, 'input_claims':lstm_input_claims, 'input_label':l}, {'output_binary':[0]})
        return
    description_shape = tf.TensorShape([num_words_abstract, embedding_dim])
    claims_shape = tf.TensorShape([num_words_claims, embedding_dim])
    label_shape = tf.TensorShape([num_words_label_description, embedding_dim])
    lstm_dataset = tf.data.Dataset.from_generator(lstm_data_generator, ({'input_abstract':tf.float64, 'input_claims':tf.float64, 'input_label':tf.float64}, {'output_binary':tf.float64}), ({'input_abstract':description_shape,'input_claims':claims_shape, 'input_label':label_shape}, {'output_binary':tf.TensorShape([1])}))
    lstm_dataset = lstm_dataset.repeat().batch(64, drop_remainder=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=directory_prefix+"checkpoints/LSTMWithoutAttention-ThreeHeadsLabelDepth4.h5",
                                                             save_weights_only=False,
                                                                                                              verbose=1)
    model.fit(lstm_dataset, epochs=5, steps_per_epoch=5, callbacks=[cp_callback])
    return jsonify({"message":"model succeeded"})


