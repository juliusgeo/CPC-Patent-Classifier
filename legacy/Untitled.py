#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import gensim, logging
import csv, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


# In[ ]:


wv = gensim.models.KeyedVectors.load("patents.wv", mmap='r')
train_arr = np.empty((100,))
train_labels = np.array([])
ret = []
ret_labels = []
def get_sentence_vector(sentence):
    ret = []
    for word in sentence:
        try:
            ret.append(wv[porter.stem(word)])
        except:
            pass
    return np.mean(ret, axis=0)
with open('../test_dataset.csv') as csvfile:
	porter = PorterStemmer()
	for row in csv.reader(csvfile):
		try:
			#wv.most_similar(porter.stem(i), topn=1)[0][0]
			mean_arr = get_sentence_vector(word_tokenize(row[1]))
			ret.append(mean_arr)
			ret_labels.append(row[0])
		except:
			print("didn't work")
			pass
print(np.vstack(ret).shape)
try:
	np.save("test_arr.npy", np.vstack(ret))
	np.save("test_arr_labels.npy", np.vstack(ret_labels))
except:
    pass


# In[3]:


training_arr = np.load("training_arr.npy")
print(training_arr.shape)
training_arr_labels = np.load("training_arr_labels.npy")
print(training_arr_labels.shape)


# Section A
#     Class 01
#         Subclass B
#             Group 33
#                 Main group 00

# In[ ]:


test_arr = np.load("test_arr.npy")
print(test_arr.shape)
test_arr_labels = np.load("test_arr_labels.npy")

def convert_labels_to_vector(label):
    sections = set()
    classes = set()
    subclasses = set()
    groups = set()
    main_groups = set()
    for z in label[0].split(','):   
        cur = z.strip().strip('\'\"')
        sections.add(str(cur[0]))
        classes.add(cur[1:3])
        subclasses.add(str(cur[3]))
        groups.add(cur.split(" ")[1])
    return [sections, classes,
    subclasses,
    groups]

#test_arr_labels = np.array([convert_labels_to_vector(i) for i in test_arr_labels])
#training_arr_labels = np.array([convert_labels_to_vector(i) for i in training_arr_labels])


# In[ ]:


import random
def convert_label_to_vector(label):
    first_half, second_half = label.split(" ")
    return [ord(i) for i in first_half]+[sum([ord(i) for i in second_half])]
all_labels = set([tuple(i) for i in np.load("label_pop.npy")])
training_dataset = []
n = 0
zed = len(training_arr_labels)
for label, vector in zip(training_arr_labels, training_arr):
    label_vectors = [convert_label_to_vector(z.strip().strip('\'\"')) for z in label[0].split(',')]
    non_true_vectors = []
    while len(non_true_vectors) <= len(label_vectors):
        cur = random.sample(all_labels, 1)
        if cur not in label_vectors:
            non_true_vectors.append(cur)
    labels = [True for _ in label_vectors]+[False for _ in non_true_vectors]
    left = np.vstack((np.vstack(label_vectors), np.vstack(non_true_vectors)))
    right = np.vstack([vector]*len(label_vectors+non_true_vectors))
    for l, r, label in zip(left, right, labels):
        training_dataset.append([label, np.hstack((l, r))])
    print(float(n)/zed)
    n=n+1

print(training_dataset)        


# In[ ]:


training_dataset = np.array(training_dataset, dtype=object)


# In[ ]:


training_dataset = np.load("training_dataset.npy")


# In[ ]:






# In[ ]:





# In[ ]:


def data_generator():
    l = len(training_vectors)
    n = 0
    while n < l:
        yield (training_vectors[n].reshape((1, 1005)), [training_labels[n]])
        n = n+1


# In[4]:


all_labels = set([tuple(i) for i in np.load("label_pop.npy")])


# In[146]:


import random
def convert_label_to_vector(label):
    first_half, second_half = label.split(" ")
    return [ord(i) for i in first_half]+[sum([ord(i) for i in second_half])]

def data_generator():
    n = 0
    for label, vector in zip(training_arr_labels, training_arr):
        label_vectors = [convert_label_to_vector(z.strip().strip('\'\"')) for z in label[0].split(',')]
        non_true_vectors = []
        while len(non_true_vectors) <= len(label_vectors):
            cur = random.sample(all_labels, 1)
            non_true_vectors.append(cur)
        labels = [1 for _ in label_vectors]+[0 for _ in non_true_vectors]
        left = np.vstack((np.vstack(label_vectors), np.vstack(non_true_vectors)))
        right = np.vstack([vector]*len(label_vectors+non_true_vectors))
        for l, r, label in zip(left, right, labels):
            ret = (np.hstack((l, r)).reshape((1, 1005)), [label])
            yield ret
def data_generator_test():
    n = 0
    for label, vector in zip(testing_arr_labels, testing_arr):
        label_vectors = [convert_label_to_vector(z.strip().strip('\'\"')) for z in label[0].split(',')]
        non_true_vectors = []
        while len(non_true_vectors) <= len(label_vectors):
            cur = random.sample(all_labels, 1)
            non_true_vectors.append(cur)
        labels = [1 for _ in label_vectors]+[0 for _ in non_true_vectors]
        left = np.vstack((np.vstack(label_vectors), np.vstack(non_true_vectors)))
        right = np.vstack([vector]*len(label_vectors+non_true_vectors))
        for l, r, label in zip(left, right, labels):
            ret = (np.hstack((l, r)).reshape((1, 1005)), [label])
            yield ret


# In[154]:


import tensorflow as tf
import os
dataset = tf.data.Dataset.from_generator(data_generator, (tf.int64, tf.int64), (tf.TensorShape([1, 1005]), tf.TensorShape([1])))
test_dataset = tf.data.Dataset.from_generator(data_generator_test, (tf.int64, tf.int64), (tf.TensorShape([1, 1005]), tf.TensorShape([1])))
test_dataset = test_dataset.batch(160, drop_remainder=True).repeat()
train_dataset = dataset.batch(160, drop_remainder=True).repeat()
print(train_dataset)
print(test_dataset)
train_dataset.cache(filename='cached_dataset')


# In[174]:


#train_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensors((training_vectors[:300000], training_labels[:300000])), tf.data.Dataset.from_tensors((training_vectors[300000:], training_labels[300000:]))))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=(None, 1005)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, dropout=.2)),
    tf.keras.layers.Dense(100, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.train.AdamOptimizer(.01),
              metrics=['accuracy'])


# In[ ]:


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)
history = model.fit(train_dataset, epochs=20,
                    validation_data=test_dataset, 
                    validation_steps=600,
                   steps_per_epoch=4000)

test_loss, test_acc = model.evaluate(test_dataset, steps=10)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[ ]:





# In[128]:


model.save_weights('./my_checkpoint')


# In[24]:


testing_arr = np.load("test_arr.npy")
print(testing_arr.shape)
testing_arr_labels = np.load("test_arr_labels.npy")
print(testing_arr_labels.shape)


# In[ ]:





# In[ ]:





# In[171]:


from numpy import newaxis
label_len = len(all_labels)
left = np.vstack(all_labels)
for label, vector in zip(testing_arr_labels, testing_arr):
        label_vectors = [convert_label_to_vector(z.strip().strip('\'\"')) for z in label[0].split(',')]
        right = np.vstack([vector]*label_len)
        combo = np.hstack((left, right)).reshape((label_len, 1005))
        combo = combo[newaxis, :]
        print(combo.shape)
        predictions = model.predict(combo, batch_size=10)
        print(predictions)
        break


# In[ ]:





# In[52]:


all_labels = set([tuple(i) for i in np.load("label_pop.npy")])


# In[ ]:





# In[168]:


pred = model.predict(test_dataset, steps=100)


# In[169]:


pred


# In[ ]:




