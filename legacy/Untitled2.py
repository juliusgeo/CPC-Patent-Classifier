#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gensim, logging
import csv, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


# In[2]:


training_arr = np.load("training_arr.npy")
print(training_arr.shape)
training_arr_labels = np.load("training_arr_labels.npy")
print(training_arr_labels.shape)
testing_arr = np.load("test_arr.npy")
print(testing_arr.shape)
testing_arr_labels = np.load("test_arr_labels.npy")
print(testing_arr_labels.shape)


# In[57]:


import random
def convert_label_to_vector(label):
    first_half, second_half = label.split(" ")
    return [ord(i) for i in first_half]
def data_generator():
    for label, vector in zip(training_arr_labels, training_arr):
        label_vectors = [convert_label_to_vector(z.strip().strip('\'\"')) for z in label[0].split(',')]
        left = np.vstack(label_vectors)
        right = np.vstack([vector]*len(label_vectors))
        for l, r in zip(left, right):
            ret = (np.hstack((r)).reshape((1, 1000)), np.hstack(l).reshape(1, 4))
            yield ret

def data_generator_test():
    n = 0
    for label, vector in zip(testing_arr_labels, testing_arr):
        label_vectors = [convert_label_to_vector(z.strip().strip('\'\"')) for z in label[0].split(',')]
        left = np.vstack(label_vectors)
        right = np.vstack([vector]*len(label_vectors))
        for l, r in zip(left, right):
            ret = (np.hstack(r).reshape((1, 1000)), np.hstack(l).reshape(1,4))
            yield ret


# In[58]:


e = data_generator().next()
e


# In[60]:


import tensorflow as tf
import os
#tf.disable_eager_execution()
dataset = tf.data.Dataset.from_generator(data_generator, (tf.int64, tf.int64), (tf.TensorShape([1, 1000]), tf.TensorShape([1, 4])))
test_dataset = tf.data.Dataset.from_generator(data_generator_test, (tf.int64, tf.int64), (tf.TensorShape([1, 1000]), tf.TensorShape([1, 4])))
test_dataset = test_dataset.batch(128, drop_remainder=True).repeat()
train_dataset = dataset.batch(32, drop_remainder=True).repeat()
print(train_dataset)
print(test_dataset)
for x, y in train_dataset:
    print(x, y)


# In[61]:


#train_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensors((training_vectors[:300000], training_labels[:300000])), tf.data.Dataset.from_tensors((training_vectors[300000:], training_labels[300000:]))))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, input_shape=(None, 1000)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu')
])
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
model.summary()
model.compile(loss='mse',
              optimizer=tf.train.AdamOptimizer(1e-4),
              metrics=['accuracy'])


# In[62]:


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)
history = model.fit(train_dataset, epochs=1,
                    validation_data=test_dataset, 
                    validation_steps=600,
                   steps_per_epoch=4000)



# In[63]:


test_loss, test_acc = model.evaluate(test_dataset, steps=100)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[64]:


from numpy import newaxis
p = model.predict(np.vstack(testing_arr).reshape(-1, 1, 1000))



# In[69]:


for label, vector in zip(testing_arr_labels, testing_arr):
    label_vectors = [convert_label_to_vector(z.strip().strip('\'\"')) for z in label[0].split(',')]
    print(label_vectors[0])
    p = model.predict(np.vstack(vector).reshape(1, -1, 1000))
    print([chr(i) for i in p[0]])


# In[ ]:




