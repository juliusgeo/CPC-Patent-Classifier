#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gensim, logging
import csv, string


# In[ ]:


test_text = []
test_labels = []
with open('../test_dataset.csv') as csvfile:
    porter = PorterStemmer()
    for row in csv.reader(csvfile):
        test_text.append(row[1])
        test_labels.append(row[0])


# In[22]:


import pandas as pd
current_dataframe = pd.read_csv('../dataset.csv',sep=',', header = None).to_numpy()#, chunksize=100000)
tag_dict = {}
for row in current_dataframe:
    label, description=row[0], row[1]
    label_vectors = [z.strip().strip('\'\"')[:4] for z in label.split(',')]
    label_vectors = set([i for i in label_vectors])
    print(label_vectors)
    print(description)
    for label in label_vectors:
        if label in tag_dict.keys():
            continue
        else:
            tag_dict[label] = description
            break
    print(len(tag_dict))


# In[11]:


import pickle
with open("class_descriptions_from_patents.pickle", 'rb') as f:
    old_labels = pickle.load(f)

tag_dict = {k: v for k, v in tag_dict.items() if k in old_labels.keys()}


# In[14]:


print(len(tag_dict.keys()))


# In[15]:


import re

from lxml import etree
import os
from os import listdir
import string
from gensim.parsing.preprocessing import remove_stopwords
printable = set(string.printable)
from os.path import isfile, join
onlyfiles = [f for f in listdir('CPCSchemeXML201908') if isfile(join('./CPCSchemeXML201908', f))]
tag_dict = {}
lengths = []
for file in onlyfiles:
    if file==".DS_Store":
        continue
    if len(file.split('-')) < 3:
        continue
    tree = etree.parse(os.path.join('CPCSchemeXML201908', file))
    notags = etree.tostring(tree, encoding='utf8', method='text')
    cur_val = notags.decode('ascii', "ignore")
    cur_val = re.sub('[A-Z].+/[0-9]{1,4}', '', cur_val)
    cur_val = re.sub("[^a-zA-Z\s]+", "", cur_val)
    cur_val = remove_stopwords(cur_val).lower()
    if(len(file.split('-')[2].split(".")[0]) < 4):
        continue
    cur_val = set(cur_val.split())
    for i in cur_val:
        if len(i) <= 2:
                 del i
    lengths.append(len(cur_val))
    tag_dict[file.split('-')[2].split(".")[0]] = ' '.join(cur_val)
print(len(tag_dict.keys()))
print(max(lengths))


# In[17]:


print(len({k[:3]: v for k, v in tag_dict.items()}))


# In[54]:


for i in tag_dict:
    for k in tag_dict:
        if tag_dict[i] == tag_dict[k] and i != k:
            print("same")


# Section A
#     Class 01
#         Subclass B
#             Group 33
#                 Main group 00

# In[13]:


import pickle
with open("class_descriptions_from_patents.pickle", 'wb') as f:
   pickle.dump(tag_dict, f)


# In[ ]:


import re

for i in list(tag_dict.keys()):
    cur_val = tag_dict[i]
    stripped_val = 
    tag_dict[i] = stripped_val


# In[18]:


print(tag_dict["C03C"])


# In[19]:


compare = textdistance.Tanimoto()


# In[ ]:


tags = list(map(lambda x: x[:4], tag_dict.keys()))
ns = []
recalls = np.arange(0, 1.001, .01)
for label, patent in zipped:
    labels = set([i.split()[0].strip("\"") for i in label.split(',')])
    cur = np.vstack([compare.similarity(patent, label) for label in tags])
    idxes = np.flip(np.argsort(cur[:, 0]))
    e = [1 if tags[i] in labels else 0 for i in idxes]
    cur = []
    total_ones = e.count(1)
    n = 0
    ones_seen = 0
    while ones_seen < total_ones:
        if e[n] == 1:
            ones_seen = ones_seen+1
        n = n+1
        cur.append((ones_seen/float(n), ones_seen/floaxt(total_ones)))
    new_cur = []
    for r in recalls:
        max_of_larger = max([i[0] for i in cur if i[1] >= r])
        new_cur.append(max_of_larger)
    print(max(new_cur))
    ns.append(new_cur)
        


# In[ ]:


ns


# In[ ]:


ns = np.vstack(ns)
p = np.mean(ns, axis=0)

print(np.std(ns, axis=0) )


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'nbagg')
#np.save('prc_weds.npy', np.hstack((recalls, p)))
import matplotlib.pyplot as plt
plt.plot(recalls, p)
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




