
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[5]:


results = dict()
output = dict()
tasks = os.listdir('./tasks')
word_embeddings = os.listdir('./word_vec')
tasks.remove('.DS_Store')

for i in word_embeddings:
    print('evaluating', i)
    results = dict()
    for j in tasks:
        results[j] = os.popen('python3 evaluate_wordSim.py ./word_vec/{} ./tasks/{}'.format(i, j)).read().replace('\n', '')
    print(results)
    output[i] = results


# In[6]:


pd.DataFrame(output).T.to_csv('intrinsic_results.csv')

