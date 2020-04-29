#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gensim 
from gensim.test.utils import datapath
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.linalg import norm

def sparsity_ratio(WE):
    zeros = WE.vectors[WE.vectors== 0 ]
    return len(zeros) / (len(WE.vectors) * WE.vectors.shape[1])

def sorted_by(word_embedding, word, w_sort):
    wv = np.mean(word_embedding[w_sort], axis=0)
    v = word_embedding[word]
    out = v[wv.argsort()]
    return np.reshape(out, (1,-1))

def plot_heatmap(word_embedding, dim=900, word='', w_sort = []):
    fig = plt.figure(figsize=(100,7))
    #fig.suptitle(word+' sorted by '+' '.join(w_sort), fontsize=70)
    plt.subplots_adjust(left=0, right=1, top=0.86, bottom=0.23)
    x1 = sorted_by(word_embedding, word, w_sort)
    sns.heatmap(x1.reshape((dim,1)).T, cmap='RdBu_r', center=0, xticklabels=0)
    
def Interpretability_dim(WE):
    matrix  = WE.vectors
    norm1 = np.linalg.norm(matrix, axis=1)
    norm1 [norm1 == 0] = np.finfo(float).eps
    matrix /= norm1.reshape((len(norm1),1))
    var_mat = np.var(matrix, axis = 0)
    ind = np.argsort(var_mat)[-10:][::-1]
    var_mat[ind]
    my_index = []
    for i in matrix[..., ind].T:
        my_index.append(np.argsort(i)[-10:][::-1])
    words = []
    for i in my_index:
        temp_vec = matrix[i]
        temp_list = []
        for j in temp_vec:
            temp_list.append(WE.similar_by_vector(j, topn=1)[0][0])    
        words.append(temp_list)    
    return pd.DataFrame(words)

    
def dist_pos(we, data = pd.read_csv('/home/server01/workspace/test.csv'), dim = 1000, title=''):
    cnt_noun = 0
    cnt_verb = 0
    cnt_adj = 0
    # verb dist
    
    verb_distribution = np.zeros((1,dim))
    for i in data['Verbs']:
        try:
            verb_distribution += we[i]
            cnt_verb+=1
        except:
            pass
#             print(i, 'not found')

    # nouns dist
    noun_distribution = np.zeros((1,dim))
    for i in data['Nouns']:
        try:
            noun_distribution += we[i]
            cnt_noun+=1
        except:
            pass
#             print(i, 'not found')

    # adj dist
    adjectives_distribution = np.zeros((1,dim))
    for i in data['Adjectives']:
        try:
            adjectives_distribution += we[i]
            cnt_adj+=1
        except:
            pass
#             print(i, 'not found')
    print(cnt_adj)
    noun_distribution/= cnt_noun
    verb_distribution/= cnt_verb
    adjectives_distribution/= cnt_adj
    fig = plt.figure(figsize=(30, 13))
    fig.suptitle(title, fontsize= 25)
    plt.plot(noun_distribution.ravel(), label= 'noun')
    plt.plot(verb_distribution.ravel(), label= 'verb')
    plt.plot(adjectives_distribution.ravel(), label= 'adj')
    plt.legend(fontsize = 25)
    return verb_distribution/len(verb_distribution), noun_distribution/len(noun_distribution), adjectives_distribution/len(adjectives_distribution)



    
def topkwords(word_embedding, word, rank = 1,  k = 10):
    ind = word_embedding[word].argsort()[-rank]
    top_words = normalize(word_embedding)[...,ind].argsort()[-k:][::-1]
    out = []
    for i in top_words:
        out.append(list(word_embedding.vocab.keys())[i])
    return out

def topk_dim(word_embedding, dim, k = 10):
    top_words = normalize(word_embedding)[...,dim].argsort()[-k:][::-1]
    temp = []
    for i in top_words:
        temp.append(list(word_embedding.vocab.keys())[i])
    return temp

def normalize(word_embedding):
    from scipy.linalg import norm
    mat = word_embedding.vectors
    vector_norm = norm(mat, ord=2, axis=1)
    normalized_mat = mat / vector_norm.reshape(len(word_embedding.vocab.keys()),1)
    return normalized_mat

