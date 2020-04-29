import os
from nltk import word_tokenize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def loadVectors(word_embedding):
    
    data = open(word_embedding,"r").readlines()
    vectors = {}
    for row in data:
        vals = row.split()
        word = vals[0]
        vals = np.array( [float(val) for val in vals[1:]] )
        vectors[word] = vals
    embedding_size = len(vals)
    print("embedding_size = ",embedding_size)
    return vectors, embedding_size

def getFeats(sentence, vectors, embedding_size):
    
    ret = np.zeros(embedding_size)
    cnt = 0
    for word in sentence:
        if word.lower() in vectors:
            ret+=vectors[word.lower()]
            cnt+=1
    if cnt>0:
        ret/=cnt
    return ret

def label_encode(labels):
    
    le = LabelEncoder()
    return le.fit_transform(labels)

fp = pd.read_csv('./data/fp.csv')
bc = pd.read_csv('./data/bc.csv')
ch = pd.read_csv('./data/chr.csv')
all_of_them = pd.concat([fp, bc, ch])
all_of_them.drop(columns='Unnamed: 0', inplace=True)

all_of_them.reset_index(drop=True, inplace=True)

fp = '../intrinsic/word_vec/'

classifiers = ['SVM', 'GNB', 'RF', 'PA']

final = {}

for emb in os.listdir(fp):

    vectors, embedding_size = loadVectors(fp+emb)
    
    x = []
    
    for i in all_of_them.text:
        sentence = [word.lower() for word in word_tokenize(i)]
        x.append(getFeats(sentence, vectors, embedding_size))
    all_of_them['x'] = x
    
    x_pol = list(all_of_them[all_of_them['polarity']!='NOT_LABELED']['x'])
    y_pol = list(all_of_them[all_of_them['polarity']!='NOT_LABELED']['polarity'])
    
    y_pol = label_encode(y_pol)
    
    x_fa = list(all_of_them[all_of_them['factuality']!='NOT_LABELED']['x'])
    y_fa = list(all_of_them[all_of_them['factuality']!='NOT_LABELED']['factuality'])
    
    y_fa = label_encode(y_fa)
    
    x_train_pol, x_test_pol, y_train_pol, y_test_pol = train_test_split(x_pol, y_pol, test_size = 0.25)
    x_train_fa, x_test_fa, y_train_fa, y_test_fa = train_test_split(x_fa, y_fa, test_size = 0.25)
    
    
    clf = [[SVC(kernel="linear", C=0.025, class_weight='balanced'),
        SVC(kernel="linear", C=0.1, class_weight='balanced'),
        SVC(kernel="linear", C=5, class_weight='balanced'),
        SVC(kernel="linear", C=10, class_weight='balanced'),
        SVC(kernel="linear", C=50, class_weight='balanced'),
        SVC(kernel="linear", C=100, class_weight='balanced'),
        SVC(kernel="linear", C=500, class_weight='balanced'),
        SVC(kernel="linear", C=1000, class_weight='balanced'),
        SVC(kernel="linear", C=0.25, class_weight='balanced'),
        SVC(gamma=2, C=0.1, class_weight='balanced'),
        SVC(gamma=2, C=0.25, class_weight='balanced'),
        SVC(C=0.1, class_weight='balanced'),
        SVC(C=5, class_weight='balanced'),
        SVC(C=10, class_weight='balanced'),
        SVC(C=50, class_weight='balanced'),
        SVC(C=100, class_weight='balanced'),
        SVC(C=500, class_weight='balanced'),
        SVC(C=1000, class_weight='balanced'),
        SVC(class_weight='balanced')],
        [GaussianNB()],
        [RandomForestClassifier()],
        [PassiveAggressiveClassifier()]
      ]
    
    score = {}
    res = {}
    
    
    for i in range(len(clf)):
        scores = []
        for j in clf[i]:
            j.fit(x_train_pol, y_train_pol)
            scores.append(j.score(x_test_pol, y_test_pol))
        score[classifiers[i]]= max(scores)
        
    res['pol'] = score
    
    clf = [[SVC(kernel="linear", C=0.025, class_weight='balanced'),
        SVC(kernel="linear", C=0.1, class_weight='balanced'),
        SVC(kernel="linear", C=5, class_weight='balanced'),
        SVC(kernel="linear", C=10, class_weight='balanced'),
        SVC(kernel="linear", C=50, class_weight='balanced'),
        SVC(kernel="linear", C=100, class_weight='balanced'),
        SVC(kernel="linear", C=500, class_weight='balanced'),
        SVC(kernel="linear", C=1000, class_weight='balanced'),
        SVC(kernel="linear", C=0.25, class_weight='balanced'),
        SVC(gamma=2, C=0.1, class_weight='balanced'),
        SVC(gamma=2, C=0.25, class_weight='balanced'),
        SVC(C=0.1, class_weight='balanced'),
        SVC(C=5, class_weight='balanced'),
        SVC(C=10, class_weight='balanced'),
        SVC(C=50, class_weight='balanced'),
        SVC(C=100, class_weight='balanced'),
        SVC(C=500, class_weight='balanced'),
        SVC(C=1000, class_weight='balanced'),
        SVC(class_weight='balanced')],
        [GaussianNB()],
        [RandomForestClassifier()],
        [PassiveAggressiveClassifier()]
      ]
    
    score = {}
    
    for i in range(len(clf)):
        scores = []
        for j in clf[i]:
            j.fit(x_train_fa, y_train_fa)
            scores.append(j.score(x_test_fa, y_test_fa))
        score[classifiers[i]]= max(scores)
    res['fa'] = score
    
    
    final[emb] = res
    
    
    pd.DataFrame.from_dict(final).to_csv('./extrinsic_results')
