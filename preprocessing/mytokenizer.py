import re
import io
import unicodedata
import threading
from nltk.tokenize import TreebankWordTokenizer
import os


num_threads = 12



def find_text(directory):
    files = []

    for d1 in os.listdir(directory):
        if d1.endswith(".nxml"):
            files.append([directory+d1, d1])
    
    return files


# In[2]:

def remove_url(inputString):
    return re.sub(r"http\S+ | www\S+", "", inputString)
    

def removePunctuation(inputString):
    no_punct = ""
    numbers = ['0','1','2','3','4','5','6','7','8','9']
    punctuations = set(string.punctuation)
    numbers += list(punctuations)
    for char in inputString:
        if char not in numbers:
            no_punct = no_punct + char.lower()
        else :
            no_punct = no_punct + ' '
    
    return no_punct




def writeListOfList2File(fname, inputList):
    # write list of list to file, one item per line
    with io.open(fname, 'a', encoding='utf-8') as f:
        for item in inputList:
            #print item
            text = " ".join(item)
            #print text
            text = str(text)
            f.write(text + "\n")


def sent2file(fn):
    t = TreebankWordTokenizer()
    with open(fn[0], 'r') as f:
        sentencelist = []
        for line in f:
            result = removePunctuation(line.strip())
            result = remove_url(result)
            result = t.tokenize(result)
            result = [re.sub(r'\`\`|\'\'','\"',word) for word in result]
            sentencelist.append(result)
        writeListOfList2File('/home/server01/workspace/pmc/data/tokenized/'+fn[1], sentencelist)




def target_f(file_list):
    for f in file_list:
        sent2file(f)


# In[ ]:


directory = '/home/server01/workspace/pmc/data/geniass/'

file_list = find_text(directory)
threads = []
for i in range(num_threads):
    each_thread = int(len(file_list)/(num_threads))
    if i == num_threads-1:
        t= threading.Thread(target=target_f,args=[file_list[each_thread*(i):]])  
    else:
        t= threading.Thread(target=target_f,args=[file_list[each_thread*(i):each_thread*(i+1)]])
    threads.append(t)
    t.start()
    
for t in threads:
    t.join()
