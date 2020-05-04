
# coding: utf-8

# In[18]:


import os
import threading


# In[19]:


num_threads = 16


# In[20]:


def find_text(directory):
    files = []

    for d1 in os.listdir(directory):
        if d1.endswith(".nxml"):
            files.append(directory+d1)
    
    return files


# In[ ]:


def target_function(file_list, thread_num):

    for i in file_list:
        with open(i, 'r') as f1:
            with open('/home/server01/workspace/pmc/data/clean/cat{}.txt'.format(thread_num), 'a') as f2:
                for sent in f1.readlines():
                    f2.write(sent)


# In[26]:


directory = '/home/server01/workspace/pmc/data/tokenized/'

file_list = find_text(directory)

threads = []

for i in range(num_threads):
    each_thread = int(len(file_list)/(num_threads))
    if i == num_threads-1:
        t= threading.Thread(target=target_function,args=[file_list[each_thread*(i):], i])  
    else:
        t= threading.Thread(target=target_function,args=[file_list[each_thread*(i):each_thread*(i+1)], i])
    threads.append(t)
    t.start()
    
for t in threads:
    t.join()


# In[27]:


for i in range(num_threads):
    with open('/home/server01/workspace/pmc/data/clean/cat{}.txt'.format(i), 'r') as f1:
        with open('/home/server01/workspace/pmc/data/clean/cat.txt', 'a') as f2:
            for sent in f1.readlines():
                f2.write(sent)

