
import os, fnmatch
import numpy as np
import pubmed_parser as pp
import threading

num_threads = 16

def extract_text(file_path):
    abstract = pp.parse_pubmed_xml(file_path)['abstract']
    body = pp.parse_pubmed_paragraph(file_path)
    body_text =''
    for i in body:
        body_text += i['text'] + ' '
    with open('../data/text/' + file_path[file_path.rfind('/')+ 1 :], 'w') as myfile:
        myfile.write(abstract + ' ' + body_text)

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def run(file_list):
    for i in file_list:
        extract_text(i)



directory = '/home/server01/workspace/pmc/data/scratch/'

file_list = find('*.nxml',directory)
print(file_list)
threads = []
for i in range(num_threads):
    each_thread = int(len(file_list)/(num_threads))
    if i == num_threads-1:
        t= threading.Thread(target=run, args=[file_list[each_thread*(i):]])  
    else:
        t= threading.Thread(target=run, args=[file_list[each_thread*(i):each_thread*(i+1)]])
    threads.append(t)
    t.start()
    
for t in threads:
    t.join()

# %%
