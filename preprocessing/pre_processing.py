
#!/bin/bash



import os
import sys
import threading

from subprocess import call



num_threads = 32




def segment_text(f):
    for i in f:
        #print(i)
        os.system("./geniass/geniass "+i[0]+  " /home/server01/workspace/pmc/data/geniass/" +i[1])




import os
a= input("enter num")
tasks = []
threads = []

directory =  "/home/server01/workspace/pmc/data/text/" 
os.mkdir( "/home/server01/workspace/pmc/data/geniass/") 


for d1 in os.listdir(directory):
    if d1.endswith(".nxml"):
        print(directory + d1)
        tasks.append([directory + d1, d1])
        
for i in range(num_threads):
    each_thread = int(len(tasks)/num_threads)
    if i == num_threads-1:
        t= threading.Thread(target=segment_text,args=[tasks[each_thread*(i):]])
    else:
        t= threading.Thread(target=segment_text,args=[tasks[each_thread*(i):each_thread*(i+1)]])

    threads.append(t)
    t.start()
    
    
for t in threads:
    t.join()

