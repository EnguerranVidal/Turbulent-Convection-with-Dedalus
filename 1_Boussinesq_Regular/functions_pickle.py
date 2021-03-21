#functions_pickle.py

# IMPORTS --------------------------------------------------
# We first import the necessary libraries like mentionned.

import numpy as np
import time
import pickle

# FUNCTIONS -------------------------------------------

def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill='#',time0=0):
    ''' Allows us to print a progress bar for the simulation '''
    percent=("{0:."+str(decimals)+"f}").format(100*(iteration/float(total)))
    filledLength=int(length*iteration // total)
    bar=fill*filledLength+'-'*(length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix,bar,percent,suffix)+" "+str(time.time()-time0)+" "+str(iteration))
    # Print New Line on Complete
    if iteration==total: 
        print()
     
def Kronecker(i,j):
    ''' Replaces the Kronecker parameter '''
    if i==j:
        return 1
    else:
        return 0

def save_unique_array(file_name,data):
    ''' Allows us to save an array or value in a .pickle file '''
    file=open(file_name,'ab')
    pickle.dump(data,file,pickle.HIGHEST_PROTOCOL)
    file.close()

def load_files(file_name,frequency=2):
    ''' Allows us to load every array or value in a .pickle file '''
    data=[]
    file=open(file_name,'rb')
    i=0
    while True:
        try:
            if i==0 or i%frequency==0:
                data.append(pickle.load(file))
            else:
                trash=pickle.load(file)
            i=i+1
        except EOFError:
            break
    return data

def clear_file(file_name):
    ''' Allows us to clear every array or value in a .pickle file '''
    file=open(file_name,'wb')
    file.close()
