# functions_txt.py

# IMPORTS --------------------------------------------------
# We first import the necessary libraries like mentionned.

import numpy as np

# FUNCTIONS -------------------------------------------

def n_lines(filename):
    ''' Gives the number of lines contained in a txt file '''
    with open(filename, 'r') as reader:
        line = reader.readline()
        i=0
        while line != '':  # The EOF char is an empty string
            i=i+1
            line = reader.readline()
    return i

def save_fct_txt(X,Y,filename):
    ''' Saves two arrays X and Y in lines as : |X| |Y|'''
    n=X.shape[0]
    with open(filename, 'w') as adder:
        for i in range(n):
            string=str(X[i])+' '+str(Y[i])
            if i==0:
                adder.write(string)
            else:
                adder.write('\n'+string)

def read_state_file(filename):
    ''' Extract two arrays X and Y in lines as : |X| |Y|'''
    arrays=[]
    values=[]
    with open(filename, 'r') as reader:
        line = reader.readline()
        print(line)
        line=line.split()
        shape=(int(line[0]),int(line[1]))
        line = reader.readline()
        i=0
        while line != '':
            if i%2==0:
                values.append(line)
            else:
                arrays.append(line)
            i=i+1
            line=reader.readline()
    n=len(values)
    for i in range(n):
        values[i]=float(values[i])
        arrays[i]=string2array(arrays[i],shape)
    return values,arrays

def string2array(string,shape):
    ''' transforms a string of floats into an array'''
    string=string[1:-1]
    array=np.fromstring(string, dtype=np.float64, sep=' ')
    array=array.reshape(shape)
    return array

def array2string(array):
    ''' transforms an array into a string of floats '''
    n=len(array)
    string=' '
    for i in range(n):
        string=string+str(array[i])+' '
    return string          

def append_unique_value(filename,data):
    ''' writes a float value into a txt file '''
    n=n_lines(filename)
    with open(filename, 'a') as adder:
        if n==0:
            adder.write(str(data))
        else:
            adder.write('\n'+str(data))

def append_unique_array(filename,data):
    ''' writes an array into a txt file '''
    n=n_lines(filename)
    data=data.flatten()
    string=array2string(data)
    with open(filename, 'a') as adder:
        if n==0:
            adder.write(string)
        else:
            adder.write('\n'+string)

def load_last_array(filename,shape):
    '''Loads the last array contained into a txt file '''
    with open(filename, 'r') as reader:
        line = reader.readline()
        i=0
        while line != '':
            i=i+1
            last_line=line
            line = reader.readline()
    array=string2array(last_line,shape)
    return array

def load_last_value(filename):
    '''Loads the last float value contained into a txt file '''
    with open(filename, 'r') as reader:
        line = reader.readline()
        i=0
        while line != '':
            i=i+1
            last_line=line
            line = reader.readline()
    last_line=float(last_line)
    return last_line

def load_arrays(filename,frequency,shape):
    '''Loads the arrays contained into a txt file '''
    arrays=[]
    with open(filename, 'r') as reader:
        line = reader.readline()
        i=0
        while line != '':
            if i%frequency==0:
                arrays.append(line)
            i=i+1
            line=reader.readline()
    n=len(arrays)
    for i in range(n):
        arrays[i]=string2array(arrays[i],shape)
    return arrays

def load_values(filename,frequency):
    '''Loads the float values contained into a txt file '''
    values=[]
    with open(filename, 'r') as reader:
        line = reader.readline()
        i=0
        while line != '':
            if i%frequency==0:
                values.append(line)
            i=i+1
            line=reader.readline()
    n=len(values)
    for i in range(n):
        values[i]=float(values[i])
    return values

def clear_file(filename):
    ''' Allows us to clear every array or value in a txt file '''
    f=open(filename, 'w')
    f.close()
    
