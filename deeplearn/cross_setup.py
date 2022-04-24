import os
import sys
import numpy as np
#SETUP
n_fold=5
path=sys.argv[1] #dice dove si trova il file train.txt
path=path+"/train.txt"

dir_folder=sys.argv[2] #dice dove creare le cartelle e i vari file

#conto il numero di righe nel file
file = open(path, "r")
nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
n_righe = len(nonempty_lines)

lines_per_file = int(n_righe/n_fold)
print("Numero File: ",n_righe)
print("Grandezza Fold: ",lines_per_file)
print("Grandezza Split:",n_righe-lines_per_file)
smallfile = None
a=[]
b=[]

#creo il vettore 2 dimensioni
with open(path) as bigfile:
    for lineno, line in enumerate(bigfile):
        b.append(line)
        if (lineno+1) % lines_per_file == 0:           
           a.append(b)
           b=[]
    
for i in range(n_fold):
    #creo la cartella corrispondente se non esiste già
    folder= dir_folder+"/F"+str(i+1) #es: content + /F +1 =content/F1 
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder+"/model"):
        os.makedirs(folder+"/model")
    if not os.path.exists(folder+"/OUTPUT"):
        os.makedirs(folder+"/OUTPUT")
    if not os.path.exists(folder+"/LOG"):
        os.makedirs(folder+"/LOG")
        

    #creo il file train e il file test corrispondenti
    test_name = folder+'/test.txt'  #_{}.txt'.format(i+1)
    test = open(test_name, "w")
    train_name = folder+'/train.txt'#_{}.txt'.format(i+1)
    train = open(train_name, "w")
  
    for j in range(n_fold):
        if i==j:
            for line in range(len(a[j])):
                test.write(a[j][line])
        else:
            for line in range(len(a[j])):
                train.write(a[j][line])    
