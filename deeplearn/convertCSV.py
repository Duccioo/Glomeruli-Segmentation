

# importing pandas library
import pandas as pd
import sys  

file=sys.argv[1]
path_out=sys.argv[2]

filename= file.split("\\")
filename=filename[-1]
print(filename)
risultati = pd.read_csv(file, delimiter = ' ')
                      
#risultati.columns = ['Nome', 'Jaccard', 'Dice',"# Glomeruli Output","# Glomeruli Input","Accuracy","Precision","Recall"]

risultati.columns = ['Nome', 'Jaccard', 'Dice',"# Glomeruli Output","# Glomeruli Input","F1 SCORE","Accuracy","Precision","Recall","TRUE Positive","TRUE Negative","FALSE Positive","FALSE Negative"]
#store dataframe into csv file
risultati.to_csv(path_out+filename.replace(".txt",".csv"),index = None)