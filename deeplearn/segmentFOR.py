#Funzione che dato una cartella di immagini e le relative maschere ricava vari indici (matrice di confusione,dice,jaccard)
#-opzioni:-#
#-c per salvare le versioni delle maschere e dell'output della rete a colori
#-u per saltare le immagini tutte nere
#-n per far svolgere alla rete solo un numero n finito di immagini

import tensorflow as tf
from PIL import Image
import glob, os
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import ndarray
from datetime import datetime
import sys
from collections import Counter
import argparse
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#CONFIG:
dir_log="/content/LOG/" #cartella per salvare i file di log
dir_in="/content/INPUT/MiniTest/" #cartella per prendere le maschere di test
dir_out="/content/OUTPUT/" #cartella per salvare le maschere della rete
path_model="/content/deeplearn/frozen_model.pb" #percorso dove è il modello della rete già addestrato
#vado a prendere l'ora per salvare il file di log
now = datetime.now()
date = now.strftime("%d_%m-%H_%M")

#opzioni per utilizzare solo n immagini e saltare quelle che ritornano 1.0
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_img", help="", type=int)
parser.add_argument("-u", "--no1", help="",action="store_true")
parser.add_argument("-c", "--color", help="",action="store_true")
args = parser.parse_args()


def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

label_colours = [(0,0,0), (255,255,255)]
                #0=unclassified, 1=globoli

def decode_labels(mask, num_images=1, num_classes=2):
    n, h, w, c = mask.shape
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

#------------------------------------------------------------------------------#
#indice Jaccard (Prof)
def jaccard(inputs, target):
    im1 = np.asarray(inputs).astype(np.bool)
    im2 = np.asarray(target).astype(np.bool)
    if im1.shape != im2.shape: 
          print(im1.shape)
          print(im2.shape)
          raise ValueError("Errore Dimensionamento: im1 e im2 devono avere la stessa dimensione!.")
    intersection = np.logical_and(im1,im2)
    union = np.logical_or(im1,im2)
    #print(np.divide(float(np.sum(intersection)),float(np.sum(union)))).astype(float)
    return np.divide(float(np.sum(intersection)),float(np.sum(union))).astype(float)

#variante Jaccard (usando counter)
def jaccard_Counter(inputs,target):
    a=inputs.ravel()
    b=target.ravel()
    _a = Counter(a)
    _b = Counter(b)
    c = (_a - _b) + (_b - _a)
    n = sum(c.values())
    return 1-(n/(len(a) + len(b) - n))

#variante Jaccard (usando operazioni tra vettori booleani)
def jaccard_binary(x,y):
  x = np.asarray(x, np.bool) 
  y = np.asarray(y, np.bool) 
  if np.bitwise_and(x, y).sum() == 0 and np.bitwise_or(x, y).sum() == 0:
        return 1.0
  return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

#Dice
def dice_loss(inputs, target):
    num = np.size(target,0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

#Confusion Matrix
def compute_confusion_matrix(inputs,target):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    inputs1=inputs.reshape(-1)
    target1=target.reshape(-1)
    CM = confusion_matrix(target1,inputs1)
    
    if CM.ndim==1:
      TN=CM[0][0]
    else:
      TN = CM[0][0]
      
      try:
        FP = CM[0][1]
        FP = CM[1][0]
        TP = CM[1][1]
      except:
        TP=0 
      else:
        print("")
      
     
    return(TP, FP, TN, FN)

#ACCURACY
def ACCURACY(tp,fp,tn,fn):
  n=float(tp+tn+fp+fn)
  if n==0:
    return 0
  else:
   return float((tp+tn)/(tp+tn+fp+fn))
  
#PRECISION
def PRECISION(tp,fp,tn,fn):

  if (tp+fp)==0 and (tp)==0:
    return 1

  elif (tp+fp)==0:
    return 0
  
  else:
    return float((tp)/(tp+fp))

#RECALL    
def RECALL(tp,fp,tn,fn):

  if (tp+fn)==0 and (tp)==0:
    return 1
    
  elif (tp+fn)==0:
    return 0

  else:
    return float((tp)/(tp+fn))


#funzione per creare immagine che evidenzia glomeruli con colori
def imshow_components(labels,path):
    # Map component labels to hue val

    if np.max(labels)==0:
        label_hue = np.uint8((179*labels)/1)
    else:
      label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    cv2.imwrite(path, labeled_img)
    
#----------------------MAIN---------------------------------#

graph = load_graph(path_model)
# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/create_inputs/img_filename:0')
y = graph.get_tensor_by_name('prefix/predictions:0')

plt.rcParams["figure.figsize"] = (10, 10)
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

i=0
with tf.compat.v1.Session(graph=graph, config=config) as sess:
    os.chdir(dir_in) #cartella dove sono le immagini
    print("----------------Start---------------")
    for file in glob.glob("*.jpg"): #ciclo le immagini dentro la cartella

      if args.n_img and i==args.n_img: #controllo parametro opzionale n
        break  

      in_mask_path=dir_in+file.replace(".jpg","")+'_seg.png' #path immagine di test
      target_img = Image.open(in_mask_path) #prendo l'immagine già segmentata di test
      
      img = file
      out_img_path= dir_out+"OUT_"+file #path dell'immagine che restituisce la rete
      #in_mask_path = dir_out+"IN_"+file #(opzionale) path per salvare l'immagine in Bianco e nero della maschera 
      
      
      y_out = sess.run(y, feed_dict={
        x: img  
      })
      
      in_mask= np.array(target_img) # converto in array la maschera di test
      decoded_out = decode_labels(y_out.reshape(1, 512, 512, 1)).reshape(512, 512, 3)
      output_img = Image.fromarray(decoded_out) 
      output_img.save(out_img_path)#salvo la maschera

      #mi calcolo gli indici che mi servono
      DICE=1-dice_loss(decoded_out,in_mask)
      JACCARDB=jaccard_binary(in_mask, decoded_out)
    
      #(opzionale) posso scartare le immagini che ritornano 1.0 con DICE, ovvero le immagini completamente nere, selezionando no1
      if args.no1 and JACCARDB==1.0: #controllo parametro opzionale no1
        continue

      
      #JACCARDC=jaccard_Counter(decoded_out,in_mask)
      #JACCARD=jaccard(convert_BW(target_img, in_mask_path),  output_img)
      tp, fp, tn, fn=compute_confusion_matrix(in_mask, decoded_out)
      ACCURACY1=accuracy_score(in_mask.reshape(-1),decoded_out.reshape(-1))#(tp+tn)/(tp+tn+fp+fn)
      
      #preparo le immagini per contare i glomeruli
      #immagini predette dalla rete
      out_component = cv2.imread(out_img_path, 0)
      out_component = cv2.threshold(out_component, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
      num_labels_out, labels_out = cv2.connectedComponents(out_component)
      
      #immagini di test
      in_component = cv2.imread(in_mask_path, 0)
      in_component = cv2.threshold(in_component, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
      num_labels_in, labels_in = cv2.connectedComponents(in_component)
    
      #stampo gli indici:
      print(str(i+1)+" "+file+":")
      print("Jaccard:",JACCARDB)
      print("Dice: ",DICE)
      print('Matrice di Confusione:','\n [', tn, fp,'] \n [', fn, tp,"]")
      print("Accuracy=",ACCURACY1, " Precision=", PRECISION(tp,fp,tn,fn)," Recall=",RECALL(tp,fp,tn,fn))
      print("Numero Glomeruli: rete OUT->",num_labels_out-1,"test IN->",num_labels_in-1)
      
      if args.color: #controllo parametro opzionale color se è definito allora:
        #salvo le immagini che evidenziano i glomeruli con colori diversi
        imshow_components(labels_in, in_mask_path)
        imshow_components(labels_out,out_img_path)

      #creo un stringa da salvare poi in un file di log
      string=file+" "+str(JACCARDB)+" "+str(DICE)+" "+str(num_labels_out-1) +" "+str(num_labels_in-1)+" "+str(ACCURACY1)+" "+str(PRECISION(tp,fp,tn,fn))+" "+str(RECALL(tp,fp,tn,fn))
      f=open(dir_log+"log"+date+".txt", "a+")
      f.write(string+"\n")
      print("-----------------------------")
      i=i+1
f.close()      

