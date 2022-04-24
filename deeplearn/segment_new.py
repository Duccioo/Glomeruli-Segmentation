import tensorflow as tf
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cv2

#Configurazione:
frozen_model=sys.argv[1]
file_out=sys.argv[2]
out_img_path="/content/OUTPUT"
#il primo argomento dice dove prendere il modello
#il secondo argomento dice dove prendere l'immagine


def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
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


graph = load_graph(frozen_model)
print("caricamento dati in corso bip bip...")
# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/create_inputs/img_filename:0')
y = graph.get_tensor_by_name('prefix/predictions:0')

plt.rcParams["figure.figsize"] = (10, 10)
config = tf.ConfigProto(device_count={'GPU': 0})
# We launch a Session
with tf.Session(graph=graph, config=config) as sess:
    if os.path.isdir(file_out): 
        print("Fuck you martina")   
        os.chdir(file_out) #cartella dove sono le immagini
        for file in glob.glob("*.jpg"): #ciclo le immagini dentro la cartella  
            img = file
            y_out = sess.run(y, feed_dict={
                x: img
            })
            decoded_out = decode_labels(y_out.reshape(1, 512, 512, 1)).reshape(512, 512, 3)
            output_img = Image.fromarray(decoded_out)
            output_img.save(out_img_path+'/OUT_'+file)
            out_component = cv2.imread(out_img_path+'/OUT_'+file, 0)
            out_component = cv2.threshold(out_component, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
            num_labels_out, labels_out = cv2.connectedComponents(out_component)
            imshow_components(labels_out,out_img_path+'/OUT_'+file)

    elif os.path.isfile(file_out):  
        print("we è solo un'immagine? PFFFF")  
        img = file_out
        y_out = sess.run(y, feed_dict={
            x: img
        })
        decoded_out = decode_labels(y_out.reshape(1, 512, 512, 1)).reshape(512, 512, 3)
        output_img = Image.fromarray(decoded_out)
        output_img.save(out_img_path+'/OUTPUT.jpg')
        out_component = cv2.imread(out_img_path+'/OUTPUT.jpg', 0)
        out_component = cv2.threshold(out_component, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        num_labels_out, labels_out = cv2.connectedComponents(out_component)
        imshow_components(labels_out,out_img_path+'/OUTPUT.jpg')
    print("Taaa da!!")
    
