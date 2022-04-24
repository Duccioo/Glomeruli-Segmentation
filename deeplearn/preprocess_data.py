import glob
from PIL import Image
import numpy as np
import os
import tqdm
import multiprocessing
import threading
import BW
import sys
import argparse
#--------------------------COMANDI----------------------------------------------#
parser = argparse.ArgumentParser()

#comandi per seleziona l'opzione di augmented data di 90,180 e 270 gradi
parser.add_argument("-90",'--a90_degree', action="store_true")
parser.add_argument("-180",'--a180_degree', action="store_true")
parser.add_argument("-270",'--a270_degree', action="store_true")

#comando per selezionare la cartella dove sono le immagini da preprocessare
parser.add_argument('--folder_in', dest='folder_in', type=str, default='train')

#comando per selezionare la cartella dove buttare fuori le immagini preprocessate
parser.add_argument('--folder_out', dest='folder_out', type=str, default='train')

option = parser.parse_args()
#-------------------------------------------------------------------------------------#
# used instead of proprocess_data.ipynb -> adjust dirs and path separator, here \\ for windows, replace by / for linux
# multithreading dies not work -> replaced by for loop
dest_dir = option.folder_out
dest_dir_masks = os.path.join( dest_dir, 'masks')
dest_dir_img = os.path.join( dest_dir, 'img')
data_dir= option.folder_in
palette = {(0,   0,   0) : 0 ,
         (0,  0, 255) : 0 ,
         (255,  255,  255) : 1
          }

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


images = glob.glob(data_dir+'/*.jpg')
masks = glob.glob(data_dir+'/*.png')

masks.sort()
images.sort()

assert( len(images) == len(masks))


def rotate(img, img_name, mask, mask_name, degree, postfix):
    img = img.rotate(degree)
    mask = mask.rotate(degree)

    mask_arr = np.array(mask)
    mask_conved = convert_from_color_segmentation(mask_arr)

    img.save(os.path.join(dest_dir_img, postfix + img_name))
    Image.fromarray(mask_conved).save(os.path.join(dest_dir_masks,postfix + mask_name))
    mask.save(os.path.join(data_dir, postfix + mask_name))
    img.save(os.path.join(data_dir, postfix + img_name))
    return


def process(args):
    image_src, mask_src = args
    #/content/MiniTest/1_10X13.jpg
    
    #image_name = '_'.join(image_src.split('/')[-3:])      # -1 for absolute directories!
    #mask_name = '_'.join(mask_src.split('/')[-3:])
    image_name = image_src.replace(data_dir,"")
    image_name=image_name.replace("/","")
    mask_name = mask_src.replace(data_dir,"")	
    mask_name=mask_name.replace("/","")

    img = Image.open(image_src)
    mask = Image.open(mask_src)
    

    img = img.resize((512, 512), Image.NEAREST)
    mask = mask.resize((512, 512), Image.NEAREST)
    #BWmask=BW.convert_BW(mask)

    if option.a90_degree:
        rotate(img, image_name, mask, mask_name, 90, "90_")
    if option.a180_degree:    
        rotate(img, image_name, mask, mask_name, 180, "180_")
    if option.a270_degree:    
        rotate(img, image_name, mask, mask_name, 270, "270_")

    mask_arr = np.array(mask)
    mask_conved = convert_from_color_segmentation(mask_arr)

    img.save(os.path.join(dest_dir_img, image_name))
    Image.fromarray(mask_conved).save(os.path.join(dest_dir_masks,mask_name))
    #BWmask.save(os.path.join(dest_dir_masks, mask_name))

#if __name__ ==  '__main__':
 #for i in range(len(masks)):
  #  print(str(i+1)+"/"+str(len(masks)) +": "+ images[i]+" / "+masks[i])
   # process((images[i], masks[i]))

if __name__ ==  '__main__':
    pool = multiprocessing.Pool(10)
    tasks = []
    for i in range(len(masks)):
        tasks.append((images[i], masks[i]))

    for _ in tqdm.tqdm(pool.imap_unordered(process, tasks), total=len(tasks)):
        pass
