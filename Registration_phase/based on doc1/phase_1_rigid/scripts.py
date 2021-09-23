import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import exposure
from skimage import img_as_float

#########################################
def Extract_bb_class(meta_data, image_name):
    
    # extract Label column
    this_image_Label = json.loads(meta_data[meta_data['External ID'] == image_name].Label.iloc[0])    
    # Bounding Box
    bbox_ax = this_image_Label['objects'][0]['bbox']          
    # Class Type
    class_type = this_image_Label['classifications'][0]['answer']['value']

    return bbox_ax, class_type

###########################################

def read_dataset(path):
    meta_data = pd.read_csv(str(path+path.split('/')[-2]+'.csv'))
    print("meta data readed.")
    dataset = []

    folders = os.listdir(path)
    folders = [folder for folder in folders if len(folder)==1]
    for folder in folders:
        folder_content = []
        image_names = os.listdir(path+folder)
        for image_name in image_names:
            file_path = path+folder+'/'+image_name
            image = cv2.imread(file_path , cv2.IMREAD_GRAYSCALE)   
            bb , class_type = Extract_bb_class(meta_data, image_name)
            folder_content.append((image , class_type , bb , file_path ))
        dataset.append(folder_content)
    print("images readed.")

    return dataset
###################################################
def Histogram_Matching(main_dataset):
    dataset = []
    for item1 , item2 in main_dataset:
        if item1[1] == 'lge':
            lge_image = item1
            cine_image = item2
        elif item2[1] == 'lge':
            lge_image = item2
            cine_image = item1
        
        matched = exposure.match_histograms(lge_image[0], cine_image[0])
        matched = matched/matched.max()
        matched *= 255
        matched = matched.astype(np.uint8)
        new_lge_image = (matched , lge_image[1] , lge_image[2] ,  lge_image[3])
        dataset.append((cine_image , new_lge_image))
    return dataset
#########################################
def crop_this(image , bbox_ax):
    cropped_image = image[bbox_ax['top']:bbox_ax['top']+bbox_ax['height'], bbox_ax['left']:bbox_ax['left']+bbox_ax['width']]
    shape = cropped_image.shape
    image_2d = cropped_image.astype(float)
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)
    return image_2d_scaled
#############################################
def crop_bb(main_dataset):
    cropped_dataset = []
    for cine_item , lge_item in main_dataset:  
        cine_image = cine_item[0]
        bbox_ax_cine =  cine_item[2]

        lge_image = lge_item[0]              
        bbox_ax_lge =  lge_item[2]

        croped_cine = crop_this(cine_image , bbox_ax_cine )
        croped_lge = crop_this(lge_image , bbox_ax_lge )
        
        #cine_destination = cine_item[3][:-4] + '_cine.png'
        #lge_destination = lge_item[3][:-4] + '_lge.png'

        #cv2.imwrite(cine_destination , croped_cine)
        #cv2.imwrite(lge_destination , croped_lge)
        new_cine_item = (croped_cine , cine_item[1] , cine_item[2], cine_item[3])
        new_lge_item = (croped_lge , lge_item[1] , lge_item[2], lge_item[3])
        cropped_dataset.append((new_cine_item , new_lge_item))
    return cropped_dataset


################################################

################################################
