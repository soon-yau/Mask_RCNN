#!/usr/bin/env python
# coding: utf-8

# Welcome to the world where fashion meets computer vision! This is a starter kernel that applies Mask R-CNN with COCO pretrained weights to the task of [iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6).

# In[26]:


import os
import gc
import sys
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from tqdm import tqdm

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold


# In[2]:


DATA_DIR = Path('/home/ubuntu/efs/kaggle/imaterialist/')
ROOT_DIR = Path('/home/ubuntu/efs/kaggle/imaterialist/maskrcnn/logs')

# For demonstration purpose, the classification ignores attributes (only categories),
# and the image size is set to 512, which is the same as the size of submission masks
NUM_CATS = 46
IMAGE_SIZE = 512

N_FOLD = 6
# # Dowload Libraries and Pretrained Weights

# In[3]:


'''
!git clone https://www.github.com/matterport/Mask_RCNN.git
os.chdir('Mask_RCNN')

!rm -rf .git # to prevent an error when the kernel is committed
!rm -rf images assets # to prevent displaying images at the bottom of a kernel
'''


# In[4]:


print(ROOT_DIR/'Mask_RCNN')
sys.path.append("/home/ubuntu/github/Mask_RCNN/")
#sys.path.append(ROOT_DIR/'Mask_RCNN')
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[5]:




# In[27]:


#!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
#!ls -lh mask_rcnn_coco.h5

#COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
COCO_WEIGHTS_PATH =  "/home/ubuntu/efs/kaggle/imaterialist/maskrcnn/logs/fashion20190602/mask_rcnn_fashion_0000.h5"


# # Set Config

# In[7]:


segment_df = pd.read_csv(DATA_DIR/"train.csv")


# In[8]:


dataset_size = len(list(segment_df.ImageId.unique()))
train_ratio = (N_FOLD-1)/N_FOLD
train_size = int(dataset_size*train_ratio)//32*32
val_size = int(dataset_size-train_size)
print(train_size)


# Mask R-CNN has a load of hyperparameters. I only adjust some of them.

# In[9]:


class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 4
    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high
    
    BACKBONE = 'resnet101'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = train_size/(GPU_COUNT*IMAGES_PER_GPU)#1000
    VALIDATION_STEPS = val_size/(GPU_COUNT*IMAGES_PER_GPU)#200
    
config = FashionConfig()
config.display()


# # Make Datasets

# In[10]:


with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]


# In[11]:


#segment_df = pd.read_csv(DATA_DIR/"train.csv")

multilabel_percent = len(segment_df[segment_df['ClassId'].str.contains('_')])/len(segment_df)*100
print(f"Segments that have attributes: {multilabel_percent:.2f}%")


# Segments that contain attributes are only 3.46% of data, and [according to the host](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/90643#523135), 80% of images have no attribute. So, in the first step, we can only deal with categories to reduce the complexity of the task.

# In[12]:


segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]

print("Total segments: ", len(segment_df))
segment_df.head()


# Rows with the same image are grouped together because the subsequent operations perform in an image level.

# In[13]:


image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
image_df = image_df.join(size_df, on='ImageId')

print("Total images: ", len(image_df))
image_df.head()


# Here is the custom function that resizes an image.

# In[14]:


def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img


# The crucial part is to create a dataset for this task.

# In[15]:


class FashionDataset(utils.Dataset):

    def __init__(self, df):
        super().__init__(self)
        
        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("fashion", i+1, name)
        
        # Add images 
        for i, row in df.iterrows():
            self.add_image("fashion", 
                           image_id=row.name, 
                           path=str(DATA_DIR/'train'/row.name), 
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [label_names[int(x)] for x in info['labels']]
    
    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            
        return mask, np.array(labels)


# Let's visualize some random images and their masks.

# In[16]:



df_catlist = segment_df.groupby('ImageId')['CategoryId'].agg(lambda x: sorted(set(x)))

category_list = []

for i, row in df_catlist.iteritems():
    temp = sorted(set([label_descriptions['categories'][int(cat)]['supercategory'] for cat in row]))
    lowerhalf = 'legs and feet' in temp or 'lowerbody' in temp
    upperhalf = 'upperbody' in temp or 'wholebody' in temp
    label = 0 if lowerhalf and upperhalf else 1  
    category_list.append(label)

skf = RepeatedStratifiedKFold(n_splits=N_FOLD, n_repeats=10)
splitted = skf.split(image_df, category_list)

def gen_dataset():
    train_index, val_index = next(splitted)
    train_df = image_df.iloc[train_index]
    valid_df = image_df.iloc[val_index]
    
    train_dataset = FashionDataset(train_df)
    train_dataset.prepare()

    valid_dataset = FashionDataset(valid_df)
    valid_dataset.prepare()
    return train_dataset, valid_dataset


# Let's visualize class distributions of the train and validation data.

# In[18]:




# Note that any hyperparameters here, such as LR, may still not be optimal
LR = np.array([1, 1/3., np.power(1/3,2), np.power(1/3,3)])*1e-4
EPOCHS = [5, 10,15,20]

import warnings 
warnings.filterwarnings("ignore")


# This section creates a Mask R-CNN model and specifies augmentations to be used.

# In[30]:


model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])


# In[24]:


augmentation = iaa.Sequential([
    iaa.Fliplr(0.5), # only horizontal flip here
    # rotate and translation
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-40, 40)),
    # crop
    #iaa.CropAndPad(percent=(-0.1, 0.1)),
    # drop out pixel up to 10%
    #iaa.Dropout([0.01, 0.1])
])


# In[ ]:
for lr, epoch in zip(LR, EPOCHS):
    train_dataset, valid_dataset = gen_dataset()
    model.train(train_dataset, valid_dataset,
                learning_rate=lr,
                epochs=epoch,
                layers='all',
                augmentation=augmentation)


print("Training Complete")
