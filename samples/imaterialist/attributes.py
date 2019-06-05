#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import sys
import json
import glob
import random
from datetime import datetime
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import itertools
from tqdm import tqdm
import imutils


from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchsummary

from torchvision import models, transforms
import torchvision

import imgaug.augmenters as iaa
# In[2]:


IMAGE_DIR = Path('/home/ubuntu/efs/kaggle/imaterialist/train')
DATA_DIR = Path('/home/ubuntu/efs/kaggle/imaterialist/')
ROOT_DIR = Path('/home/ubuntu/efs/kaggle/imaterialist/pytorch/logs')

# For demonstration purpose, the classification ignores attributes (only categories),
# and the image size is set to 512, which is the same as the size of submission masks
NUM_CATS = 46
IMAGE_SIZE = 512


# In[3]:


'''
with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]
attributes_list = [i['name'] for i in label_descriptions['attributes']]
'''


# In[4]:


class AttribsDataset(Dataset):
    def __init__(self, df, transforms):
        self.n_attributes = 92
        '''
        # create dataset
        segment_df = pd.read_csv(DATA_DIR/"train.csv")
        idx_with_attribs = segment_df['ClassId'].str.contains('_')
        idx_without_attribs = ~idx_with_attribs

        segment_with_attribs = segment_df[idx_with_attribs]
        n_without_attribs = len(segment_with_attribs)//2
        segment_without_attribs = pd.DataFrame.sample(segment_df[idx_without_attribs], n=n_without_attribs) 
        
        self.df = pd.concat([segment_with_attribs, segment_without_attribs])
        '''
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        image = cv2.imread(os.path.join(IMAGE_DIR, row.ImageId))

        # mask out
        h, w,_ = image.shape#row.Height, row.Width
        mask = np.zeros((h*w), dtype=np.uint8)
        encodedPixels = list(map(int,row.EncodedPixels.split(' ')))
        for startPos, runLen in zip(encodedPixels[0::2], encodedPixels[1::2]):
            mask[startPos+1:startPos+1+runLen] = 1
        mask = np.transpose(np.reshape(mask,(w,h)))        
        #plt.imshow(mask)

        # crop
        bx,by,bw,bh = cv2.boundingRect(mask)
        maskedImage = image* np.repeat(np.expand_dims(mask,-1), 3, axis=2)
        cropped = maskedImage[by:by+bh, bx:bx+bw,:]

        # resize largest dim to 512
        resized_cropped = self.resize_apsect_ratio(cropped, 512)
        #plt.imshow(resized_cropped)
        #plt.show()
        class_id = int(row.ClassId.split('_')[0])
        #print("Class ID", class_id, label_names[class_id])

        attributes = row.ClassId.split('_')[1:]
        #for attrib in attributes:
        #    print(attrib, attributes_list[int(attrib)])

        labels = np.zeros((self.n_attributes), np.float32)
        if len(attributes)>0:
            labels[np.array(list(map(int,attributes)))] = 1
        
        if self.transforms:
            resized_cropped = self.transforms.augment_image(resized_cropped)

        resized_cropped = np.float32(resized_cropped.transpose((2, 0, 1)))/127.5 - 1

        return resized_cropped, np.float32(class_id), labels
    
    def resize_apsect_ratio(self, img, dim=512):
        img_h, img_w, _ = img.shape
        if img_h>img_w:
            resized = imutils.resize(img, height=dim)
            total_pad = dim-resized.shape[1]
            l_pad = total_pad//2
            r_pad = total_pad -l_pad
            resized = cv2.copyMakeBorder(resized, 0, 0, l_pad, r_pad, cv2.BORDER_CONSTANT, None, 0.0)
        else:
            resized = imutils.resize(img, width=dim)
            total_pad = dim-resized.shape[0]
            top_pad = total_pad//2
            bottom_pad = total_pad -top_pad
            resized = cv2.copyMakeBorder(resized,top_pad, bottom_pad,  0, 0, cv2.BORDER_CONSTANT, None, 0.0)
        assert resized.shape[:2]==(dim, dim)
        return resized        


# In[5]:


segment_df = pd.read_csv(DATA_DIR/"train.csv")
idx_with_attribs = segment_df['ClassId'].str.contains('_')
idx_without_attribs = ~idx_with_attribs

segment_with_attribs = segment_df[idx_with_attribs]
df = segment_with_attribs[segment_with_attribs['ClassId'].str.split('_').str[0]<="12"]

'''
n_with_attribs = len(segment_with_attribs)

# make total to be multiples of 32
n_without_attribs = int(n_with_attribs*1.5//32*32-n_with_attribs)

# exlucde cat < 13 that are not annotated
segment_without_attribs = segment_df[idx_without_attribs]
segment_without_attribs = segment_without_attribs[segment_without_attribs['ClassId'].str.split('_').str[0]>"12"]
segment_without_attribs = pd.DataFrame.sample(segment_without_attribs, n=n_without_attribs) 

df = pd.concat([segment_with_attribs, segment_without_attribs])

print("with attribs", n_with_attribs)
print("without attribs", n_without_attribs)
'''
print("total", len(df))

#assert(len(df)%32==0)

class_list = df['ClassId'].str.split('_').str[0].tolist()
print("class_list type, len", type(class_list), len(df))
# In[6]:


skf = RepeatedStratifiedKFold(n_splits=6, n_repeats=10, random_state=8888)
df_folder = skf.split(df, class_list)
#rkf = RepeatedKFold(n_splits=8, n_repeats=10, random_state=8888)
#df_folder = rkf.split(df)
'''
train_transforms = transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomAffine(degrees=40, translate=(0.1,0.1), scale=(0.9,1.1)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


val_transforms = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
'''
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5), # only horizontal flip here
    # rotate and translation
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-10, 10)),
])


def gen_dataset():
    train_index, val_index = next(df_folder)
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]
    
    train_dataset = AttribsDataset(train_df, augmentation)
    val_dataset = AttribsDataset(val_df, None)
    print("train and val size", len(train_dataset), len(val_dataset))

    return train_dataset, val_dataset


# # Model

# In[7]:


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        base_model = models.resnet101(pretrained=True)
        
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
                        nn.Linear(2048,1024),
                        nn.ReLU(),
                        nn.Linear(1024,512),
                        nn.ReLU(),
                        nn.Linear(512,92),
                        nn.Sigmoid())
                
    def forward(self, images, class_id):
        features = self.base(images)
        y = self.fc(features.view(-1,2048))
        #class_id_repeat = class_id.repeat(12,1,1,1)
        #class_id_repeat = class_id.view(-1,1, 1,1).repeat(1,12,1,1)
        #print(features.shape, class_id.shape, class_id_repeat.shape)
        #x = torch.cat((features, class_id_repeat), dim=1)
        #x = x.view((-1,2060))

        #y = self.fc(x)
        return y
    
model = Resnet().to("cuda")
model = nn.DataParallel(model)
#output = model(images.to("cuda"), class_id.to("cuda"))


# 
# # Training

# In[8]:


def Train(model, criterion, optimizer, scheduler, num_epochs=25):
    logdir = "/home/ubuntu/efs/kaggle/imaterialist/checkpoints/attribs/%s"%                    datetime.now().strftime('%Y%m%d-%H%M')
    
    if os.path.isdir(logdir) != True:        
        os.makedirs(logdir)
        print("Create directory ", logdir)

    log = SummaryWriter(logdir)
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    step = 0
    stepInEpoch = 0
    since = time.time()
    for epoch in range(num_epochs):
        if epoch%10 == 0:
            print("Getting new fold of dataset")
            train_dataset, val_dataset = gen_dataset()
            dataLoaders = {'train':DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True),
                           'val':DataLoader(val_dataset, batch_size=batch_size, num_workers=32)}

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            image_count = 0

            # Iterate over data.
            for images, class_ids, labels in dataLoaders[phase]: 
                images = images.to("cuda:0")
                class_ids = class_ids.to("cuda:0")
                labels = labels.to("cuda:0")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, class_ids)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                image_count += images.size(0)
                running_loss += loss.item() * images.size(0)

                step+=1
                
            #epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss = running_loss / image_count
            # deep copy the model
            if phase =='val':
                print('[Epoch %d] Valid loss=%.4f'%(epoch, epoch_loss))
                # log
                log.add_scalar("valid_loss", epoch_loss, step)
                    
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    #best_model_wts = copy.deepcopy(model.state_dict())
                    
                    modelName = '%s/model_step_%d.pth' % (logdir, step)
                    torch.save(model.state_dict(), modelName)
                    print("Save model ",modelName)
            else:
                    timeElapsed = time.time() - since
                    since = time.time()
                    meanLoss = running_loss/image_count
                    print('[%d] training loss=%.4f Time in %.0f m %.0f s'%                          (step, loss.item(), timeElapsed//60, timeElapsed%60))                    
                        
                    # log
                    log.add_scalar("train_loss", loss.item(), step)
                    log.add_scalar("learning_rate", scheduler.get_lr()[0], step)
        #print('{} Loss: {:.4f}'.format(
            #    phase, epoch_loss))


        #print()


    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #return model


# Optimizer
learningRate = 1e-4

nEpoch = 100
lrDecayStep = 3
lrDecayRate = 0.95
'''
for name, param in model.named_parameters():
    print(name)
    param.requires_grad = False if "module.fc." not in name
'''
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = learningRate)
expDecayOptimizer = optim.lr_scheduler.StepLR(optimizer, step_size=lrDecayStep, gamma=lrDecayRate)


# In[9]:


batch_size = 32
bestModel = Train(model, criterion, optimizer, expDecayOptimizer, nEpoch)


# In[ ]:




