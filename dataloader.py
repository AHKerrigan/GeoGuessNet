import csv

#from matplotlib import transforms
from geopy.geocoders import Nominatim

from torch.utils.data import Dataset

from PIL import Image as im
import os
import torch

import cv2

import numpy as np
import glob
import random

from utilities.video_transforms import *
from utilities.volume_transforms import  *

from einops import rearrange

import json
from collections import Counter


#from transformers import ViTModel, ViTConfig

def my_transform():
	video_transform_list = [
        RandomCrop(size=336),
        Resize(size=224),
		ClipToTensor(channel_nb=3),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]
	video_transform = Compose(video_transform_list)
	return  video_transform



def get_BDD_train():

    ground_folder  = '/home/alec/Documents/BigDatasets/BDD100k_Big/Ground/train/'
    class_info = json.load(open("video_data.json"))

    #aerial_folder = '/home/alec/Documents/geolocalization2/BDD100k_Mini/Aerial/train/SanFrancisco/'
    #ground_folder  = '/home/alec/Documents/geolocalization2/BDD100k_Mini/Ground/train/SanFrancisco/'

    fnames = [] 
    classes = []

    valid = 0
    total = 0

    parent = []
    
    for vid in sorted(os.listdir(str(ground_folder))):
        if vid in class_info:
            fnames.append(ground_folder + vid)
            classes.append(class_info[vid]['classes'])
            parent.append(class_info[vid]['classes'][0])
            #print(class_info[vid]['classes'])
            valid += 1
            total += 1
        else:
            continue
        
        # For small scale testing
        #if valid >= 1000:
        #    break
        #print(class_info[vid]['classes'])
    
    print(Counter(parent))
    return fnames, classes

def get_BDD_val():

    ground_folder  = '/home/alec/Documents/BigDatasets/BDD100k_Big/Ground/val/'
    class_info = json.load(open("video_data.json"))

    #ground_folder  = '/home/alec/Documents/geolocalization2/BDD100k_Mini/Ground/val/SanFrancisco/'

    fnames = [] 
    classes = []

    valid = 0
    total = 0
    
    for vid in sorted(os.listdir(str(ground_folder))):
        if vid in class_info:
            fnames.append(ground_folder + vid)
            classes.append(class_info[vid]['classes'])
            #print(class_info[vid]['classes'])
            valid += 1
            total += 1
        else:
            continue
        
        #if valid >= 1000:
        #    break
        #print(class_info[vid]['classes'])
    print(valid, "videos")
    return fnames, classes

def read_frames(fname, one_frame=False):
    path = glob.glob(fname + '/*.jpg')
    
    vid = []
    coords = []
    for img in path:
        buffer = im.open(img).convert('RGB')
        coords.append(list(float(c) for c in (img.split("/")[-1][3:-4].split("_"))))
        vid.append(buffer)
        if one_frame:
            break
    return vid, coords

class BDDDataset(Dataset):

    def __init__(self, crop_size = 112, split='train', one_frame=False):

        np.random.seed(0)

        if split == 'train':
            fnames, self.classes = get_BDD_train()
        else:
            fnames, self.classes = get_BDD_val()
        self.one_frame = one_frame

        np.random.shuffle(fnames)
        self.data = fnames


        print("Loaded data, total vids", len(fnames))
        self.crop_size = crop_size  # 112
        self.transform = my_transform()

    def __getitem__(self, idx):

        #print(self.data[0])
        sample = self.data[idx]
        coords = []
        if not self.one_frame:
            vid, coords = read_frames(sample)
            vid = vid[:15]
            coords = coords[:15]
        else:
            vid, coords = read_frames(sample, self.one_frame)
        vid = self.transform(vid)
        return vid, torch.FloatTensor(coords), torch.LongTensor(self.classes[idx])

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dataset = BDDDataset(one_frame=True, split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False, drop_last=False)


    for i, (vid, coords, classes) in enumerate(dataloader):
        #print(classes.shape)
        continue
