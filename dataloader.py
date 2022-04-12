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
    
    for vid in sorted(os.listdir(str(ground_folder))):
        fnames.append(ground_folder + vid)
        if vid in class_info:
            valid += 1
            total += 1
        else:
            total += 1
        #print(class_info[vid]['classes'])

    print("We have", valid / total, "percent of the videos")

    return fnames

def get_BDD_val():

    ground_folder  = '/home/alec/Documents/BigDatasets/BDD100k_Big/Ground/val/'

    #ground_folder  = '/home/alec/Documents/geolocalization2/BDD100k_Mini/Ground/val/SanFrancisco/'

    fnames = [] 
    
    for vid in sorted(os.listdir(str(ground_folder))):
        fnames.append(ground_folder + vid)

    return fnames

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
            fnames = get_BDD_train()
        else:
            fnames = get_BDD_val()
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
        return vid, torch.FloatTensor(coords)

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dataset = BDDDataset(one_frame=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False, drop_last=False)


    for i, (vid, coords) in enumerate(dataloader):
        for c in coords:
            c = c.numpy()[0]
            if c[1] < -120:
                print("This video is in New York")
            else:
                print("This video is not in New York")
