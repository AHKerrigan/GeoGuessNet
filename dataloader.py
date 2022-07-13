import csv

#from matplotlib import transforms
from geopy.geocoders import Nominatim
from networks import *
from torch.utils.data import Dataset

from PIL import Image as im
import os
import torch

import pandas as pd

import numpy as np
import glob
import random

#from utilities.video_transforms import *
#from utilities.volume_transforms import  *

import torchvision.transforms as transforms 
from torchvision.utils import save_image


from einops import rearrange
import csv

import json
from collections import Counter

import matplotlib.pyplot as plt
from os.path import exists 
from tqdm import tqdm
import pickle

import torch.nn.functional as F

#from transformers import ViTModel, ViTConfig

# Need to change this to torchvision transforms 
def my_transform():
	video_transform_list = [
        RandomCrop(size=600),
        Resize(size=224),
		ClipToTensor(channel_nb=3),
		Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	]
	video_transform = Compose(video_transform_list)
	return  video_transform

def m16_transform():

    m16_transform_list = transforms.Compose([
        transforms.RandomAffine((1, 15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return m16_transform_list
def m16_val_transform():
    m16_transform_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return m16_transform_list    

def get_mp16_train(classfile="mp16_places365_mapping_h3.json", gpsfile="mp16_places365.csv", opt=None):

    class_info = json.load(open(opt.resources + classfile))
    gps_info = pd.read_csv(opt.resources + gpsfile)

    #print("The classes should have been", class_info['34/8d/9055806529.jpg'])
    base_folder = opt.mp16folder

    fnames = []
    classes = []
    scenes = []
    gps = []

    for row in tqdm(gps_info.iterrows()):
        filename = base_folder + row[1]['IMG_ID']
        if row[1]['IMG_ID'] in class_info and exists(filename):
            fnames.append(filename)
            classes.append([int(x) for x in class_info[row[1]['IMG_ID']]])
            scenes.append([row[1]['S3_Label'], row[1]['S16_Label'], row[1]['S365_Label']])        
            gps.append([float(row[1]['LAT']), float(row[1]['LON'])])
    

    return fnames, classes, scenes, gps

def get_yfcc25600_test(classfile="yfcc25600_places365.csv", opt=None):

    class_info = pd.read_csv(opt.resources + classfile)
    base_folder = opt.yfcc25600folder

    fnames = []
    classes = []

    for row in class_info.iterrows():
        filename = base_folder + row[1]['IMG_ID']
        if exists(filename):
            fnames.append(filename)
            #print(row[1]['LAT'])
            classes.append([float(row[1]['LAT']), float(row[1]['LON'])])
    
    #print(classes)
    return fnames, classes, classes, classes

def get_im2gps3k_test(classfile="im2gps3k_places365.csv", opt=None):

    class_info = pd.read_csv(opt.resources + classfile)
    base_folder = opt.im2gps3k

    fnames = []
    classes = []

    for row in class_info.iterrows():
        filename = base_folder + row[1]['IMG_ID']
        if exists(filename):
            fnames.append(filename)
            #print(row[1]['LAT'])
            classes.append([float(row[1]['LAT']), float(row[1]['LON'])])
    
    #print(classes)
    return fnames, classes, classes, classes

def get_bdd_train(classfile="BDDTrain_places365_mapping_h3.json", opt=None):

    class_info = json.load(open(opt.resources + classfile))

    #print("The classes should have been", class_info['34/8d/9055806529.jpg'])
    base_folder = opt.BDDfolder + 'train/'

    fnames = []
    classes = []

    for row in class_info:
        filename = base_folder + row

        for img in sorted(os.listdir(filename)):
            if exists(filename + "/" + img):
                fnames.append(filename + "/" + img)
                classes.append([int(x) for x in class_info[row]])
    

    return fnames, classes, classes, classes

def get_bdd_test(classfile="bdd100k_val_places365.csv", opt=None):

    class_info = pd.read_csv(opt.resources + classfile)

    #print("The classes should have been", class_info['34/8d/9055806529.jpg'])
    base_folder = opt.BDDfolder + 'val/'

    fnames = []
    classes = []

    for row in class_info.iterrows():
        filename = base_folder + row[1]['IMG_ID']

        for img in sorted(os.listdir(filename)):
            if exists(filename + "/" + img):
                fnames.append(filename + "/" + img)
                info = img[:-4].split('_')
                classes.append([float(info[1]), float(info[2])])

                # Just test on the first image of each folder
                break
    

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
    
class M16Dataset(Dataset):

    def __init__(self, crop_size = 112, split='train', opt=None):

        np.random.seed(0)
        
        self.split = split 
        if split == 'train':
            fnames, classes, scenes, gps = get_mp16_train(opt=opt)
        if split == 'train1M':
            fnames, classes = get_mp16_train(classfile="mp16_places365_1M_mapping_h3.json", opt=opt)            
        if split == 'yfcc25600':
            fnames, classes, scenes, gps = get_yfcc25600_test(opt=opt)
        if split == 'im2gps3k':
            fnames, classes, scenes, gps = get_im2gps3k_test(opt=opt)
        if split == 'trainbdd':
            fnames, classes = get_bdd_train(opt=opt)        
        if split == 'testbdd':
            fnames, classes = get_bdd_test(opt=opt)      
        
        if opt.hier_eval:
            if opt.trainset == 'train':
                maps = pickle.load(open(opt.resources+"class_map.p", "rb"))
                self.coarse2medium = maps[0]
                self.medium2fine = maps[1]

                self.medium2fine[1019] = 0
                self.medium2fine[4595] = 0
                self.medium2fine[4687] = 0
        
        temp = list(zip(fnames, classes, scenes, gps))
        np.random.shuffle(temp)
        self.fnames, self.classes, self.scenes, self.gps = zip(*temp)
        self.fnames, self.classes, self.scenes, self.gps = list(self.fnames), list(self.classes), list(self.scenes), list(self.gps)

        self.data = self.fnames

        print("Loaded data, total vids", len(fnames))
        if self.split in ['train', 'trainbdd']:
            self.transform = m16_transform()
        else:
            self.transform = m16_val_transform()

    def __getitem__(self, idx):

        #print(self.data[0])
        sample = self.data[idx]
        '''
        coords = []
        if not self.one_frame:
            vid, coords = read_frames(sample)
            vid = vid[:15]
            coords = coords[:15]
        else:
            vid, coords = read_frames(sample, self.one_frame)
        '''
        vid = im.open(sample).convert('RGB')
        vid = self.transform(vid)

        #print(self.classes[idx])
        if self.split in ['train', 'train1M', 'trainbdd'] :
            return vid, torch.Tensor(self.classes[idx]).to(torch.int64), torch.Tensor(self.scenes[idx]).to(torch.int64), torch.Tensor(self.gps[idx])
        else:
            return vid, torch.Tensor(self.gps[idx])

    def __len__(self):
        return len(self.data)



import argparse
import config
import pickle

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import pandas as pd  
from transformers import ViTFeatureExtractor, ViTModel

from train_and_eval import eval_images_weighted

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from hdbscan import HDBSCAN
if __name__ == "__main__":


    opt = config.getopt()

    coarse2medium = {}
    medium2fine = {}

    model = JustResNet()
    model.load_state_dict(torch.load("/home/alec/Documents/GeoGuessNet/weights/ResNet50+Hier+Scenes+NewData.pth"))
    #model.load_state_dict(torch.load("/home/alec/Documents/GeoGuessNet/weights/Testing OOD.pth")['state_dict'])
    
    
    #model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    _ = model.to(opt.device)
    
    #train_dataset = M16Dataset(split='train', opt=opt)
    #pickle.dump(train_dataset, open("weights/datasettemp.pkl", "wb"))

    train_dataset = pickle.load(open("weights/datasettemp.pkl", "rb"))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)

    #eval_images_weighted(train_dataloader, model, 1, 1, opt)
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    total = 0
    all_files = []
    for i ,(imgs, classes, scenes, gps, sample) in bar:
        #coarse_classes = classes[:,0].to(opt.device)
        #medium_classes = classes[:,1]
        imgs = imgs.to(opt.device)

        with torch.no_grad():
            x1, x2, x3, _ = model(imgs, evaluate=True)

        prev_ce = F.cross_entropy(x1, classes[:, 0].to(opt.device), reduce=False)

        mask = prev_ce > 7.8213
        mask = torch.nonzero(mask).squeeze(1).tolist()

        if len(mask) > 0:
            sample = np.array(sample)[mask]  
            all_files.append(sample)
        

    all_files = np.concatenate(all_files, axis=0)
    df = pd.DataFrame(all_files.T)
    df.to_csv("weights/filteredfiles.csv")






