import time
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from colorama import Fore, Style

from einops import rearrange

import torch
import torch.nn.functional as F
import pickle

import geopy
from geopy.distance import geodesic as GD
from tqdm import tqdm

import wandb
import evaluate
import pandas as pd

# Currently training on 4 frames
def train_single_frame(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch):


    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader 

    losses = []
    running_loss = 0.0
    dataset_size = 0


    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))
    for i ,( vid, coords, classes) in bar:

        batch_size = vid.shape[0]

        labels1 = classes[:,0].unsqueeze(1).repeat(1, 15)
        labels2 = classes[:,1].unsqueeze(1).repeat(1, 15) 
        labels1 = rearrange(labels1, 'bs c -> (bs c)')
        labels2 = rearrange(labels2, 'bs c -> (bs c)')

        labels1 = torch.tensor(labels1).to(opt.device)
        labels2 = torch.tensor(labels2).to(opt.device)

        vid = rearrange(vid, 'bs ch l h w -> (bs l) ch h w ')
        vid = vid.to(opt.device)

        outs1, outs2 = model(vid)

        loss1 = criterion(outs1, labels1)
        loss2 = criterion(outs2, labels2)

        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        model.zero_grad()       
        #scheduler.step()

        losses.append(loss.item())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    print("The loss of epoch", epoch, "was ", np.mean(losses))
    #writer.add_scalar('Loss/Train', np.mean(losses), epoch)

def train_images(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch, val_dataloader=None):

    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader 

    losses = []
    running_loss = 0.0
    dataset_size = 0


    val_cycle = (len(data_iterator.dataset.data) // (opt.batch_size * 164))
    print("Outputting loss every", val_cycle, "batches")
    print("Validating every", val_cycle*5, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))

    for i ,(imgs, classes) in bar:

        batch_size = imgs.shape[0]
        labels1 = classes[:,0]
        #labels1 = rearrange(labels1, 'bs c -> (bs c)')

        labels1 = labels1.to(opt.device)


        imgs = imgs.to(opt.device)

        optimizer.zero_grad()
        outs1 = model(imgs)
        '''
        print("======================THE PREDICTIONS ================================")
        print(torch.argmax(outs1, dim=1))
        print("======================THE TRUTH       ======================================")
        print(labels1)
        '''
        torch.set_printoptions(edgeitems=30)

        loss = criterion(outs1, labels1)

        loss.backward()

        optimizer.step()     
        #scheduler.step()

        losses.append(loss.item())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
        
        if i % val_cycle == 0:
            wandb.log({"Training Loss" : loss.item()})
        if val_dataloader != None and i % (val_cycle * 5) == 0:
            eval_images(val_dataloader, model, epoch, opt)
    print("The loss of epoch", epoch, "was ", np.mean(losses))
    
def distance_accuracy(targets, preds, dis=2500, set='im2gps3k', trainset='train', opt=None):
    if trainset == 'train':
        coarse_gps = pd.read_csv(opt.resources + "cells_50_5000_images_4249548.csv") 
    if trainset == 'train1M':
        coarse_gps = pd.read_csv(opt.resources + "cells_50_5000_images_1M.csv")   

    course_preds = list(coarse_gps.iloc[preds][['latitude_mean', 'longitude_mean']].to_records(index=False))
    course_target = [(x[0], x[1]) for x in targets]   

    total = len(course_target)
    correct = 0

    for i in range(len(course_target)):
        #print(GD(course_preds[i], course_target[i]).km)
        if GD(course_preds[i], course_target[i]).km <= dis:
            correct += 1

    return correct / total

def eval_images(val_dataloader, model, epoch, opt):

    data_iterator = val_dataloader

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []

    for i, (imgs, classes) in bar:

        labels = classes.cpu().numpy()

        imgs = imgs.to(opt.device)
        outs = model(imgs)
        outs = torch.argmax(outs, dim=-1).detach().cpu().numpy()

        targets.append(labels)
        preds.append(outs)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    '''
    macrof1 = f1_score(targets, preds, average='macro')
    weightedf1 = f1_score(targets, preds, average='weighted')
    accuracy =  accuracy_score(targets, preds)
    '''
    #np.set_printoptions(precision=15)
    #print(targets)
    acc2500 = distance_accuracy(targets, preds, dis=2500, trainset=opt.trainset, opt=opt)
    acc750 = distance_accuracy(targets, preds, dis=750, trainset=opt.trainset,opt=opt)
    acc200 = distance_accuracy(targets, preds, dis=200, trainset=opt.trainset, opt=opt)
    acc25 = distance_accuracy(targets, preds, dis=25, trainset=opt.trainset, opt=opt)
    acc1 = distance_accuracy(targets, preds, dis=1, trainset=opt.trainset, opt=opt)

    print("Epoch", epoch)

    print("Accuracy2500 is", acc2500)
    print("Accuracy750 is", acc750)
    print("Accuracy200 is", acc200)
    print("Accuracy25 is", acc25)
    print("Accuracy1 is", acc1)

    wandb.log({"2500 Accuracy" : acc2500})
    wandb.log({"750 Accuracy" : acc750})
    wandb.log({"200 Accuracy" : acc200})
    wandb.log({"25 Accuracy" : acc25})
    wandb.log({"1 Accuracy" : acc1})


def eval_one_epoch(val_dataloader, model, epoch, opt):

    data_iterator = val_dataloader

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []
    for i, (vid, coords, classes) in bar:

        labels = classes[:,0].cpu().numpy()

        vid = vid.to(opt.device)[:,:,0,:,:]
        outs, _ = model(vid)
        outs = torch.argmax(outs, dim=-1).detach().cpu().numpy()

        targets.append(labels)
        preds.append(outs)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

   # print(targets, preds)
    print("Epoch", epoch)
    print("Macro F1 Score is", f1_score(targets, preds, average='macro'))
    print("Weighted F1 Score is", f1_score(targets, preds, average='weighted'))
    print("Accuracy is", accuracy_score(targets, preds))



