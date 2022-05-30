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

    for i ,(imgs, classes, gps) in bar:

        batch_size = imgs.shape[0]
        labels1 = classes[:,0]
        labels2 = classes[:,1]
        labels3 = classes[:,2]
        #labels1 = rearrange(labels1, 'bs c -> (bs c)')

        labels1 = labels1.to(opt.device)
        labels2 = labels2.to(opt.device)
        labels3 = labels3.to(opt.device)


        imgs = imgs.to(opt.device)

        optimizer.zero_grad()
        outs1, outs2, outs3, _ = model(imgs)

        torch.set_printoptions(edgeitems=30)

        loss1 = 0
        loss2 = 0
        loss3 = 0

        loss1 = criterion(outs1, labels1)
        loss2 = criterion(outs2, labels2)
        loss3 = criterion(outs3, labels3)

        loss = loss1 + loss2 + loss3

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
            if opt.trainset == 'train':
                wandb.log({"Training Loss" : loss.item()})
            else:
                wandb.log({opt.trainset + " Training Loss" : loss.item()})
            #print("interation", i, "of", len(data_iterator))
        if val_dataloader != None and i % (val_cycle * 5) == 0:
            eval_images(val_dataloader, model, epoch, opt)
    print("The loss of epoch", epoch, "was ", np.mean(losses))

def train_images_metric(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch, val_dataloader=None):

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

    for i ,(imgs, classes, gps) in bar:

        batch_size = imgs.shape[0]
        labels1 = classes[:,0]
        labels2 = classes[:,1]
        labels3 = classes[:,2]
        #labels1 = rearrange(labels1, 'bs c -> (bs c)')

        labels1 = labels1.to(opt.device)
        labels2 = labels2.to(opt.device)
        labels3 = labels3.to(opt.device)


        imgs = imgs.to(opt.device)

        optimizer.zero_grad()
        outs1, outs2, outs3, feats = model(imgs)

        torch.set_printoptions(edgeitems=30)

        loss1 = 0
        loss2 = 0
        loss3 = 0

        loss1 = criterion(outs1, labels1)
        loss2 = criterion(outs2, labels2)
        loss3 = criterion(outs3, labels3)


        gpsloss = gps_loss(feats, feats.flip(0), gps, gps.flip(0))

        #print("Losses are ", loss1.item(), loss2.item(), loss3.item(), gpsloss.item())

        loss = loss1 + loss2 + loss3 + gpsloss

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
            if opt.trainset == 'train':
                wandb.log({"Training Loss" : loss.item()})
            else:
                wandb.log({opt.trainset + " Training Loss" : loss.item()})
            #print("interation", i, "of", len(data_iterator))
        if val_dataloader != None and i % (val_cycle * 5) == 0:
            eval_images(val_dataloader, model, epoch, opt)
    print("The loss of epoch", epoch, "was ", np.mean(losses))
    
def distance_accuracy(targets, preds, dis=2500, set='im2gps3k', trainset='train', opt=None):
    if trainset == 'train':
        coarse_gps = pd.read_csv(opt.resources + "cells_50_1000_images_4249548.csv") 
    if trainset == 'train1M':
        coarse_gps = pd.read_csv(opt.resources + "cells_50_1000_images_1M.csv")
    if trainset == 'trainbdd':
        coarse_gps = pd.read_csv(opt.resources + "BDD-50-200images.csv")        

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
    return

    data_iterator = val_dataloader

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []

    for i, (imgs, classes) in bar:

        labels = classes.cpu().numpy()

        imgs = imgs.to(opt.device)
        with torch.no_grad():
            outs1, outs2, outs3, _ = model(imgs)
        outs = torch.argmax(outs3, dim=-1).detach().cpu().numpy()

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
    accuracies = []
    for dis in opt.distances:

        acc = distance_accuracy(targets, preds, dis=dis, trainset=opt.trainset, opt=opt)
        print("Accuracy", dis, "is", acc)
        if opt.testset == 'im2gps3k':
            wandb.log({ str(dis) + " Accuracy" : acc})
        else:
            wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})


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

def gps_loss(x_feats, y_feats, x_gps, y_gps):
    x_gps = x_gps.cpu().numpy()
    y_gps = y_gps.cpu().numpy()


    print(x_feats[0], x_feats[-1], y_feats[0], y_feats[-1], x_gps[0], x_gps[-1], y_gps[0], y_gps[-1])

    dis = []
    for i in range(len(x_gps)):
       dis.append(GD(x_gps[i], y_gps[i]).meters)

    feat_dist = torch.linalg.vector_norm(x_feats - y_feats, ord=2, dim=1)
    feat_dist = 4 * feat_dist / feat_dist.max()
    
    gps_dis = torch.FloatTensor(dis).cuda()
    gps_dis = 4 * gps_dis / gps_dis.max()

    return F.l1_loss(feat_dist, gps_dis)


    

