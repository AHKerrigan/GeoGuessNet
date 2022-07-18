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
from geopy.distance import great_circle as GCD
from tqdm import tqdm

from torch.autograd import Variable

import wandb
import evaluate
import random
import pandas as pd

import copy

def train_images(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch, val_dataloaders=None, scaler=None):

    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader 

    losses = []
    running_loss = torch.tensor([0.0]).to(opt.device)
    dataset_size = 0


    loss_cycle = (len(data_iterator.dataset.data) // (opt.batch_size * opt.loss_per_epoch))
    val_cycle = (len(data_iterator.dataset.data) // (opt.batch_size * opt.val_per_epoch))
    print("Outputting loss every", loss_cycle, "batches")
    print("Validating every", val_cycle, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))

    for i ,(imgs, classes, scenes, gps) in bar:

        batch_size = imgs.shape[0]

        ############## Geolocalization labels  ##############
        labels1 = classes[:,0]
        labels2 = classes[:,1]
        labels3 = classes[:,2]
        #labels1 = rearrange(labels1, 'bs c -> (bs c)')

        labels1 = labels1.to(opt.device)
        labels2 = labels2.to(opt.device)
        labels3 = labels3.to(opt.device)

        ##############    Scene labels     ##############
        scenelabels = scenes[:,1]

        scenelabels = scenelabels.to(opt.device)

        # Cast to mixed precision to speed things up 
        with torch.cuda.amp.autocast():
            imgs = imgs.to(opt.device)
            
            ##############  Get Outputs ##############

            if opt.scene:
                outs1, outs2, outs3, scene1 = model(imgs)
            else:
                outs1, outs2, outs3, _ = model(imgs, evaluate=True)
            loss1 = criterion(outs1, labels1)
            loss2 = criterion(outs2, labels2)
            loss3 = criterion(outs3, labels3)

            loss = loss1 + loss2 + loss3

            if opt.scene:
                sceneloss = criterion(scene1, scenelabels)
                loss += sceneloss

            #loss /= opt.accumulate

        scaler.scale(loss).backward()

        if ((i + 1) % opt.accumulate == 0) or (i+ 1 == len(data_iterator.dataset.data)):

            running_loss += (loss * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            scaler.step(optimizer) 
            scaler.update()
            optimizer.zero_grad()   

        #losses.append(loss.item())

        running_loss += (loss * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        
        step = ((epoch + 1) * opt.loss_per_epoch) + ((i+1) / loss_cycle)
        if (i+1) % loss_cycle == 0:
            bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss.item(),
                        LR=optimizer.param_groups[0]['lr'])
            if opt.trainset == 'train':
                if opt.wandb: wandb.log({"Training Loss" : loss.item(), 'Step' : int(step)})
            else:
                if opt.wandb: wandb.log({opt.trainset + " Training Loss" : loss.item()})
            #print("interation", i, "of", len(data_iterator))
        if val_dataloaders != None and i % val_cycle == 0:
            for val_dataloader in val_dataloaders:
                eval_images_weighted(val_dataloader=val_dataloader, model=model, epoch=epoch, step=int(step), opt=opt)
            #eval_images(val_dataloader, model, epoch, opt)

def train_images_filtered(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch, val_dataloader=None, original_model=None):

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

    for i ,(imgs, classes, scenes, gps) in bar:

        batch_size = imgs.shape[0]

        ############## Geolocalization labels  ##############
        labels1 = classes[:,0]
        labels2 = classes[:,1]
        labels3 = classes[:,2]
        #labels1 = rearrange(labels1, 'bs c -> (bs c)')

        labels1 = labels1.to(opt.device)
        labels2 = labels2.to(opt.device)
        labels3 = labels3.to(opt.device)

        ##############    Scene labels     ##############
        scenelabels1 = scenes[:,0]
        scenelabels2 = scenes[:,1]
        scenelabels3 = scenes[:,2]

        scenelabels1 = scenelabels1.to(opt.device)
        scenelabels2 = scenelabels2.to(opt.device)
        scenelabels3 = scenelabels3.to(opt.device)

        imgs = imgs.to(opt.device)

        optimizer.zero_grad()
        
        ##############  Get Outputs ##############
        if opt.scene:
            outs1, outs2, outs3, scene1, scene2, scene3 = model(imgs)
        else:
            outs1, outs2, outs3, x = model(imgs, evaluate=True)

        with torch.no_grad():
            prev_out, _, _, x = original_model(imgs, evaluate=True)
        
        ########### Create the Filter ############################

        prev_ce = F.cross_entropy(prev_out, labels1, reduce=False)
        mask = prev_ce < 7.8213
        mask = torch.nonzero(mask)

        torch.set_printoptions(edgeitems=30)

        loss1 = 0
        loss2 = 0
        loss3 = 0

        #loss1 = criterion(outs1, labels1)
        loss1 = F.cross_entropy(outs1, labels1, reduce=False)[mask].mean()
        loss2 = F.cross_entropy(outs2, labels2, reduce=False)[mask].mean()
        loss3 = F.cross_entropy(outs3, labels3, reduce=False)[mask].mean()

        loss = loss1 + loss2 + loss3

        if opt.scene:
            sceneloss1 = 0
            sceneloss2 = 0
            sceneloss3 = 0

            sceneloss1 = criterion(scene1, scenelabels1)
            sceneloss2 = criterion(scene2, scenelabels2)
            sceneloss3 = criterion(scene3, scenelabels3)

            sceneloss = sceneloss1 + sceneloss2 + sceneloss3

            loss += sceneloss

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
                if opt.wandb: wandb.log({"Training Loss" : loss.item()})
                else: print(f"Training Loss at epoch {epoch} step {i}: {loss.item()}")
            else: 
                if opt.wandb: wandb.log({opt.trainset + " Training Loss" : loss.item()})
                else: print(f"Training Loss at epoch {epoch} step {i}: {loss.item()}")
            #print("interation", i, "of", len(data_iterator))
        if val_dataloader != None and i % (val_cycle * 5) == 0:
            eval_images_weighted(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt)
            #eval_images(val_dataloader, model, epoch, opt)
    print("The loss of epoch", epoch, "was ", np.mean(losses))

def train_images_ood(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch, val_dataloaders=None):

    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader 

    losses = []
    running_loss = 0.0
    dataset_size = 0


    loss_cycle = (len(data_iterator.dataset.data) // (opt.batch_size * opt.loss_per_epoch))
    val_cycle = (len(data_iterator.dataset.data) // (opt.batch_size * opt.val_per_epoch))
    print("Outputting loss every", loss_cycle, "batches")
    print("Validating every", val_cycle, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))

    for i ,(imgs, classes, scenes, gps) in bar:

        batch_size = imgs.shape[0]

        ############## Geolocalization labels  ##############
        labels1 = classes[:,0]
        labels2 = classes[:,1]
        labels3 = classes[:,2]
        #labels1 = rearrange(labels1, 'bs c -> (bs c)')

        labels1 = labels1.to(opt.device)
        labels2 = labels2.to(opt.device)
        labels3 = labels3.to(opt.device)

        imgs = imgs.to(opt.device)

        optimizer.zero_grad()
        
        ##############  Get Outputs ##############
        outs1, outs2, outs3, _ = model(imgs, evaluate=False)

        # Get the one hot labels
        y1 = F.one_hot(labels1, num_classes=3298)
        y2 = F.one_hot(labels2, num_classes=7202)
        y3 = F.one_hot(labels3, num_classes=12893)

        #loss1 = criterion(outs1, labels1)
        #loss2 = criterion(outs2, labels2)
        loss3 = criterion(outs3, labels3)

        #loss = loss1 + loss2 + loss3
        loss = loss3

        loss.backward()

        optimizer.step()     

        losses.append(loss.item())

        dataset_size += batch_size

        running_loss += (loss.item() * batch_size)

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'], 
                        )
        
        step = ((epoch + 1) * opt.loss_per_epoch) + ((i+1) / loss_cycle)
        if (i+1) % loss_cycle == 0:
            if opt.trainset == 'train':
                if opt.wandb: wandb.log({"Training Loss" : loss.item(), 'Step' : int(step)})
            else:
                if opt.wandb: wandb.log({opt.trainset + " Training Loss" : loss.item()})
            #print("interation", i, "of", len(data_iterator))
        if val_dataloaders != None and i % val_cycle == 0:
            for val_dataloader in val_dataloaders:
                eval_images_weighted(val_dataloader=val_dataloader, model=model, epoch=epoch, step=int(step), opt=opt)
            #eval_images(val_dataloader, model, epoch, opt)
    print("The loss of epoch", epoch, "was ", np.mean(losses))
    
def distance_accuracy(targets, preds, dis=2500, set='im2gps3k', trainset='train', opt=None):
    if trainset in ['train', 'traintriplet']:
        coarse_gps = pd.read_csv(opt.resources + "cells_50_1000.csv") 
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
        #if GD(course_preds[i], course_target[i]).km <= dis:
        if GCD(course_preds[i], course_target[i]).km <= dis:
            correct += 1

    return correct / total

def eval_images(val_dataloader, model, epoch, step, opt):
    #return

    data_iterator = val_dataloader

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []

    for i, (imgs, classes) in bar:

        labels = classes.cpu().numpy()

        imgs = imgs.to(opt.device)
        with torch.no_grad():
            outs1, outs2, outs3, _ = model(imgs, evaluate=True)
        cls = torch.argmax(cls, dim=-1).detach().cpu().numpy()

        targets.append(labels)
        preds.append(cls)

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


def eval_images_weighted(val_dataloader, model, epoch, step, opt):
    #return

    data_iterator = val_dataloader

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []

    for i, (imgs, classes) in bar:

        labels = classes.cpu().numpy()

        imgs = imgs.to(opt.device)
        with torch.no_grad():
            outs1, outs2, outs3, _ = model(imgs, evaluate=True)

        outs1 = F.softmax(outs1, dim=1)
        outs2= F.softmax(outs2, dim=1)
        outs3 = F.softmax(outs3, dim=1)

        '''
        coarseweights = torch.ones(outs2.shape).cuda()
        mediumweights = torch.ones(outs3.shape).cuda()

        for i in range(outs2.shape[1]):
            coarseweights[:,i] = outs1[:,val_dataloader.dataset.coarse2medium[i]]

        outs2 = outs2 * coarseweights


        for i in range(outs3.shape[1]):
            mediumweights[:,i] = outs2[:,val_dataloader.dataset.medium2fine[i]]
        outs3 = outs3 * mediumweights

        '''
        outs3 = torch.argmax(outs3, dim=-1).detach().cpu().numpy()


        targets.append(labels)
        preds.append(outs3)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    #np.set_printoptions(precision=15)
    #print(targets)
    accuracies = []
    for dis in opt.distances:

        acc = distance_accuracy(targets, preds, dis=dis, trainset=opt.trainset, opt=opt)
        print("Accuracy", dis, "is", acc)
        if val_dataloader.dataset.split == 'im2gps3k':
            if opt.wandb: wandb.log({ str(dis) + " Accuracy" : acc, 'Step' : step})
            else: print(f"{str(dis)} Accuracy at epoch {epoch} step {step}: {acc}")
        else:
            if opt.wandb: wandb.log({val_dataloader.dataset.split + " " +  str(dis) + " Accuracy" : acc, 'Step' : step})
            else: print(f"{str(dis)} Accuracy at epoch {epoch} step {step}: {acc}")

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


    

