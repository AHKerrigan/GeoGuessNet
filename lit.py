import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F

import networks
from einops import rearrange
from train_and_eval import distance_accuracy
import pandas as pd
from geopy.distance import great_circle as GCD

import sys 
import gc
import pickle

from operator import itemgetter

from pympler import tracker

class Distance(torchmetrics.Metric):

    full_state_update: bool = False
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.trainset in ['train', 'traintriplet']:
            self.coarse_gps = pd.read_csv(opt.resources + "cells_50_1000.csv") 


        self.add_state(f"correct2500", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(f"correct750", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(f"correct200", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(f"correct25", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(f"correct1", torch.tensor(0), dist_reduce_fx="sum")

        self.all_dis = {
            2500 : self.correct2500,
            750 : self.correct750,            
            200 : self.correct200,
            25 : self.correct25,
            1 : self.correct1,
        }
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for dis in self.opt.distances:
            course_preds = list(self.coarse_gps.iloc[preds.cpu().numpy()][['latitude_mean', 'longitude_mean']].to_records(index=False))
            course_target = [(x[0], x[1]) for x in targets]   

            total = len(course_target)
            correct = 0

            for i in range(len(course_target)):
                #print(GD(course_preds[i], course_target[i]).km)
                #if GD(course_preds[i], course_target[i]).km <= dis:
                if GCD(course_preds[i], course_target[i]).km <= dis:
                    correct += 1
            self.all_dis[dis] += torch.tensor(correct)
        self.n_observations += preds.numel()
    def compute(self):

        ret = torch.zeros(len(self.opt.distances)).cuda()
        for i, dis in enumerate(self.opt.distances):
            #print(f"dis = {dis} self.all_dis[dis] / self.n_observations - {self.all_dis[dis]} / {self.n_observations}")
            ret[i] = self.all_dis[dis] / self.n_observations
        return ret


class ValEveryNSteps(pl.Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.run_evaluation(test_mode=False)


class LitModel(pl.LightningModule):
    def __init__(self, opt, model, n_dataloaders=2):
        super().__init__()
        self.opt = opt
        self.model = model
        self.n_dataloaders = 2

        self.val_accuracy = [Distance(opt).cuda(), Distance(opt).cuda()]

        if opt.trainset == 'train':
            maps = pickle.load(open(opt.resources+"class_map.p", "rb"))
            self.coarse2medium = maps[0]
            self.medium2fine = maps[1]

            self.medium2fine[1019] = 0
            self.medium2fine[4595] = 0
            self.medium2fine[4687] = 0


    def forward(self, x, evaluate=False):
        x = self.model(x, evaluate=evaluate)
        return x

    def training_step(self, batch, batch_idx):
        imgs, classes, scenes, gps = batch

        batch_size = imgs.shape[0]

        ############## Geolocalization labels  ##############
        labels1 = classes[:,0]
        labels2 = classes[:,1]
        labels3 = classes[:,2]

        ##############    Scene labels     ##############
        scenelabels = scenes[:,1]

        if self.opt.tencrop:
            imgs = rearrange(imgs, 'bs crop ch h w -> (bs crop) ch h w')

        if self.opt.scene:
            outs1, outs2, outs3, scene1, test = self(imgs)
        else:
            outs1, outs2, outs3, _ = self(imgs, evaluate=True)

        if self.opt.tencrop:
            outs1 = rearrange(outs1, '(bs crop) classes -> bs crop classes', crop=10).mean(1)
            outs2 = rearrange(outs2, '(bs crop)  classes -> bs crop classes', crop=10).mean(1)
            outs3 = rearrange(outs3, '(bs crop) classes -> bs crop classes', crop=10).mean(1)
            scene1 = rearrange(scene1, '(bs crop) classes -> bs crop classes', crop=10).mean(1)
        
        loss1 = F.cross_entropy(outs1, labels1, label_smoothing=0.1)
        loss2 = F.cross_entropy(outs2, labels2, label_smoothing=0.1)
        loss3 = F.cross_entropy(outs3, labels3, label_smoothing=0.1)

        loss = loss1 + loss2 + loss3

        if self.opt.scene:
            sceneloss = F.cross_entropy(scene1, scenelabels)
            loss += sceneloss
        
        self.log("Training Loss", loss, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, classes = batch


        labels = classes

        if self.opt.tencrop:
            imgs = rearrange(imgs, 'bs crop ch h w -> (bs crop) ch h w')

        outs1, outs2, outs3, _ = self(imgs, evaluate=True)

        # Handle ten crop
        if self.opt.tencrop:
            outs1 = rearrange(outs1, '(bs crop) classes -> bs crop classes', crop=10).mean(1)
            outs2 = rearrange(outs2, '(bs crop)  classes -> bs crop classes', crop=10).mean(1)
            outs3 = rearrange(outs3, '(bs crop) classes -> bs crop classes', crop=10).mean(1)

        outs1 = F.softmax(outs1, dim=1)
        outs2= F.softmax(outs2, dim=1)
        outs3 = F.softmax(outs3, dim=1)

        coarseweights = torch.ones(outs2.shape).cuda()
        mediumweights = torch.ones(outs3.shape).cuda()

        for i in range(outs2.shape[1]):
            coarseweights[:,i] = outs1[:,self.coarse2medium[i]]

        outs2 = outs2 * coarseweights

        for i in range(outs3.shape[1]):
            mediumweights[:,i] = outs2[:,self.medium2fine[i]]
        outs3 = outs3 * mediumweights

        outs3 = torch.argmax(outs3, dim=-1)

        batch_results = self.val_accuracy[dataloader_idx](outs3, labels)

        if dataloader_idx == 0:
            dataset_name = ""
        else:
            dataset_name = "yfcc25600 "

        #for i, dis in enumerate(self.opt.distances):
        #    self.log(f"{dataset_name}{dis} Accuracy batch", batch_results[i], prog_bar=False, sync_dist=True)
        #return {'targets' : labels, 'preds': outs3}

    #def validation_step_end(self, batch_parts):
    #
    #    print(batch_parts['targets'].shape)
    #    print(batch_parts['preds'].shape)

    def validation_epoch_end(self, validation_step_outputs):
        
        acc0 = self.val_accuracy[0].compute()
        acc1 = self.val_accuracy[1].compute()

        for i, dis in enumerate(self.opt.distances):
            self.log(f"{dis} Accuracy", acc0[i], prog_bar=False, sync_dist=True)
            self.log(f"yfcc25600 {dis} Accuracy", acc1[i], prog_bar=False, sync_dist=True)

        #mem = tracker.SummaryTracker()
        #print(sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.step_size, gamma=0.5)
        return [optimizer], [lr_scheduler]