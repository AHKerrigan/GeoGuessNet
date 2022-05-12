import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn

#import network
import dataloader
from train_and_eval import train_single_frame, eval_one_epoch

from torch.utils.tensorboard import SummaryWriter

from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from networks import GeoCLIP, VGGTriplet, BasicNetVLAD
import networks 

parser = argparse.ArgumentParser()

opt = parser.parse_args()
opt.kernels = multiprocessing.cpu_count()
opt.size = 224

opt.n_epochs = 300

opt.description = "Testing_oneframe_class"
opt.evaluate = False

opt.lr = 1e-4

opt.batch_size = 16


train_dataset = dataloader.BDDDataset(split='train')
val_dataset = dataloader.BDDDataset(split='val')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)

opt.device = torch.device('cuda')

#weights = [1/40619, 1/2063, 1/1147, 1/4391]
#class_weights = torch.FloatTensor(weights).cuda()
#criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
criterion = torch.nn.CrossEntropyLoss()

model = networks.JustResNet()

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=500, 
#                                                   eta_min=1e-6)

writer = SummaryWriter(log_dir='runs/'+opt.description)

_ = model.to(opt.device)

acc10 = 0
for epoch in range(opt.n_epochs): 

    if not opt.evaluate:
        _ = model.train()

        #train_one_epoch(train_dataloader, ground_model, aerial_model, ground_optimizer, aerial_optimzer, criterion, opt, epoch, writer)
        #train_one_epoch_temp1(train_dataloader, model, optimizer, opt, epoch, writer)
        #train_one_epoch_temp1(train_dataloader, model, optimizer, opt, epoch, writer)
        train_single_frame(train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, opt=opt, epoch=epoch, writer=writer)
    eval_one_epoch(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt, writer=writer)
    
    
    #acc10 = max(acc10, validate_one_epoch(val_dataloader, model, opt, epoch, writer))

    #print("Best acc10 is", acc10)
    #validate_loss(val_dataloader, model, opt, epoch, writer)

