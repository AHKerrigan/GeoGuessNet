import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn

#import network
import dataloader
from train_and_eval import eval_images_weighted, train_images, train_images_ood, eval_images, train_images_filtered, eval_images_weighted


#from torch.utils.tensorboard import SummaryWriter
import wandb
import dill as pickle

from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from networks import GeoCLIP, VGGTriplet, BasicNetVLAD
import networks 
from config import getopt
import copy
import torch.distributed as dist
#from torchsummary import summary
opt = getopt()


if opt.wandb:
    config = {
        'learning_rate' : opt.lr,
        'epochs' : opt.n_epochs,
        'batch_size' : opt.batch_size,
        'architecture' : opt.archname
    }

    wandb.init(project='geoguessnet', 
            entity='crcvgeo',
            config=config)
    wandb.run.name = opt.description
    wandb.save()



#train_dataset = dataloader.M16Dataset(split=opt.trainset, opt=opt)
#pickle.dump(train_dataset, open("weights/train_datasets.pkl", "wb"))
with open("weights/train_datasets.pkl", "rb") as f:
    train_dataset = pickle.load(f)
val_dataset1 = dataloader.M16Dataset(split=opt.testset1, opt=opt)
val_dataset2 = dataloader.M16Dataset(split=opt.testset2, opt=opt)

# Non-distributed training
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False, pin_memory=True)
val_dataloader1 = torch.utils.data.DataLoader(val_dataset1, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False, pin_memory=True)
val_dataloader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False, pin_memory=True)

val_dataloaders = [val_dataloader1, val_dataloader2]

if opt.loss == 'ce':
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
if opt.loss == 'isomax':
    criterion = networks.IsoMaxLoss()

if opt.model == 'JustResNet':
    model = networks.JustResNet(trainset=opt.trainset)
    torch.backends.cudnn.benchmark = True
if opt.model == 'JustResNetOOD':
    model = networks.JustResNetOOD(trainset=opt.trainset)
if opt.model == 'GeoGuess1':
    model = networks.GeoGuess1(trainset=opt.trainset)
if opt.model == 'translocator':
    model = networks.Translocator(trainset='train')
if opt.model == 'isomax':
    model = networks.IsoMax()

#dup_model = copy.deepcopy(model)
#for param in dup_model.parameters():
#    param.requires_grad = False
#dup_model.load_state_dict(torch.load('/home/alec/Documents/GeoGuessNet/weights/ResNet50+Hier+Scenes+NewData.pth'))


optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.5)

if opt.mixed_pres:
    scaler = torch.cuda.amp.GradScaler()
if opt.gpus > 1:
    model = nn.DataParralel(model)


_ = model.to(opt.device)
if opt.wandb: wandb.watch(model, criterion, log="all")

acc10 = 0
for epoch in range(opt.n_epochs): 

    if not opt.evaluate:
        _ = model.train()

        #train_one_epoch(train_dataloader, ground_model, aerial_model, ground_optimizer, aerial_optimzer, criterion, opt, epoch, writer)
        #train_one_epoch_temp1(train_dataloader, model, optimizer, opt, epoch, writer)
        #train_one_epoch_temp1(train_dataloader, model, optimizer, opt, epoch, writer)
        #train_single_frame(train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, opt=opt, epoch=epoch, writer=writer)
        train_images(train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, opt=opt, epoch=epoch, val_dataloaders=val_dataloaders, scaler=scaler)
        #train_images_ood(train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, opt=opt, epoch=epoch, val_dataloaders=val_dataloaders, scaler=scaler)
        #train_images_filtered(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch, val_dataloader=val_dataloader, original_model=dup_model)

    #eval_one_epoch(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt, writer=writer)
    save_dict = {
        'state_dict' : model.state_dict(),
        'epoch' : epoch,
        'optimizer' : optimizer,
        'scheduler' : scheduler
    }
    torch.save(save_dict, 'weights/' + opt.description + '.pth')
    #eval_images(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt)
    scheduler.step()
    
    
    #acc10 = max(acc10, validate_one_epoch(val_dataloader, model, opt, epoch, writer))

    #print("Best acc10 is", acc10)
    #validate_loss(val_dataloader, model, opt, epoch, writer)

