import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn

#import network
import dataloader
from train_and_eval import eval_images_weighted, train_images, train_images_ood, eval_images, train_images_filtered, eval_images_weighted

import wandb
import dill as pickle

import networks 
from config import getopt
import copy
import torch.distributed as dist
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from lit import LitModel, ValEveryNSteps
from pytorch_lightning.callbacks import TQDMProgressBar
from collections import OrderedDict
#from torchsummary import summary

# remove whenever that scatter gather deprecation is fixed in lightning
import warnings
warnings.filterwarnings("ignore")

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

opt = getopt()


if opt.wandb:
    #config = {
    #    'learning_rate' : opt.lr,
    #    'epochs' : opt.n_epochs,
    #    'batch_size' : opt.batch_size,
    #    'architecture' : opt.archname
    #}

    wandb_logger = WandbLogger(project="geoguessnet",
                               name=opt.description,
                               entity='crcvgeo')



#train_dataset = dataloader.M16Dataset(split=opt.trainset, opt=opt)
#pickle.dump(train_dataset, open("weights/train_datasets.pkl", "wb"))
with open("weights/train_datasets.pkl", "rb") as f:
    train_dataset = pickle.load(f)
val_dataset1 = dataloader.M16Dataset(split=opt.testset1, opt=opt)
val_dataset2 = dataloader.M16Dataset(split=opt.testset2, opt=opt)

train_batch = opt.batch_size
if opt.tencrop: 
    val_batch = opt.batch_size // 10
else:
    val_batch = opt.batch_size
# Non-distributed training
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, num_workers=opt.kernels, shuffle=True, drop_last=False, pin_memory=True)
val_dataloader1 = torch.utils.data.DataLoader(val_dataset1, batch_size=val_batch, num_workers=opt.kernels, shuffle=True, drop_last=False, pin_memory=True)
val_dataloader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=val_batch, num_workers=opt.kernels, shuffle=True, drop_last=False, pin_memory=True)

val_dataloaders = [val_dataloader1, val_dataloader2]

if opt.loss == 'ce':
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(opt.device)
if opt.loss == 'isomax':
    criterion = networks.IsoMaxLoss()

if opt.model == 'JustResNet':
    model = networks.JustResNet(trainset=opt.trainset)
    torch.backends.cudnn.benchmark = True
if opt.model == 'JustResNetOOD':
    model = networks.JustResNetOOD(trainset=opt.trainset)
if opt.model == 'GeoGuess1':
    model = networks.GeoGuess1(trainset=opt.trainset)
if opt.model == 'GeoGuess2':
    model = networks.GeoGuess2(trainset=opt.trainset)
if opt.model == 'GeoGuess3':
    model = networks.GeoGuess3(trainset=opt.trainset)
if opt.model == 'translocator':
    model = networks.Translocator(trainset='train')
if opt.model == 'isomax':
    model = networks.IsoMax()

#dup_model = copy.deepcopy(model)
#for param in dup_model.parameters():
#    param.requires_grad = False
#state_dict = remove_data_parallel((torch.load('weights/SceneConf-16Scenes-NewData.pth')['state_dict']))
#model.load_state_dict(state_dict)
#if opt.wandb: wandb.watch(model, criterion, log="all")

n_steps = len(train_dataset) // (opt.batch_size)
loss_cycle = (len(train_dataset) // (opt.batch_size * opt.loss_per_epoch))
val_cycle = (len(train_dataset) // (opt.batch_size * opt.val_per_epoch))

# We want to loss/val cycle to be consistent with the accumulator 
while loss_cycle % opt.accumulate != 0:
    loss_cycle += 1
while val_cycle % opt.accumulate != 0:
    val_cycle += 1


print("Outputting loss every", loss_cycle, "batches")
print("Validating every", val_cycle, "batches")

#val_callback = ValEveryNSteps(val_cycle)
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='weights/',
                                                   filename="{epoch}--"+opt.model+"--"+opt.description,
                                                   every_n_train_steps = n_steps // 3)

LitModel = LitModel(opt=opt, model=model)

if opt.cluster:
    progress_bar_refresh_rate = loss_cycle
else:
    progress_bar_refresh_rate = 1

progress_bar = TQDMProgressBar(refresh_rate=progress_bar_refresh_rate)


trainer = pl.Trainer(accelerator ="gpu", 
                     devices= -1, 
                     strategy="dp", 
                     precision=16, 
                     accumulate_grad_batches=opt.accumulate, 
                     logger=wandb_logger,
                     max_epochs=40,
                     callbacks=[checkpoint_callback, progress_bar],
                     val_check_interval = 1 / opt.val_per_epoch,
                     log_every_n_steps = loss_cycle)
trainer.fit(LitModel, train_dataloaders = train_dataloader, val_dataloaders = [val_dataloader1, val_dataloader2])
#trainer.validate(LitModel, dataloaders=[val_dataloader1, val_dataloader2])