import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn

#import network
import dataloader
from train_and_eval import train_single_frame, eval_one_epoch, train_images, eval_images

#from torch.utils.tensorboard import SummaryWriter
import wandb

from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from networks import GeoCLIP, VGGTriplet, BasicNetVLAD
import networks 

parser = argparse.ArgumentParser()

opt = parser.parse_args()
opt.kernels = multiprocessing.cpu_count()
opt.size = 224

opt.n_epochs = 20

opt.description = 'Trying ResNet50 with Im2GPS'
opt.evaluate = False

opt.lr = 1e-2

opt.batch_size = 256


train_dataset = dataloader.M16Dataset(split='train')
val_dataset = dataloader.M16Dataset(split='im2gps3k')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)

opt.device = torch.device('cuda')


config = {
    'learning_rate' : opt.lr,
    'epochs' : opt.n_epochs,
    'batch_size' : opt.batch_size,
    'architecture' : "Just ResNet50"
}

wandb.init(project='GeoGuessNet', 
        entity='ahkerrigan',
        config=config)

#weights = [1/40619, 1/2063, 1/1147, 1/4391]
#class_weights = torch.FloatTensor(weights).cuda()
#criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
criterion = torch.nn.CrossEntropyLoss()

model = networks.JustResNet()

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=500, 
#                                                   eta_min=1e-6)


_ = model.to(opt.device)
wandb.watch(model, criterion, log="all")

acc10 = 0
for epoch in range(opt.n_epochs): 

    if not opt.evaluate:
        _ = model.train()

        #train_one_epoch(train_dataloader, ground_model, aerial_model, ground_optimizer, aerial_optimzer, criterion, opt, epoch, writer)
        #train_one_epoch_temp1(train_dataloader, model, optimizer, opt, epoch, writer)
        #train_one_epoch_temp1(train_dataloader, model, optimizer, opt, epoch, writer)
        #train_single_frame(train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, opt=opt, epoch=epoch, writer=writer)
        train_images(train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, opt=opt, epoch=epoch, val_dataloader=val_dataloader)
    #eval_one_epoch(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt, writer=writer)
    torch.save(model.state_dict(), weights + '/' opt.description + '.pth')
    eval_images(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt)
    scheduler.step()
    
    
    #acc10 = max(acc10, validate_one_epoch(val_dataloader, model, opt, epoch, writer))

    #print("Best acc10 is", acc10)
    #validate_loss(val_dataloader, model, opt, epoch, writer)

