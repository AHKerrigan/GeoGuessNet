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
from tqdm import tqdm


import evaluate

def train_single_frame(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch, writer):


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

        labels = classes[:,0]
        labels = torch.tensor(labels).to(opt.device)
        vid = vid.to(opt.device)

        outs = model(vid)
        #print(outs.shape)
        #print(labels.shape)

        loss = criterion(outs, labels)

        loss.backward()

        optimizer.step()
        model.zero_grad()       
        scheduler.step()

        losses.append(loss.item())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    print("The loss of epoch", epoch, "was ", np.mean(losses))
    writer.add_scalar('Loss/Train', np.mean(losses), epoch)

def eval_one_epoch(val_dataloader, model, epoch, opt, writer):

    data_iterator = val_dataloader

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []
    for i, (vid, coords, classes) in bar:

        labels = classes[:,0].cpu().numpy()

        vid = vid.to(opt.device)
        outs = torch.argmax(model(vid), dim=-1).detach().cpu().numpy()

        targets.append(labels)
        preds.append(outs)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

   # print(targets, preds)
    print("Epoch", epoch)
    print("Macro F1 Score is", f1_score(targets, preds, average='macro'))
    print("Weighted F1 Score is", f1_score(targets, preds, average='weighted'))
    print("Accuracy is", accuracy_score(targets, preds))



