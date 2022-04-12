import time
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import numpy as np

from colorama import Fore, Style

from einops import rearrange

import torch
import torch.nn.functional as F
import pickle

import geopy

import evaluate

def train_single_frame(train_dataloader, model, criterion, optimizer, opt, epoch, writer):


    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader 

    losses = []

    print("Starting Epoch", epoch)

    for i ,( vid, coords) in enumerate(data_iterator):

        if i % 10 == 0:
            print("Starting minibatch", i)
        labels = []
        for c in coords:
            c = c.numpy()[0]
            if c[1] > -72.2003206:
                # Not in US (I think)
                #print("Not in US I think")
                labels.append([0.0, 0.0, 0.0, 0.0, 1.0])
            elif c[1] > -100:
                # Your in New York
                #print("new York")
                labels.append([1.0, 0.0, 0.0, 0.0, 0.0])
            elif c[0] > 37.8184:
                #print("Berkley")
                # You;re in berkley
                labels.append([0.0, 1.0, 0.0, 0.0, 0.0])
            elif c[1] < -122.2781913:
                # Youre in sanfran
                #print("San Fran")
                labels.append([0.0, 0.0, 1.0, 0.0, 0.0])
            else:
                # Bay area
                #print("Bay Area")
                labels.append([0.0, 0.0, 0.0, 1.0, 0.0])
        print(labels)
        labels = torch.tensor(labels).to(opt.device)
        vid = vid.to(opt.device)

        outs = model(vid)
        print(outs)

        loss = criterion(outs, labels)

        model.zero_grad()
        loss.backward()

        optimizer.step()

        losses.append(loss.item())
    print("The loss of epoch", epoch, "was ", np.mean(losses))
    writer.add_scalar('Loss/Train', np.mean(losses), epoch)
