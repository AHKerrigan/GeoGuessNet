import argparse
import multiprocessing
#import argparge
import torch
import networks

def getopt():
    parser = argparse.ArgumentParser()

    
    opt = parser.parse_args()
    #opt = parser.parse_known_args()[0]
    opt.gpus = 1
    opt.kernels = multiprocessing.cpu_count()
    #opt.kernels = opt.gpus*4

    opt.BDDfolder = "/home/alec/Documents/BigDatasets/BDD100k_Big/Ground/"
    opt.yfcc25600folder = "/home/alec/Documents/BigDatasets/yfcc25600/"
    opt.mp16folder = "/home/alec/Documents/BigDatasets/mp16/"
    opt.im2gps3k = "/home/alec/Documents/SmallDatasets/im2gps3ktest/"

    opt.resources = "/home/alec/Documents/BigDatasets/resources/"

    opt.size = 224

    opt.n_epochs = 40

    #opt.description = 'GeoGuess4-4.2M-Im2GPS3k-F*'
    opt.description = 'Testing'
    opt.evaluate = False
    opt.cluster = False

    # How often to report loss
    opt.loss_per_epoch = 100

    # How often to validate
    opt.val_per_epoch = 25

    opt.lr = 0.1
    opt.step_size = 4
    opt.hier_eval = True
    opt.scene = True
    opt.mixed_pres = True

    # Only applies to validation 
    opt.tencrop = True
    
    opt.loss = 'ce'
    opt.model = 'GeoGuess1'
    opt.archname = opt.model

    opt.wandb = True

    opt.batch_size = 128
    opt.accumulate = 2
    opt.distances = [2500, 750, 200, 25, 1]
    
    opt.hier_hypers = [0.5, 0.3, 0.2]
    opt.trainset = 'train'
    opt.testset1 = 'im2gps3k'
    opt.testset2 = 'yfcc25600'
    opt.device = torch.device('cuda')

    opt.batch_size = opt.batch_size * opt.gpus

    return opt