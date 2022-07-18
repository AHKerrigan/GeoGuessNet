import argparse
import multiprocessing
#import argparge
import torch
import networks

def getopt():
    parser = argparse.ArgumentParser()

    
    opt = parser.parse_args()
    opt.gpus = 1
    opt.kernels = multiprocessing.cpu_count()
    #pt.kernels = opt.gpus * 4

    opt.BDDfolder = "/home/alec/Documents/BigDatasets/BDD100k_Big/Ground/"
    opt.yfcc25600folder = "/home/alec/Documents/BigDatasets/yfcc25600/"
    opt.mp16folder = "/home/alec/Documents/BigDatasets/mp16/"
    opt.im2gps3k = "/home/alec/Documents/SmallDatasets/im2gps3ktest/"

    opt.resources = "/home/alec/Documents/BigDatasets/resources/"

    opt.size = 224

    opt.n_epochs = 20

    #opt.description = 'GeoGuess4-4.2M-Im2GPS3k-F*'
    opt.description = 'ResNet50+NoScenes+ExtraLayersDropout+NewData'
    opt.evaluate = False

    # How often to report loss
    opt.loss_per_epoch = 100

    # How often to validate
    opt.val_per_epoch = 25

    opt.lr = 0.01
    opt.step_size = 4
    opt.hier_eval = True
    opt.scene = False
    opt.mixed_pres = True
    
    opt.loss = 'ce'
    opt.model = 'JustResNet'
    opt.archname = opt.model

    opt.wandb = True

    opt.batch_size = 200
    opt.distances = [2500, 750, 200, 25, 1]
    opt.trainset = 'train'
    opt.testset1 = 'im2gps3k'
    opt.testset2 = 'yfcc25600'
    opt.device = torch.device('cuda')



    return opt