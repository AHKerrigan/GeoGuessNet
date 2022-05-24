import argparse
import multiprocessing
#import argparge
import torch

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.kernels = multiprocessing.cpu_count()

    opt.BDDfolder = '/home/alec/Documents/BigDatasets/BDD100k_Big'
    opt.yfcc25600folder = "/home/alec/Documents/BigDatasets/yfcc25600/"
    opt.mp16folder = "/home/alec/Documents/BigDatasets/mp16/"
    opt.im2gps3k = "/home/alec/Documents/SmallDatasets/im2gps3ktest/"

    opt.resources = "/home/alec/Documents/BigDatasets/resources/"

    opt.size = 224

    opt.n_epochs = 20

    opt.description = 'ViT Base-coarse'
    opt.archname = 'Just ViT'
    opt.evaluate = False

    opt.lr = 1e-2

    opt.batch_size = 128
    opt.trainset = 'train'
    opt.device = torch.device('cuda')

    return opt