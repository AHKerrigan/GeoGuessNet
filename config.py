import argparse
import multiprocessing
import argparge
import torch

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.kernels = multiprocessing.cpu_count()

    opt.BDDfolder = '/home/alec/Documents/BigDatasets/BDD100k_Big'
    opt.yfcc25600folder = "/home/c3-0/datasets/MP-16/resources/images/yfcc25600/"
    opt.mp16folder = "/home/c3-0/datasets/MP-16/resources/images/mp16/"
    opt.im2gps3k = "/home/al209167/datasets/im2gps3ktest/"

    opt.resources = "/home/al209167/datasets/Resources/"

    opt.size = 224

    opt.n_epochs = 20

    opt.description = 'ResNet101-coarse'
    opt.evaluate = False

    opt.lr = 1e-2

    opt.batch_size = 256
    opt.trainset = 'train'
    opt.device = torch.device('cuda')

    return opt