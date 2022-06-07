import argparse
import multiprocessing
#import argparge
import torch
import networks

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.kernels = multiprocessing.cpu_count()

    opt.BDDfolder = "/home/alec/Documents/BigDatasets/BDD100k_Big/Ground/"
    opt.yfcc25600folder = "/home/alec/Documents/BigDatasets/yfcc25600/"
    opt.mp16folder = "/home/alec/Documents/BigDatasets/mp16/"
    opt.im2gps3k = "/home/alec/Documents/SmallDatasets/im2gps3ktest/"

    opt.resources = "/home/alec/Documents/BigDatasets/resources/"

    opt.size = 224

    opt.n_epochs = 20

    #opt.description = 'GeoGuess4-4.2M-Im2GPS3k-F*'
    opt.description = 'Testing ThreeWay Metric'

    opt.archname = 'ResNet50'
    opt.evaluate = False

    opt.lr = 1e-2
    opt.step_size = 3

    opt.batch_size = 40
    opt.distances = [2500, 750, 200, 25, 1]
    opt.trainset = 'traintriplet'
    opt.testset = 'im2gps3k'
    opt.device = torch.device('cuda')

    return opt