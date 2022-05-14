import json
from os.path import exists 
import numpy as np


classfile="/home/alec/Documents/GeoGuessNet/resources/mp16_1M_labels.json"
class_info = json.load(open(classfile))
base_folder = '/home/alec/Documents/BigDatasets/mp16/'

fnames = []
classes = []


for row in class_info:
    filename = base_folder + row
    if exists(filename):
        fnames.append(filename)
        classes.append([int(x) for x in class_info[row]])

count = [classes[i][0] for i in range(len(classes))]
print(max(count))

hist, nums = np.histogram(count, bins=np.arange(686))
print(hist)
print(nums)

print(np.sum(hist))
most = np.argmax(hist)
print(most, hist[most])     

print(hist[57]/22196)