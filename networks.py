#from json import encoder
#from msilib.schema import Feature
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from positional_encodings import PositionalEncoding1D

from einops import rearrange

import torch.nn.functional as F
import torchvision.models as models

import torch.nn as nn

import numpy as np

import torch

from geopy.distance import lonlat, distance

from utilities.detrstuff import nested_tensor_from_tensor_list
from transformers import ViTModel


class BasicDETR(nn.Module):
    def __init__(self, n_hier=8):
        super().__init__()

        #self.detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
        self.detr = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=4, batch_first=True)
        self.hier_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.hier_query = torch.rand((n_hier, 256)).cuda()

        self.classifier0 = nn.Linear(256, 4)

    def forward(self, x):
        bs, ch, h, w = x.shape

        # Remove the frame dimension
        #x = x[:,:,0,:,:]
        #print(x.shape)

        #inputs = self.feature_extractor.pad_and_create_pixel_mask(pixel_values_list=x, return_tensors="pt")
        #inputs = self.feature_extractor(images=x, return_tensors="pt")
        #print(inputs)
        detr_outputs = self.detr(x).last_hidden_state
        hier_qs = self.hier_query.unsqueeze(0).repeat(bs, 1, 1)

        x = self.hier_decoder(hier_qs, detr_outputs)
        class0 = self.classifier0(x[:, 0, :])


        return F.relu(class0)

class JustResNet(nn.Module):
    def __init__(self, backbone=models.resnet50(pretrained=True), trainset='train'):
        super().__init__()

        
        self.n_features = backbone.fc.in_features
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone.flatten = nn.Flatten(start_dim=1)
        

        '''
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        '''

        if trainset == 'train':
            self.classification1 = nn.Linear(self.n_features, 2967)
            self.classification2 = nn.Linear(self.n_features, 6505)
            self.classification3 = nn.Linear(self.n_features, 11570)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        if trainset == 'bddtrain':
            self.classification1 = nn.Linear(self.n_features, 49)
            self.classification2 = nn.Linear(self.n_features, 215)
            self.classification3 = nn.Linear(self.n_features, 520)            
        

        #self.classification = nn.Sequential(
        #    nn.Linear(2048, 2048//2),
        #    nn.ReLU(True),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(2048//2, 686)
        #)
        #self.classification = nn.Linear(2048, 3298)

    def forward(self, x):
        bs, ch, h, w = x.shape

        
        x = self.backbone(x)
        #x = self.backbone(x)
        x1 = self.classification1(x)
        x2 = self.classification2(x)
        x3 = self.classification3(x)

        return x1, x2, x3

class GeoGuess1(nn.Module):
    def __init__(self, backbone=models.resnet101(pretrained=True), trainset='train'):
        super().__init__()

        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        
        if trainset == 'train':
            self.classification1 = nn.Linear(self.n_features, 2967)
            self.classification2 = nn.Linear(self.n_features, 6505)
            self.classification3 = nn.Linear(self.n_features, 11570)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        self.queries = torch.rand(3, 768).cuda()

        #self.classification = nn.Sequential(
        #    nn.Linear(2048, 2048//2),
        #    nn.ReLU(True),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(2048//2, 686)
        #)
        #self.classification = nn.Linear(2048, 3298)

    def forward(self, x):
        bs, ch, h, w = x.shape

        
        x = self.backbone(x).last_hidden_state
        queries = self.queries.unsqueeze(0).repeat(bs, 1, 1)

        x = self.decoder(queries, x)
        #x = self.backbone(x)
        x1 = self.classification1(x[:,0,:])
        x2 = self.classification2(x[:,1,:])
        x3 = self.classification3(x[:,2,:])

        return x1, x2, x3

# Adding 4 encoders, to be fair with seperating hierarchies 
class GeoGuess2(nn.Module):
    def __init__(self, backbone=models.resnet101(pretrained=True), trainset='train'):
        super().__init__()

        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        
        if trainset == 'train':
            self.classification1 = nn.Linear(self.n_features, 2967)
            self.classification2 = nn.Linear(self.n_features, 6505)
            self.classification3 = nn.Linear(self.n_features, 11570)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.final_encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, x):
        bs, ch, h, w = x.shape

        
        x = self.backbone(x).last_hidden_state
        x = self.final_encoder_layers(x)

        x1 = self.classification1(x[:,0,:])
        x2 = self.classification2(x[:,0,:])
        x3 = self.classification3(x[:,0,:])

        return x1, x2, x3

# 4 layers for each of the seperate hierarchies 
class GeoGuess3(nn.Module):
    def __init__(self, backbone=models.resnet101(pretrained=True), trainset='train'):
        super().__init__()

        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        
        if trainset == 'train':
            self.classification1 = nn.Linear(self.n_features, 2967)
            self.classification2 = nn.Linear(self.n_features, 6505)
            self.classification3 = nn.Linear(self.n_features, 11570)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.final_encoder_layers1 = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.final_encoder_layers2 = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.final_encoder_layers3 = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, x):
        bs, ch, h, w = x.shape

        
        x = self.backbone(x).last_hidden_state
        x1 = self.final_encoder_layers1(x)
        x2 = self.final_encoder_layers2(x)
        x3 = self.final_encoder_layers3(x)

        x1 = self.classification1(x1[:,0,:])
        x2 = self.classification2(x2[:,0,:])
        x3 = self.classification3(x3[:,0,:])

        return x1, x2, x3


class GeoGuess4(nn.Module):
    def __init__(self, backbone=models.resnet50(pretrained=True), trainset='train'):
        super().__init__()

 
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.n_features = 768

        self.hier1encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True), num_layers=2)
        self.hier2encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True), num_layers=2)
        self.hier3encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True), num_layers=2)
        

        if trainset == 'train':
            self.classification1 = nn.Linear(self.n_features, 2967)
            self.classification2 = nn.Linear(self.n_features, 6505)
            self.classification3 = nn.Linear(self.n_features, 11570)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        if trainset == 'bddtrain':
            self.classification1 = nn.Linear(self.n_features, 49)
            self.classification2 = nn.Linear(self.n_features, 215)
            self.classification3 = nn.Linear(self.n_features, 520)            
        

        #self.classification = nn.Sequential(
        #    nn.Linear(2048, 2048//2),
        #    nn.ReLU(True),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(2048//2, 686)
        #)
        #self.classification = nn.Linear(2048, 3298)

    def forward(self, x):
        bs, ch, h, w = x.shape

        
        x = self.backbone(x).hidden_states

        x1 = self.hier1encoder(x[-3])
        x2 = self.hier2encoder(x[-2])
        x3 = self.hier3encoder(x[-1])
        feats = x3[:,0,:]

        x1 = self.classification1(x1[:,0,:])
        x2 = self.classification2(x2[:,0,:])
        x3 = self.classification3(x3[:,0,:])
        return x1, x2, x3, x3

if __name__ == "__main__":

    image = torch.rand((1,3, 224,224))

    '''
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50")
    inputs = feature_extractor.pad_and_create_pixel_mask(pixel_values_list=image, return_tensors="pt")
    print(inputs['pixel_values'].shape)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)
    '''

    model = GeoGuess4()
    x1, x2, x3, x3 = model(image)
    print(x1.shape)


    
    #model = BasicDETR()
    #tensor = torch.rand((16, 3, 15, 224, 224))
    #x = model(tensor)
    #print(x.shape)