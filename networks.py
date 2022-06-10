#from json import encoder
#from msilib.schema import Feature
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from positional_encodings import PositionalEncoding1D

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F
import torchvision.models as models

import torch.nn as nn

import numpy as np

import torch

from geopy.distance import lonlat, distance

from utilities.detrstuff import nested_tensor_from_tensor_list
from transformers import ViTModel

import copy
import pickle

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

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

        if trainset in ['train', 'traintriplet']:
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

        self.scene1 = nn.Linear(self.n_features, 3)  
        self.scene2 = nn.Linear(self.n_features, 16)
        self.scene3 = nn.Linear(self.n_features, 365)        
        

        maps = pickle.load(open("/home/alec/Documents/BigDatasets/resources/class_map.p", "rb"))

        self.coarse2medium = maps[0]
        self.medium2fine = maps[1]

        self.medium2fine[929] = 0
        self.medium2fine[3050] = 0

        #self.classification = nn.Sequential(
        #    nn.Linear(2048, 2048//2),
        #    nn.ReLU(True),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(2048//2, 686)
        #)
        #self.classification = nn.Linear(2048, 3298)

    def forward(self, x, evaluate=False):
        bs, ch, h, w = x.shape

        
        x = self.backbone(x)

        x1 = self.classification1(x)
        x2 = self.classification2(x)
        x3 = self.classification3(x)

        s1 = self.scene1(x)
        s2 = self.scene2(x)
        s3 = self.scene3(x)
        
        if not evaluate:
            return x1, x2, x3, s1, s2, s3
        else:
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

        self.layernorm1 = nn.LayerNorm(768)
        self.layernorm2 = nn.LayerNorm(768)
        self.layernorm3 = nn.LayerNorm(768)

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

        #x1 = self.layernorm1(x[-3])
        #x2 = self.layernorm2(x[-2]) )
        #x3 = self.layernorm3(self.hier3encoder(x[-1]))
        #feats = x3[:,0,:]

        x1 = self.layernorm1(x[-5][:,0])
        x2 = self.layernorm2(x[-3][:,0])
        x3 = self.layernorm3(x[-1][:,0])

        x1 = self.classification1(x1)
        x2 = self.classification2(x2)
        x3 = self.classification3(x3)

        #print(x1.shape)
        return x1, x2, x3, x3

class GeoGuess5(nn.Module):
    def __init__(self, backbone=models.resnet50(pretrained=True), trainset='train'):
        super().__init__()

 
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.n_features = 768

        self.layernorm1 = nn.LayerNorm(768)
        self.layernorm2 = nn.LayerNorm(768)
        self.layernorm3 = nn.LayerNorm(768)

        self.b1t1 = Transformer(768, 1, 12, 64, 1024)
        self.b1t2 = Transformer(768, 1, 12, 64, 1024)

        self.b2t1 = Transformer(768, 1, 12, 64, 1024)
        self.b2t2 = Transformer(768, 1, 12, 64, 1024)

        self.b3t1 = Transformer(768, 1, 12, 64, 1024)
        self.b3t2 = Transformer(768, 1, 12, 64, 1024)
        

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

        #x1 = self.layernorm1(x[-3])
        #x2 = self.layernorm2(x[-2]) )
        #x3 = self.layernorm3(self.hier3encoder(x[-1]))
        #feats = x3[:,0,:]

        # Course Branch
        x1 = x[-5][:,0]
        x1 = self.b1t1(x1)
        x1 = self.b1t2(x1)

        # Medium Branch
        x2 = x[-3][:,0]
        x2 = self.b2t1(x2)
        x2 = self.b2t2(x2)

        # Fine Branch
        x3 = x[-1][:,0]
        x3 = self.b3t1(x3)
        x3 = self.b3t2(x3)

        x1 = self.classification1(x1)
        x2 = self.classification2(x2)
        x3 = self.classification3(x3)

        #print(x1.shape)
        return x1, x2, x3, x3

class ThreeWay(nn.Module):
    def __init__(self, backbone=models.resnet50(pretrained=True), trainset='train'):
        super().__init__()

        self.n_features = backbone.fc.in_features

        self.coarsebranch = copy.deepcopy(backbone.layer4)
        self.mediumbranch = copy.deepcopy(backbone.layer4)
        self.finebranch = copy.deepcopy(backbone.layer4)
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-3])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)


        ''''
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        '''

        if trainset in ['train', 'traintriplet']:
            self.classification1 = nn.Linear(self.n_features, 2967)
            self.classification2 = nn.Linear(self.n_features, 6505)
            self.classification3 = nn.Linear(self.n_features, 11570)
        if trainset == 'train1M':
            self.classification = nn.Linear(self.n_features * 3, 689)
        if trainset == 'bddtrain':
            self.classification = nn.Linear(self.n_features * 3, 520)            
        

        #self.classification = nn.Sequential(
        #    nn.Linear(2048, 2048//2),
        #    nn.ReLU(True),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(2048//2, 686)
        #)
        #self.classification = nn.Linear(2048, 3298)

    def forward(self, x, return_feats = False):
        bs, ch, h, w = x.shape

        x = self.backbone(x)
        x1 = self.coarsebranch(x)
        x2 = self.mediumbranch(x)
        x3 = self.finebranch(x)

        x1 = self.flatten(self.avgpool(x1))
        x2 = self.flatten(self.avgpool(x2))
        x3 = self.flatten(self.avgpool(x3))

        fullfeatures = torch.cat((x1, x2, x3), dim=1)
        cls = self.classification(fullfeatures)

        return x3, x3, x3, cls


if __name__ == "__main__":

    image = torch.rand((84,3,224,224))

    '''
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50")
    inputs = feature_extractor.pad_and_create_pixel_mask(pixel_values_list=image, return_tensors="pt")
    print(inputs['pixel_values'].shape)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)
    '''

    model = JustResNet()
    x1, x2, x3, x3 = model(image)

    print(x3.shape)



    #_ = model.to('cuda')
    #image = image.to('cuda')



    #x1, x2, x3, cls = model(image)
    #print(x1.shape, x2.shape, x3.shape, cls.shape)


    
    #model = BasicDETR()
    #tensor = torch.rand((16, 3, 15, 224, 224))
    #x = model(tensor)
    #print(x.shape)