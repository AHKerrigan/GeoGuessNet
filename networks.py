#from json import encoder
#from msilib.schema import Feature
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from positional_encodings.torch_encodings import PositionalEncoding1D

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F
import torchvision.models as models

import torch.nn as nn

import numpy as np

import torch

from geopy.distance import lonlat, distance

from utilities.detrstuff import nested_tensor_from_tensor_list
from transformers import ViTModel, DetrForSegmentation
from transformers import SwinModel
import random

import copy
import pickle

from utils import VisionTransformer, get_seg_model, HRNET_48
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
    def __init__(self, dim, hidden_dim, dropout = 0., dropout2 = -1, out_dim=-1):
        super().__init__()
        if out_dim == -1:
            out_dim = dim
        if dropout2 == -1:
            dropout2 = dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
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
        print("X shape is", x.shape)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        print("qkv shape is", qkv[0].shape, qkv[1].shape, qkv[2].shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CustomAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        qkv = (q, k, v)
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

class CustomTransformer1(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim

        self.lnk = nn.LayerNorm(dim)
        self.lnv = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                CustomAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                nn.LayerNorm(dim),
                CustomAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, q, k, v):
        q1 = self.lnk(q)
        k = self.lnk(k)
        v = self.lnv(v)
        for ln1, cxattn, ln2, sattn, ff in self.layers:
            q = ln1(q)
            x = cxattn(q, k, v) + q
            x = ln2(x)
            x = sattn(x, x, x) + x
            x = ff(x) + x
        return x

class JustResNet(nn.Module):
    def __init__(self, backbone=models.resnet18(weights='ResNet18_Weights.DEFAULT'), trainset='train'):
        super().__init__()

        '''
        self.n_features = backbone.fc.in_features
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone.flatten = nn.Flatten(start_dim=1)
        '''

        '''
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        '''

        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k", output_hidden_states=True)
        self.n_features = 1024

        if trainset in ['train', 'traintriplet']:
            self.classification1 = nn.Linear(self.n_features, 3298)
            self.classification2 = nn.Linear(self.n_features, 7202)
            #self.classification2 = FeedForward(dim=self.n_features, hidden_dim=3000, dropout = 0.1, dropout2 = 0.0, out_dim=7202)
            self.classification3 = nn.Linear(self.n_features, 12893)
            #self.classification3 = FeedForward(dim=self.n_features, hidden_dim=6000, dropout = 0.1, dropout2 = 0.0, out_dim=12893)
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
        
    def forward(self, x, evaluate=False):
        #bs, ch, h, w = x.shape

        
        #x = self.backbone(x)
        x = self.backbone(x).pooler_output

        x1 = self.classification1(x)
        x2 = self.classification2(x)
        x3 = self.classification3(x)

        #s1 = self.scene1(x)
        s = self.scene2(x)
        #s3 = self.scene3(x)
        
        if not evaluate:
            return x1, x2, x3, s, s
        else:
            return x1, x2, x3, x

class JustResNetOOD(nn.Module):
    def __init__(self, backbone=models.resnet50(weights='ResNet50_Weights.DEFAULT'), trainset='train'):
        super().__init__()

        
        self.n_features = backbone.fc.in_features
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone.flatten = nn.Flatten(start_dim=1)

        self.budget = 0.5
        self.lamb = 0.1
        
        '''
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        '''

        if trainset in ['train', 'traintriplet']:
            self.classification1 = nn.Linear(self.n_features, 3298)
            self.classification2 = nn.Linear(self.n_features, 7202)
            self.classification3 = nn.Linear(self.n_features, 12893)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        if trainset == 'bddtrain':
            self.classification1 = nn.Linear(self.n_features, 49)
            self.classification2 = nn.Linear(self.n_features, 215)
            self.classification3 = nn.Linear(self.n_features, 520)

        self.conf1 = nn.Linear(self.n_features, 1)
        self.conf2 = nn.Linear(self.n_features, 1)   
        self.conf3 = nn.Linear(self.n_features, 1)           
        
    def forward(self, x, evaluate=False):
        bs, ch, h, w = x.shape

        x = self.backbone(x)

        x1 = F.softmax(self.classification1(x), dim=-1)
        x2 = F.softmax(self.classification2(x), dim=-1)
        x3 = F.softmax(self.classification3(x), dim=-1)

        c1 = torch.sigmoid(self.conf1(x))
        c2 = torch.sigmoid(self.conf2(x))
        c3 = torch.sigmoid(self.conf3(x))

        if not evaluate:
            return x1, x2, x3, c1, c2, c3
        else:
            return x1, x2, x3, x
class IsoMaxLoss(nn.Module):
    def __init__(self, entropic_scale = 10.0):
        super(IsoMaxLoss, self).__init__()
        self.entropic_scale = entropic_scale
    def forward(self, logits, targets):

        distances = -logits
        prob_train = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        prob_targs = prob_train[range(distances.size(0)), targets]

        loss = -torch.log(prob_targs).mean()

        return loss

class IsoMax(nn.Module):
    def __init__(self, backbone=models.resnet50(weights='ResNet50_Weights.DEFAULT'), trainset='train'):
        super().__init__()

        
        self.n_features = backbone.fc.in_features
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone.flatten = nn.Flatten(start_dim=1)

        #self.prototypes1 = nn.Parameter(torch.Tensor(3298, self.n_features))
        #self.prototypes2 = nn.Parameter(torch.Tensor(7202, self.n_features // 2))
        self.prototypes3 = nn.Parameter(torch.Tensor(12893, self.n_features))

        #self.reduce = nn.Linear(self.n_features, self.n_features // 2)

        #nn.init.normal_(self.prototypes1, mean=0.0, std=1.0)
        #nn.init.normal_(self.prototypes2, mean=0.0, std=1.0)
        nn.init.normal_(self.prototypes3, mean=0.0, std=1.0)

        #self.distance_scale1 = nn.Parameter(torch.Tensor(1))
        #self.distance_scale2 = nn.Parameter(torch.Tensor(1))
        self.distance_scale3 = nn.Parameter(torch.Tensor(1))

        #nn.init.constant_(self.distance_scale1, 1.0)
        #nn.init.constant_(self.distance_scale2, 1.0)
        nn.init.constant_(self.distance_scale3, 1.0)
        
        '''
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        '''
  
        
    def forward(self, x, evaluate=False):
        bs, ch, h, w = x.shape

        #x = self.reduce(self.backbone(x))
        x = self.backbone(x)

        '''
        distances1 = torch.abs(self.distance_scale1) * torch.cdist(
            F.normalize(x), F.normalize(self.prototypes1),
            p=2.0, compute_mode='donot_use_mm_for_euclid_dist'
        )
        distances2 = torch.abs(self.distance_scale2) * torch.cdist(
            F.normalize(x), F.normalize(self.prototypes2),
            p=2.0, compute_mode='donot_use_mm_for_euclid_dist'
        )
        '''
        distances3 = torch.abs(self.distance_scale3) * torch.cdist(
            F.normalize(x), F.normalize(self.prototypes3),
            p=2.0, compute_mode='donot_use_mm_for_euclid_dist'
        )
        #x1 = -distances1
        #x2 = -distances2
        x3 = -distances3

        if not evaluate:
            return x3, x3, x3, x3
        else:
            return x3, x3, x3, x

class GeoGuess1(nn.Module):

    def __init__(self, trainset='train'):
        super().__init__()

        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768
        
        if trainset in ['train', 'traintriplet']:
            self.classification1 = nn.Linear(self.n_features, 3298)
            self.classification2 = nn.Linear(self.n_features, 7202)
            self.classification3 = nn.Linear(self.n_features, 12893)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        if trainset == 'bddtrain':
            self.classification1 = nn.Linear(self.n_features, 49)
            self.classification2 = nn.Linear(self.n_features, 215)
            self.classification3 = nn.Linear(self.n_features, 520) 

        self.trans = CustomTransformer1(self.n_features, 6, 12, 64, 1024, dropout=0.0)
        self.queries = nn.Parameter(torch.rand(16, 3, self.n_features, requires_grad=True, device='cuda'))

    def forward(self, x, evaluate=False):
        bs, ch, h, w = x.shape

        qs = rearrange(self.queries, 'scenes hiers dim -> 1 (scenes hiers) dim').repeat(bs, 1, 1)
        x = self.backbone(x).last_hidden_state

        #print(x.get_device(), flush=True)
        #print(qs.get_device(), flush=True)

        x_out = self.trans(qs, x, x)

        x_out = rearrange(x_out, 'bs (scenes hiers) dim -> bs scenes hiers dim', hiers=3)
        scene_preds = x_out[:,:,:,0].mean(2)
        scene_choice = torch.argmax(scene_preds, dim=1)

        x_out = x_out[torch.arange(bs), scene_choice]
        
        x1 = self.classification1(x_out[:, 0])
        x2 = self.classification2(x_out[:, 1])
        x3 = self.classification3(x_out[:, 2])

        # Confidence of each hierarchy
        hier_conf = torch.sigmoid(x_out[:,:,0])

        if not evaluate:
            return x1, x2, x3, scene_preds, hier_conf
        else:
            return x1, x2, x3, x_out

class TwoScaleDecoderBlock(nn.Module):
    def __init__(self, dim = 768, heads = 12, dim_head=64, dropout=0.1, mlp_dim=1024):
        super().__init__()
        
        self.dim = dim
        self.s1_sa = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
        )
        self.s2_sa = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
        )
        self.cs_sa = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(dim * 2, heads = heads, dim_head = dim_head, dropout = dropout),
        )

        self.s1_ff = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, mlp_dim, dropout = dropout),
        )
        self.s2_ff = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, mlp_dim, dropout = dropout),
        )

        self.s1_ca_ln = nn.LayerNorm(dim)
        self.s2_ca_ln = nn.LayerNorm(dim)
        self.s1_ca_attn = CustomAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.s2_ca_attn = CustomAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)

        

    def forward(self, scale1, scale2, q1, q2):

        q1 = self.s1_sa(q1) + q1
        q2 = self.s2_sa(q2) + q2

        qx = torch.cat((q1, q2), dim=-1)
        qx = self.cs_sa(qx) + qx

        q1 = qx[:, :, :self.dim]
        q2 = qx[:, :, self.dim:]

        q1 = self.s1_ca_ln(q1)
        q2 = self.s2_ca_ln(q2)

        q1 = self.s1_ca_attn(q1, scale1, scale1) + q1
        q2 = self.s2_ca_attn(q2, scale2, scale2) + q2

        q1 = self.s1_ff(q1) + q1
        q2 = self.s2_ff(q2) + q2

        return q1, q2


class GeoGuess2(nn.Module):
    def __init__(self, decoder_depth = 5, trainset='train'):
        super().__init__()

        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k", output_hidden_states=True)
        self.n_features = 768
        
        if trainset in ['train', 'traintriplet']:
            self.classification1 = nn.Linear(self.n_features * 2, 3298)
            self.classification2 = nn.Linear(self.n_features * 2, 7202)
            self.classification3 = nn.Linear(self.n_features * 2, 12893)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features * 2, 689)
            self.classification2 = nn.Linear(self.n_features * 2, 689)
            self.classification3 = nn.Linear(self.n_features * 2, 689)
        if trainset == 'bddtrain':
            self.classification1 = nn.Linear(self.n_features * 2, 49)
            self.classification2 = nn.Linear(self.n_features * 2, 215)
            self.classification3 = nn.Linear(self.n_features * 2, 520) 

        self.q1 = nn.Parameter(torch.rand(16, 3, self.n_features, requires_grad=True, device='cuda'))
        self.q2 = nn.Parameter(torch.rand(16, 3, self.n_features, requires_grad=True, device='cuda'))

        self.decoders = nn.ModuleList([])
        for _ in range(decoder_depth):
            TwoScaleDecoderBlock(dim=self.n_features, heads=12, dim_head=64, dropout=0.1, mlp_dim=1024)

    def forward(self, x, evaluate=False):
        bs, ch, h, w = x.shape

        q1 = rearrange(self.q1, 'scenes hiers dim -> 1 (scenes hiers) dim').repeat(bs, 1, 1)
        q2 = rearrange(self.q2, 'scenes hiers dim -> 1 (scenes hiers) dim').repeat(bs, 1, 1)

        x = self.backbone(x).hidden_states
        scale1 = x[-3]
        scale2 = x[-1]

        for decoder in self.decoders:
            q1, q2 = decoder(scale1, scale2, q1, q2)


        x1_out = rearrange(q1, 'bs (scenes hiers) dim -> bs scenes hiers dim', hiers=3)
        x2_out = rearrange(q2, 'bs (scenes hiers) dim -> bs scenes hiers dim', hiers=3)

        scene_preds1 = x1_out[:,:,:,0].mean(2)
        scene_preds2 = x2_out[:,:,:,0].mean(2)

        scene_choice1 = torch.argmax(scene_preds1, dim=1)
        scene_choice2 = torch.argmax(scene_preds2, dim=1)

        x1_out = x1_out[torch.arange(bs), scene_choice1]
        x2_out = x2_out[torch.arange(bs), scene_choice2]

        x_out = torch.cat((x1_out, x2_out), dim=-1)
        scene_preds = (scene_preds1 + scene_preds2) / 2
        
        x1 = self.classification1(x_out[:, 0])
        x2 = self.classification2(x_out[:, 1])
        x3 = self.classification3(x_out[:, 2])

        # Confidence of each hierarchy
        hier_conf = torch.sigmoid(x_out[:,:,0])

        if not evaluate:
            return x1, x2, x3, scene_preds, hier_conf
        else:
            return x1, x2, x3, x_out

class GeoGuess3(nn.Module):

    def __init__(self, trainset='train'):
        super().__init__()

        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k", output_hidden_states=True)
        self.n_features = 1024
        
        if trainset in ['train', 'traintriplet']:
            self.classification1 = nn.Linear(self.n_features, 3298)
            self.classification2 = nn.Linear(self.n_features, 7202)
            self.classification3 = nn.Linear(self.n_features, 12893)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        if trainset == 'bddtrain':
            self.classification1 = nn.Linear(self.n_features, 49)
            self.classification2 = nn.Linear(self.n_features, 215)
            self.classification3 = nn.Linear(self.n_features, 520) 

        self.trans = CustomTransformer1(self.n_features, 6, 12, 64, 1024, dropout=0.0)
        self.queries = nn.Parameter(torch.rand(16, 3, self.n_features, requires_grad=True, device='cuda'))

        self.ppnorm = nn.LayerNorm(self.n_features // 2)
        self.project = nn.Linear(self.n_features // 2, self.n_features)

    def forward(self, x, evaluate=False):
        bs, ch, h, w = x.shape

        qs = rearrange(self.queries, 'scenes hiers dim -> 1 (scenes hiers) dim').repeat(bs, 1, 1)
        x = self.backbone(x).hidden_states

        x1 = x[-1]
        x2 = self.project(self.ppnorm(x[-3]))
        x = torch.cat((x1, x2), dim=1)

        #print(x.get_device(), flush=True)
        #print(qs.get_device(), flush=True)

        x_out = self.trans(qs, x, x)

        x_out = rearrange(x_out, 'bs (scenes hiers) dim -> bs scenes hiers dim', hiers=3)
        scene_preds = x_out[:,:,:,0].mean(2)
        scene_choice = torch.argmax(scene_preds, dim=1)

        x_out = x_out[torch.arange(bs), scene_choice]
        
        x1 = self.classification1(x_out[:, 0])
        x2 = self.classification2(x_out[:, 1])
        x3 = self.classification3(x_out[:, 2])

        # Confidence of each hierarchy
        hier_conf = torch.sigmoid(x_out[:,:,0])

        if not evaluate:
            return x1, x2, x3, scene_preds, hier_conf
        else:
            return x1, x2, x3, x_out
class MixTransformerDeTR(nn.Module):
    def __init__(self, backbone=models.resnet101(weights='ResNet101_Weights.DEFAULT'), trainset='train'):
        super().__init__()

        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_features = 768

        self.detr_model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
        for param in self.detr_model.parameters():
            param.requires_grad = False
        
        self.n_features = 768
        self.expand = nn.Linear(256, self.n_features)

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

        self.queries = torch.rand(6, 768).cuda()

        self.trans = CustomTransformer2(768, 4, 12, 64, 1024)

    def forward(self, x, evaluate=False):
        bs, ch, h, w = x.shape

        imgout = self.backbone(x).last_hidden_state
        detrout = self.expand(self.detr_model(x).last_hidden_state)
        qs = self.queries.unsqueeze(0).repeat(bs, 1, 1)
        
        #print(qs.shape, imgout.shape, detrout.shape)
        x = self.trans(qs, imgout, detrout)
        x1 = self.classification1(x[:,0])
        x2 = self.classification2(x[:,1])
        x3 = self.classification3(x[:,2])
        s1 = self.scene1(x[:,3])
        s2 = self.scene2(x[:,4])
        s3 = self.scene3(x[:,5])


        if not evaluate:
            return x1, x2, x3, s1, s2, s3
        else:
            return x1, x2, x3
        
class MMFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn1 = CustomAttention(768, 12, 64)
        self.attn2 = CustomAttention(768, 12, 64)

        self.w = nn.Linear(768*2, 768)
    def forward(self, rgb, seg):
        
        r2s = self.attn1(rgb, seg, seg)
        s2r = self.attn2(seg, rgb, rgb)

        rgb_cls = r2s[:, 0, :]
        s2r_cls = s2r[:, 0, :]

        mixed = self.w(torch.cat((rgb_cls, s2r_cls), 1))

        return mixed
    
class Translocator(nn.Module):
    def __init__(self, trainset='train'):
        super().__init__()

        
        self.rgb_backbone = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', weight_init='skip', embed_dim=768)
        self.rgb_backbone.load_state_dict(torch.load("weights/vit.pth"), strict=False)

        self.proj_seg = nn.Linear(150, 3)

        self.seg_backbone = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', weight_init='skip', embed_dim=768)
        self.seg_backbone.load_state_dict(torch.load("weights/vit.pth"), strict=False)

        seg_config = HRNET_48
        self.hrnet = get_seg_model(seg_config)

        for param in self.hrnet.parameters():
            param.requires_grad = False

        self.n_features = 768

        self.fusion_list = nn.ModuleList([MMFusion() for i in range(12)])

        if trainset in ['train', 'traintriplet']:
            self.classification1 = nn.Linear(self.n_features, 3298)
            #self.classification2 = nn.Linear(self.n_features, 7202)
            self.classification2 = FeedForward(dim=self.n_features, hidden_dim=3000, dropout = 0.1, dropout2 = 0.0, out_dim=7202)
            #self.classification3 = nn.Linear(self.n_features, 12893)
            self.classification3 = FeedForward(dim=self.n_features, hidden_dim=6000, dropout = 0.1, dropout2 = 0.0, out_dim=12893)
        if trainset == 'train1M':
            self.classification1 = nn.Linear(self.n_features, 689)
            self.classification2 = nn.Linear(self.n_features, 689)
            self.classification3 = nn.Linear(self.n_features, 689)
        if trainset == 'bddtrain':
            self.classification1 = nn.Linear(self.n_features, 49)
            self.classification2 = nn.Linear(self.n_features, 215)
            self.classification3 = nn.Linear(self.n_features, 520)  

        self.scene = nn.Linear(768, 16)     

        
    def forward(self, x, evaluate=False):
        bs, ch, h, w = x.shape

        seg_map = self.hrnet(x)[1]

        # Scale up the segmentation map
        seg_map = rearrange(F.interpolate(seg_map, scale_factor=4), 'bs ch h w -> bs h w ch')

        # Reduce the classes of the segmentation map down to 3 channels
        seg_map = rearrange(self.proj_seg(seg_map), 'bs h w ch -> bs ch h w')

        rgb = self.rgb_backbone.forward_features(x)
        seg = self.seg_backbone.forward_features(seg_map)
        
        for i in range(12):
            seg = self.seg_backbone.blocks[i](seg)
            rgb = self.rgb_backbone.blocks[i](rgb)

            fuse = self.fusion_list[i](rgb, seg)


            seg = torch.cat((seg[:,1:,], fuse.unsqueeze(1)), 1)
            rgb = torch.cat((rgb[:,1:,], fuse.unsqueeze(1)), 1)

        x1 = self.classification1(fuse)
        x2 = self.classification2(fuse)
        x3 = self.classification3(fuse)

        scene = self.scene(fuse)
        
        if not evaluate:
            return x1, x2, x3, scene
        else:
            return x1, x2, x3, x
        
        return rgb

if __name__ == '__main__':

    x = torch.rand(32, 3, 224, 224).cuda()

    
    model = GeoGuess2()
    _ = model.to("cuda")
    x1, x2, x3, scene_preds, hier_conf = model(x)
    print(x1.shape)