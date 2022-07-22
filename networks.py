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
from transformers import ViTModel, DetrForSegmentation
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

        self.scene1 = nn.Linear(self.n_features, 3)  
        self.scene2 = nn.Linear(self.n_features, 16)
        self.scene3 = nn.Linear(self.n_features, 365)        
        
    def forward(self, x, evaluate=False):
        #bs, ch, h, w = x.shape

        
        x = self.backbone(x)

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
    def __init__(self, backbone=models.resnet50(pretrained=True), trainset='train'):
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

        c1 = F.sigmoid(self.conf1(x))
        c2 = F.sigmoid(self.conf2(x))
        c3 = F.sigmoid(self.conf3(x))

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
    def __init__(self, backbone=models.resnet50(pretrained=True), trainset='train'):
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
    def __init__(self, backbone=models.resnet50(pretrained=True), trainset='train'):
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

        self.trans = CustomTransformer1(self.n_features, 6, 12, 64, 1024, dropout=0.1)
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
        hier_conf = F.sigmoid(x_out[:,:,0])

        if not evaluate:
            return x1, x2, x3, scene_preds, hier_conf
        else:
            return x1, x2, x3, x_out

class MixTransformerDeTR(nn.Module):
    def __init__(self, backbone=models.resnet101(pretrained=True), trainset='train'):
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