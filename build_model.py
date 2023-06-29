import os
from typing import Tuple, Optional, List, Dict
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import models
from models import Classifier as ClassifierBase


model_urls = {
    'resnet50': 
    {
        'swav': 'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',
        'dino': 'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth'
    }
}


class ImageClassifier(ClassifierBase):
    """lr of backbone is 0.1 * lr of bottleneck and head. set finetune=false for same lr"""
    def __init__(self, backbone: nn.Module, num_classes: Optional[int], head: Optional[nn.Module] = None, **kwargs):
        super(ImageClassifier, self).__init__(backbone, num_classes, head=head, **kwargs)

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params



def modify_state_dict(state_dict, func, s='encoder.'):
    """returns new_stat_dict according to func and string s={'encoder.', 'module.'}
    func={remove_prefix, add_prefix}"""
    def remove_prefix(key):
        return key[len(s):] if key.startswith(s) else key
    
    def add_prefix(key):
        return s + key if (not key.startswith(s)) else key
    
    modification_functions = {
        'remove_prefix': remove_prefix,
        'add_prefix': add_prefix
    }
    
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = modification_functions[func](key)
        new_state_dict[new_key] = value
    
    return new_state_dict


def model_keys_diff(model, pretrained_weights):
    """prints the difference in keys between model and pretrained weights"""
    keys_model = set(model.state_dict().keys())
    keys_pretrained = set(pretrained_weights.keys())
    diff_keys1 = keys_model - keys_pretrained
    diff_keys2 = keys_pretrained - keys_model
    print(f"Keys in model but not in pretrained model: {diff_keys1}")
    print(f"Keys in pretrained model but not in model: {diff_keys2}")


def get_backbone(args, verbose=False):
    """returns a backbone model loaded with pretrained weights if passed as an argument"""
    if args.arch in models.__dict__:
        backbone = models.__dict__[args.arch](pretrained=False)
        if 'swav' in args.pretrained:
            ckpt = torch.hub.load_state_dict_from_url(model_urls[args.arch]['swav'], model_dir='pretrained')
            ckpt = modify_state_dict(ckpt, 'remove_prefix', 'module.')
            backbone.load_state_dict(ckpt, strict=False)
        if args.pretrained == 'dino':
            ckpt = torch.hub.load_state_dict_from_url(model_urls[args.arch]['dino'], model_dir='pretrained')
            backbone.load_state_dict(ckpt, strict=False)
        if verbose:
            model_keys_diff(backbone, ckpt)
            # print(summary(backbone))
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(args.arch, pretrained=True)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def build_model(args, verbose=False):
    """return model depending on the method"""
    backbone = get_backbone(args, verbose=verbose)
    model = ImageClassifier(backbone, args.embed_dim)
    
    return model