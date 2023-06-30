import os
from yacs.config import CfgNode as CN
import argparse

# global config object
_C = CN()

# ----------------configuration options------------------------ #

# number of workers for Dataloader (set according to )
_C.num_workers = 24
# random seed
_C.rng_seed = 42
# configuration directory
_C.cfg_dir = 'cfgs'
# dataset directory
_C.dataset_dir = 'data/raw'
# output directory
_C.out = 'output'
# base configuration yaml file
_C.cfg_base = 'cfg.yaml'
# resume checkpoint
_C.resume = ''
# train batch size
_C.train_batch = 128
# sampler
_C.sampler = 'MPerClassSampler'
# val batch size
_C.val_batch = 32
# image size
_C.img_size = 224
# initial width crop
_C.width_crop = 0
# optimizer
_C.optimizer = 'adam'
# lr
_C.lr = 0.01
# weight decay
_C.weight_decay = 0.0
# scheduler
_C.scheduler = 'lambda'
# lr_gamma
_C.lr_gamma = 0.001
# lr_decay
_C.lr_decay = 0.75

# model
_C.arch = 'resnet50'
# pretrained weights
_C.pretrained = 'dino'
# linear embedding dimension
_C.embed_dim = 64

# loss params
# margin
_C.margin = 0.2
# type_of_triplets
_C.type_of_triplets = 'all'

# number of epochs
_C.epochs = 10

# ----------------default config-------------------------------- #

# import the defaults as a global singleton:
cfg = _C  # `from config import cfg`

_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

def dump_cfg(config_name='cfg.yaml'):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.cfg_dir, config_name)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCVPRIG23 Writer Verification Config Dump')
    parser.add_argument('--name', default='cfg.yaml', required=False, type=str, help='path to yaml config')
    args = parser.parse_args()
    
    dump_cfg(args.name)