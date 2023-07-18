import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build_model import build_model, modify_state_dict
from config import cfg
from datasets import PairwiseDataset, transform


def submit_file(df, proba):
    """create a csv in submission format"""
    out_df = pd.DataFrame()
    out_df['id'] = df['img1_name'] + '_' + df['img2_name']
    out_df['proba'] = pd.Series(proba)
    return out_df


def normalise_cosine_similarity(arr, do=True):
    """transform range of [-1, +1] to [0, 1]"""
    if do:
        arr = (arr + 1.0) / 2
    return arr


def test(mode, model, test_loader, device, y_true=None):
    model.eval()
    
    criterion1 = torch.nn.CosineSimilarity().to(device)
    sim_list_unnormalized = []
    sim_list_normalized = []
    # with torch.inference_mode():
    with torch.no_grad():
        for batch_idx, (batch1, batch2) in enumerate(test_loader):
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            
            f1 = model(batch1)
            f2 = model(batch2)
        
            similarity = criterion1(f1, f2)
            sim_list_unnormalized.extend(similarity.cpu().numpy())
            
            f1 = F.normalize(f1)
            f2 = F.normalize(f2)
            similarity = criterion1(f1, f2)
            sim_list_normalized.extend(similarity.cpu().numpy())
    
    sim_list_unnormalized = normalise_cosine_similarity(np.array(sim_list_unnormalized))
    sim_list_normalized = normalise_cosine_similarity(np.array(sim_list_normalized))
    test_auc_unnormalized = test_auc_normalized = 0.0
    if mode == 'val':
        test_auc_unnormalized = roc_auc_score(y_true, sim_list_unnormalized)
        test_auc_normalized = roc_auc_score(y_true, sim_list_normalized)
        # print(f'AUC Unnormalized: {test_auc_unnormalized}\nAUC Normalized: {test_auc_normalized}')
    
    return sim_list_unnormalized, sim_list_normalized, test_auc_unnormalized, test_auc_normalized


def bulk_metrics(root: str, model, device):
    """view AUC of models on val split and save proba of val, test split
    inside a root directory
    """
    df_val = pd.read_csv(osp.join(cfg.dataset_dir, f'val.csv'))
    df_test = pd.read_csv(osp.join(cfg.dataset_dir, f'test.csv'))
    y_true = df_val['label'].values
    val_transform = transform(cfg, split='val')
    val_dataset = PairwiseDataset(osp.join(cfg.dataset_dir, 'val'), df_val, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
    
    test_dataset = PairwiseDataset(osp.join(cfg.dataset_dir, 'semi_test'), df_test, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.val_batch, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
    checkpoint_name = 'model_best_base.pth.tar'
    # checkpoint_name = 'checkpoint_base.pth.tar'
    
    for dirpath, dirnames, filenames in os.walk(root):
        if (checkpoint_name in filenames)  and ('test_proba.csv' not in filenames):
            checkpoint = torch.load(osp.join(dirpath, checkpoint_name))
            epoch = checkpoint['epoch']
            load_state_dict = checkpoint['state_dict']
            load_state_dict = modify_state_dict(load_state_dict, 'remove_prefix', '_orig_mod.')
            model.load_state_dict(load_state_dict, strict=True)
            auc = checkpoint['auc']
            
            sim_list_unnormalized, sim_list_normalized, val_auc_unnormalized, val_auc_normalized = \
                test('val', model, val_loader, device, y_true)
            print(f'{dirpath} Epoch: {epoch}, Saved AUC: {auc}, Val AUC Unnormalized: {val_auc_unnormalized},  Val AUC Normalized: {val_auc_normalized}\n')
            
            sim_list_unnormalized, sim_list_normalized, _, _ = \
                test('test', model, test_loader, device)
            df_submit = submit_file(df_test, sim_list_normalized)
            df_submit.to_csv(osp.join(dirpath, 'test_proba.csv'), index=False)
            # break
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCVPRIG23 Writer Verification')
    parser.add_argument('--config', required=False, type=str, help='path to yaml config')
    parser.add_argument('--mode', default='test', required=False, type=str, help='inference mode. (val/test)')
    parser.add_argument('--ckpt', default='pretrained/model_best_base.pth.tar', type=str, help='path to saved checkpoint')
    parser.add_argument('--output', default='test_proba.csv', type=str, help='csv output path for test set')
    
    args = parser.parse_args()
    if args.config:
        cfg.merge_from_file(args.config)

    os.environ['CUDA_VISIBLE_DEVICES']='7'
    device = torch.device('cuda', 0)
    
    model = build_model(cfg)
    model.to(device)

    if args.mode == 'bulk': # custom mode to evaluate multiple checkpoints
        bulk_metrics(cfg.out, model, device)
        exit()
    
    # load the checkpoint
    checkpoint = torch.load(args.ckpt)
    load_state_dict = checkpoint['state_dict']
    load_state_dict = modify_state_dict(load_state_dict, 'remove_prefix', '_orig_mod.')
    model.load_state_dict(load_state_dict, strict=True)
    
    csv_file = f'finale-test-pairs.csv'
    df = pd.read_csv(osp.join(cfg.dataset_dir, csv_file)) # make sure this is the test set csv
    val_transform = transform(cfg, split='val')
    split_dir = 'test' # change this (semi_test or test)
    y_true = None
    if args.mode == 'val':
        split_dir = 'val'
        y_true = df['label'].values
    dataset = PairwiseDataset(osp.join(cfg.dataset_dir, split_dir), df, val_transform)
    dataloader = DataLoader(dataset, batch_size=cfg.val_batch, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
    
    sim_list_unnormalized, _, auc_unnormalized, _ = \
        test(args.mode, model, dataloader, device, y_true)

    if args.mode == 'val':
        print(f'Val AUC: {auc_unnormalized}')
    df_submit = submit_file(df, sim_list_unnormalized)
    df_submit.to_csv(args.output, index=False)
    