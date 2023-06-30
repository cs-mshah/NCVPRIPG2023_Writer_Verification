import os
import numpy as np
import pandas as pd
import argparse
from pprint import pprint
from tqdm import tqdm
import os.path as osp
from datetime import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from pytorch_metric_learning import distances, losses, miners, reducers, samplers

from sklearn.metrics import f1_score, roc_auc_score
from config import cfg
import wandb
from build_model import build_model
from utils.utils import set_seed, get_optimizer_and_scheduler, Losses, save_checkpoint
from datasets import PairwiseDataset, transform
import lovely_tensors as lt
lt.monkey_patch()


def load_train_objs(args):
    """returns the dataset, model"""

    model = build_model(args)
    model = torch.compile(model)
    train_transform = transform(args, split='train')
    
    train_dataset = datasets.ImageFolder(osp.join(args.dataset_dir, 'train'), transform=train_transform)
    
    if args.sampler == 'MPerClassSampler':
        sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset))
    else:
        sampler = RandomSampler(train_dataset)
    
    return train_dataset, sampler, model


def train(args, model, train_loader, optimizer, scheduler, epoch, device):
    model.train()
    
    ### pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=args.margin, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=args.margin, distance=distance, type_of_triplets=args.type_of_triplets)
    ### pytorch-metric-learning stuff ###
    
    all_losses = Losses()
    all_losses.add_loss('triplet_margin_loss')
    
    for batch_idx, (data, labels) in tqdm(enumerate(train_loader)):
        data, labels = data.to(device), labels.to(device)
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        
        optimizer.zero_grad()
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )
        if mining_func.num_triplets:
            all_losses.update('triplet_margin_loss', loss.item(), mining_func.num_triplets)

    loss_averages = all_losses.get_averages()
    for name, avg in loss_averages.items():
        # if name != 'losses':
        wandb.log({f'train/{name}': avg}, commit=False)


def test(cfg, model, test_loader, epoch, device):
    model.eval()
    Y_true = pd.read_csv(osp.join(cfg.dataset_dir, 'val.csv'))
    y_true = Y_true['label'].values
    
    criterion = torch.nn.CosineSimilarity().to(device)
    sim_list = []
    # with torch.inference_mode():
    with torch.no_grad():
        for batch_idx, (batch1, batch2) in tqdm(enumerate(test_loader)):
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            
            f1 = model(batch1)
            f2 = model(batch2)
        
            similarity = criterion(f1, f2)
            sim_list.extend(similarity.cpu().numpy())
    
    sim_list = np.array(sim_list)

    y_pred = sim_list >= 0.5
    y_pred = y_pred.astype(int)
    test_f1 = f1_score(y_true, y_pred)
    test_auc = roc_auc_score(y_true, sim_list)
    print(f'Test f1: {test_f1}, Test AUC: {test_auc}')
    return test_f1, test_auc


def main():
    parser = argparse.ArgumentParser(description='NCVPRIG23 Writer Verification')
    parser.add_argument('--config', required=False, type=str, help='path to yaml config')
    args = parser.parse_args()
    if args.config:
        cfg.merge_from_file(args.config)

    cfg.run_started = datetime.today().strftime("%d-%m-%y_%H%M")
    device = torch.device('cuda', 0)
    set_seed(cfg.rng_seed)
    cfg.out = osp.join(cfg.out, f'{cfg.run_started}')
    os.makedirs(cfg.out, exist_ok=True)
    
    # wandb init
    wandb.init(entity='manan-shah',
               project='NCVPRIG23 Writer Verification',
               config=cfg,
               id=None,
               resume='allow',
               allow_val_change=True,
               settings=wandb.Settings(start_method="fork"))
    
    train_dataset, sampler, model = load_train_objs(cfg)
    model.to(device)
    
    # validation dataset
    df = pd.read_csv(osp.join(cfg.dataset_dir, 'val.csv'))
    val_transform = transform(cfg, split='val')
    val_dataset = PairwiseDataset(osp.join(cfg.dataset_dir, 'val'), df, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch, num_workers=cfg.num_workers, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
    
    optimizer, scheduler = get_optimizer_and_scheduler(cfg, model.get_parameters())
    
    start_epoch = 0
    best_auc = 0
    
    if cfg.resume:
        assert os.path.isfile(
            cfg.resume), "Error: no checkpoint directory found!"
        checkpoint = torch.load(cfg.resume)
        start_epoch = checkpoint['epoch']
        load_state_dict = checkpoint['state_dict']
        model.load_state_dict(load_state_dict, strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        auc = checkpoint['auc']
        print(f'resuming from checkpoint. Epoch: {start_epoch}, AUC: {auc}\n')

    for epoch in range(start_epoch, cfg.epochs):
        print(f'embedder_lr: {scheduler.get_last_lr()[1]}')
        train(cfg, model, train_loader, optimizer, scheduler, epoch, device)
        test_f1, test_auc = test(cfg, model, val_loader, epoch, device)
        
        wandb.log({'train/embedder_lr': scheduler.get_last_lr()[1]}, commit=False)
        wandb.log({'test/f1': test_f1}, commit=False)
        wandb.log({'test/auc': test_auc})
        
        is_best = test_auc > best_auc

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'f1': test_f1,
            'auc': test_auc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, cfg.out, is_best, tag='base')
    
    print(f'Best AUC: {best_auc}')
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='7'
    main()