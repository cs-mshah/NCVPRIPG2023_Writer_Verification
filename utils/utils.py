import torch
import numpy as np
import shutil
import random
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def select_device(threshold=2) -> torch.device:
    ''' 
    returns a single GPU with the most free memory above the threshold
    threshold(in GB)
    '''
    threshold *= 1e9
    device = -1
    free_mem = 0
    is_gpu = False
    
    for i in range(torch.cuda.device_count()):
        device_properties = torch.cuda.get_device_properties(i)
        free_memory = device_properties.total_memory -  torch.cuda.memory_allocated(i)
        if free_memory > threshold and free_memory > free_mem:
            free_mem = free_memory
            device = i
            is_gpu = True
    
    if device != -1:
        device = torch.device("cuda:"+str(device) if is_gpu else "cpu")
        print(f"Device {device} is selected and having {free_mem/(1024**2)}MB memory")
    else:
        device = torch.device("cpu")
        print("No GPU device available for selection. Returning CPU")
    return device


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class Losses:
    '''Create a class for all losses and iterate over them'''
    def __init__(self):
        self.losses_dict = {}

    def add_loss(self, name:str):
        self.losses_dict[name] = AverageMeter()

    def update(self, name:str, value, n=1):
        self.losses_dict[name].update(value, n)

    def reset(self):
        for loss in self.losses_dict.values():
            loss.reset()

    def get_averages(self):
        return {name: loss.avg for name, loss in self.losses_dict.items()}

    def __len__(self):
        return len(self.losses_dict)

    def __getitem__(self, name):
        return self.losses_dict[name]

    def __iter__(self):
        return iter(self.losses_dict)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def describe_dataloader(dataloader: torch.utils.data.DataLoader):
    """
    Prints basic information about a PyTorch DataLoader
    """
    num_samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    sample = next(iter(dataloader))
    print(f'{dataloader.dataset.__class__.__name__} {dataloader.__class__.__name__} with {num_samples} samples, batch size {batch_size}, {num_batches} batches.')
    print(f'dataloader sample: {sample}')
    

def save_checkpoint(state, save_path, is_best=False, tag='base'):
    filename=f'checkpoint_{tag}.pth.tar'
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, f'model_best_{tag}.pth.tar'))


def get_optimizer_and_scheduler(args, model_parameters):
    # Create the optimizer
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Invalid optimizer type. Must be 'sgd' or 'adam'.")
    
    # Create the learning rate scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.mode, factor=args.factor, patience=args.patience)
    elif args.scheduler == 'lambda':
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    elif args.scheduler == 'constant':
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 1)
    else:
        raise ValueError("Invalid scheduler type. Must be 'step', 'cosine', 'lambda', 'constant' or 'reduce_on_plateau'.")
    
    return optimizer, scheduler

if __name__ == '__main__':
    pass