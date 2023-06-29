from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os.path as osp
from config import cfg


class PairwiseDataset(Dataset):
    def __init__(self, dataset_root, dataframe, transform=None):
        self.dataset_root = dataset_root
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path1 = osp.join(self.dataset_root, self.dataframe.iloc[index, 0])
        img_path2 = osp.join(self.dataset_root, self.dataframe.iloc[index, 1])

        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2


def transform(args, split='train'):
    
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.crop((args.width_crop, 0, img.size[0], img.size[1]))),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.crop((args.width_crop, 0, img.size[0], img.size[1]))),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    if split == 'train':
        return train_transform
    
    return val_transform