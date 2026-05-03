import os
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, img_key, transform):
        self.data = hf_dataset
        self.img_key = img_key
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item[self.img_key]
        label = item.get('label', item.get('fine_label'))
        return self.transform(img), label


def get_dataset(name, img_size=32, batch_size=128):
    hf_map = {
        'cifar10':       ('cifar10',       'img',   10),
        'cifar100':      ('cifar100',      'img',  100),
        'mnist':         ('mnist',         'image', 10),
        'fashion_mnist': ('fashion_mnist', 'image', 10),
    }

    hf_name, img_key, num_classes = hf_map[name]

    # MNIST and Fashion-MNIST are grayscale, convert to 3 channels
    if name in ['mnist', 'fashion_mnist']:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    train_ds = load_dataset(hf_name, split='train')
    val_ds   = load_dataset(hf_name, split='test')

    train_loader = DataLoader(
        HFDatasetWrapper(train_ds, img_key, transform),
        batch_size=batch_size, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        HFDatasetWrapper(val_ds, img_key, transform),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, num_classes


if __name__ == '__main__':
    # Test with img_size=32
    print("Testing img_size=32 (L=64):")
    for name in ['cifar10', 'cifar100', 'mnist', 'fashion_mnist']:
        train_loader, val_loader, nc = get_dataset(name, img_size=32)
        x, y = next(iter(train_loader))
        print(f"  {name:15s} | classes={nc:3d} | batch={list(x.shape)}")

    # Test with img_size=64
    print("\nTesting img_size=64 (L=256):")
    for name in ['cifar10']:
        train_loader, val_loader, nc = get_dataset(name, img_size=64)
        x, y = next(iter(train_loader))
        print(f"  {name:15s} | classes={nc:3d} | batch={list(x.shape)}")