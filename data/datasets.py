# import cv2
import os
import sys

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image

NUM_DATASET_WORKERS = 4
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        dir = data_dir
        self.imgs += glob(os.path.join(dir, "*.jpg"))
        self.imgs += glob(os.path.join(dir, "*.png"))
        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert("RGB")
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose(
            [transforms.CenterCrop((self.im_width, self.im_height)), transforms.ToTensor()]
        )
        img = self.transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


class Datasets_train(Dataset):
    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir
        self.imgs = []
        dir = data_dir
        self.img_size = img_size
        self.imgs += glob(os.path.join(dir, "*.jpg"))
        self.imgs += glob(os.path.join(dir, "*.png"))
        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert("RGB")
        self.im_height, self.im_width = image.size
        if self.im_height < self.img_size or self.im_width < self.img_size:
            crop_size = self.img_size
            self.transform = transforms.Compose(
                [transforms.Resize((crop_size, crop_size)), transforms.ToTensor()]
            )
        else:
            crop_size = self.img_size
            self.transform = transforms.Compose(
                [transforms.RandomCrop((crop_size, crop_size)), transforms.ToTensor()]
            )
        img = self.transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


def get_loader(config):

    if config.DATA.DATASET == "CIFAR10":
        transform_train = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        )

        transform_test = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(
            root=config.DATA.train_data_dir, train=True, transform=transform_train, download=False
        )

        test_dataset = datasets.CIFAR10(
            root=config.DATA.test_data_dir, train=False, transform=transform_test, download=False
        )
    elif config.DATA.DATASET == "DIV2K":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),  #
                transforms.ToTensor(),
            ]
        )

        train_dataset = datasets.ImageFolder(
            root=config.DATA.train_data_dir,
            transform=transform_train,
        )
        # test_dataset = Datasets(data_dir=config.DATA.test_data_dir)
        test_dataset = datasets.ImageFolder(
            root=config.DATA.test_data_dir, transform=transform_test
        )
    elif config.DATA.DATASET in ["CelebA", "CelebA-HQ", "AFHQ", "Bird"]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.ImageFolder(
            root=config.DATA.train_data_dir, transform=transform_train
        )

        test_dataset = datasets.ImageFolder(
            root=config.DATA.test_data_dir, transform=transform_test
        )

    elif config.DATA.DATASET in ["Kodak", "CLIC2021", "OpenImg"]:
        #
        transform_train = transforms.Compose(
            [
                transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        #
        train_dataset = Datasets_train(
            data_dir=config.DATA.train_data_dir, img_size=config.DATA.IMG_SIZE
        )

        test_dataset = Datasets(config.DATA.test_data_dir)
    # print(NUM_DATASET_WORKERS)
    # seed_torch()
    if config.TRAIN.DATA_PARALLEL:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler_train,
            batch_size=config.DATA.TRAIN_BATCH,
            num_workers=NUM_DATASET_WORKERS,  # config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=NUM_DATASET_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        batch_size=config.DATA.TRAIN_BATCH,
        # worker_init_fn=worker_init_fn_seed,
        shuffle=True,
        drop_last=False,
    )

    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=config.DATA.TEST_BATCH, shuffle=False
    )

    return train_loader, test_loader



