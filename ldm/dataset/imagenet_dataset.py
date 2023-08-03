
import os.path as osp
import json
import requests
import time
import numpy as np
import io
from PIL import Image
import logging
import random
from torch.utils.data import Dataset
from torchvision import  transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
logger = logging.getLogger('global')


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.initialized = False

        with open(meta_file) as f:
            lines = f.readlines()
        self.num = len(lines)
        metas_names = []
        metas_labels = []
        for line in lines:
            filename, label = line.rstrip().split()
            metas_names.append(osp.join(self.root_dir, filename))
            metas_labels.append(int(label))
        self.metas_names = np.string_(metas_names)
        self.metas_labels = np.int_(metas_labels)
        self.initialized = False

    def _init_ceph(self):
        from petrel_client.client import Client as CephClient
        if not self.initialized:
            self.mclient = CephClient()
            self.initialized = True

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        self._init_ceph()
        try:
            img_path = str(self.metas_names[idx], encoding='utf-8')
            label = self.metas_labels[idx]
            value = self.mclient.Get(img_path)
            img_bytes = np.fromstring(value, np.uint8)
            buff = io.BytesIO(img_bytes)
            with Image.open(buff) as img:
                img = img.convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, label
        except Exception as e:
            logger.info(f'Error when load {idx}')
            logger.info(e)
            return self.__getitem__(random.randint(0, len(self.metas_names) - 1))


def build_imagenet_for_cls(is_train, args):
    transform = build_transform(is_train, args)
    if is_train:
        dataset = ImageNetDataset(
            root_dir=args.root_dir_train,
            meta_file=args.meta_file_train,
            transform=transform,
        )
    else:
        dataset = ImageNetDataset(
            root_dir=args.root_dir_val,
            meta_file=args.meta_file_val,
            transform=transform,
        )
    nb_classes = 1000
    

    return dataset, nb_classes

def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def build_imagenet_for_vqvae(is_train, args):
    if is_train:
        t = []
        if args.color_jitter > 0:
            t.append(transforms.ColorJitter(args.color_jitter,args.color_jitter,args.color_jitter))
        t.append(transforms.RandomResizedCrop(args.input_size, scale=(args.min_crop_scale, 1.0), interpolation=_pil_interp(args.train_interpolation)))
        t.append(transforms.RandomHorizontalFlip(0.5))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)

    else:
        t = []
        if args.input_size < 384:
            args.crop_pct = 224 / 256
        else:
            args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=_pil_interp(args.train_interpolation)),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)
    print(f"{'Train' if is_train else 'Test'} Data Aug: {str(transform)}")
    if is_train:
        dataset = ImageNetDataset(
            root_dir=args.root_dir_train,
            meta_file=args.meta_file_train,
            transform=transform,
        )
    else:
        dataset = ImageNetDataset(
            root_dir=args.root_dir_val,
            meta_file=args.meta_file_val,
            transform=transform,
        )
    return dataset
    


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    test_size = args.input_size
    crop = test_size < 320
    if resize_im:
        if crop:
            size = int((256 / 224) * test_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(test_size))
        else:
            t.append(
                transforms.Resize((test_size,test_size), interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
            