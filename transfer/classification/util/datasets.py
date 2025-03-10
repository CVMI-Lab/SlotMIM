# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import json
from functools import wraps
from pathlib import Path
from torchvision.datasets import ImageFolder

def file_cache(filepath_func):
    """Decorator to cache the output of a function to disk."""
    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path(filepath_func(directory))
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                if not filepath.parent.is_dir():
                    filepath.parent.mkdir(exist_ok=True)
                filepath.write_text(json.dumps(out))
            return out
        return decorated
    return decorator

class CachedImageFolder(ImageFolder):
    @file_cache(filepath_func=lambda directory: f"./cache/{Path(directory).parent.name}_{Path(directory).name}_cached_classes.json")
    def find_classes(self, directory, *args, **kwargs):
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filepath_func=lambda directory: f"./cache/{Path(directory).parent.name}_{Path(directory).name}_cached_structure.json")
    def make_dataset(self, directory, *args, **kwargs):
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset


def build_dataset(is_train, args, transform=None):
    if transform is None:
        transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = CachedImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
