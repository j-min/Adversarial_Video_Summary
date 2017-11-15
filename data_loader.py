# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path

from feature_extraction import resnet_transform
import h5py
import numpy as np


class VideoData(Dataset):
    def __init__(self, root, preprocessed=True, transform=resnet_transform, with_name=False):
        self.root = root
        self.preprocessed = preprocessed
        self.transform = transform
        self.with_name = with_name
        self.video_list = list(self.root.iterdir())

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        if self.preprocessed:
            image_path = self.video_list[index]
            with h5py.File(image_path, 'r') as f:
                if self.with_name:
                    return torch.Tensor(np.array(f['pool5'])), image_path.name[:-5]
                else:
                    return torch.Tensor(np.array(f['pool5']))

        else:
            images = []
            for img_path in Path(self.video_list[index]).glob('*.jpg'):
                img = default_loader(img_path)
                img_tensor = self.transform(img)
                images.append(img_tensor)

            return torch.stack(images), img_path.parent.name[4:]


def get_loader(root, mode):
    if mode.lower() == 'train':
        return DataLoader(VideoData(root), batch_size=1)
    else:
        return VideoData(root, with_name=True)


if __name__ == '__main__':
    pass
