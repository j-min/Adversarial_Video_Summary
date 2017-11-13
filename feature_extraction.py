# -*- coding: utf-8 -*-

from os.path import join as opj
from glob import glob
import h5py
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms


class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img

class ResNetFeature(nn.Module):
    def __init__(self, feature='resnet101'):
        """
        Args:
            feature (string): resnet101 or resnet152
        """
        super(ResNetFeature, self).__init__()
        if feature == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        else:
            resnet = models.resnet152(pretrained=True)
        resnet.float()
        resnet.eval()

        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]

    def forward(self, x):
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return res5c, pool5

class FeatureExtracter(object):
    def __init__(self):
        self.topic = opt.topic
        self.feature = opt.feature

        self.root = opj(os.pardir, 'tmp_sum', 'photo')
        self.topic_path = opj(self.root, self.topic+'_copy')
        self.feature_path = opj(self.root, self.feature+'_feature')
        self.feature_topic_path = opj(self.feature_path, self.topic)

        photostream_list = glob(opj(self.topic_path, 'photostream_*'))
        self.photostream_list = sorted(photostream_list, key=numbering)

        make_directory(self.feature_topic_path)

        self.use_gpu = torch.cuda.is_available()
        self.model = None
        self.data_transform = None

    def feature_extract(self):
        print("-"*50)
        print("yfcc100m topic : {}, feature : {}".format(opt.topic, opt.feature))
        print("yfcc100m feature extraction start!")

        if self.use_gpu:
            self.model = self.model.cuda()

        for i in range(len(self.photostream_list)):
            photostream_path = self.photostream_list[i]
            hdf5_path = opj(self.feature_topic_path, 'photostream_{:06d}.hdf5'.format(i))
            photostream = PhotoStreamDataset(photostream_path, self.data_transform)
            dataloader = DataLoader(photostream, batch_size=32, shuffle=False, num_workers=8)
            self.write_hdf5(hdf5_path, dataloader)
            print("photostream {}/{} done".format(i+1, len(self.photostream_list)))

    def write_hdf5(hdf5_path, dataloader):
        pass




class ResNetExtracter(FeatureExtracter):
    def __init__(self, feature):
        """
        Args:
            feature (string): resnet101 or resnet152
        """
        super(ResNetExtracter, self).__init__()
        self.model = ResNetFeature(feature)
        self.data_transform = transforms.Compose([
                        Rescale(224, 224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    def write_hdf5(self, hdf5_path, dataloader):
        res5c_list = []
        pool5_list = []
        for input in dataloader:
            if self.use_gpu:
                input = Variable(input.cuda(), volatile=True)
            else:
                input = Variable(input, volatile=True)
            res5c, pool5 = self.model(input)
            res5c_list.append(res5c.data)
            pool5_list.append(pool5.data)
        res5c_cat = torch.cat(res5c_list, 0).cpu().numpy()
        pool5_cat = torch.cat(pool5_list, 0).cpu().numpy()

        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('res5c', data=res5c_cat)
            f.create_dataset('pool5', data=pool5_cat)    