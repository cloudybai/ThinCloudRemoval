#!/usr/bin/python3
# coding:utf-8


import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
from PIL import Image
import random
from torchvision.transforms import functional as F
#
# class Landsat(Dataset):
#     def __init__(self, txt_path):
#         with open(txt_path, 'r') as f:
#             self.dir_list = f.readlines()
#         self.length = len(self.dir_list)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         multi_dir = self.dir_list[index].strip('\n')
#         f = h5py.File(multi_dir, 'r')
#         name = multi_dir.split('/')[-1].split('.')[0]
#         img_cloud = f['data'][:]
#         img_free = f['label'][:]
#
#         M = np.clip((np.array(img_cloud) - np.array(img_free)).sum(axis=2), 0, 1).astype(np.float32)
#         img = img_cloud.transpose([2, 0, 1]).astype(np.float32) / 255
#         gt = img_free.transpose([2, 0, 1]).astype(np.float32) / 255
#         img = torch.FloatTensor(img)
#         gt = torch.FloatTensor(gt)
#         return (img, gt,M, name)


class Landsat(Dataset):
    def __init__(self, txt_path):
        if "train" in txt_path:
            # activate training if 'train' exists
            self.image = "/home/yub3/cloud/remove_cloud_codes/cloud/new_data/train_cloud/"
            self.gt = "/home/yub3/cloud/remove_cloud_codes/cloud/new_data/train_free/"
        else:
            # else it is testing
            self.image = "/home/yub3/cloud/remove_cloud_codes/cloud/test/real_test_cloud/"
            self.gt = "/home/yub3/cloud/remove_cloud_codes/cloud/test/real_test_free/"
        image_list = os.listdir(self.image + 'B1/')
        #the image list in all 9 folders are all the same so only B1 is ok
        self.files = image_list
        self.use_aug = False
        if self.use_aug:
            self.transform = transforms.Compose([
                transforms.Resize((438, 438))
            ])
            self.data_aug = transforms.Compose([
                transforms.FiveCrop((256, 256)),
                transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
            ])

        self.img_list = []
        self.gt_list = []
        self.M_list = []
        self.name_list = []
        for index in range(len(self.files)):
            self.read_img_gt(index)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.img_list[index]
        gt = self.gt_list[index]
        M = self.M_list[index]
        name = self.name_list[index]

        return (img, gt, M, name)

    def read_img_gt(self, index):
        # image
        image = self.image
        #groundtruth
        gt = self.gt
        filename = self.files[index]
        #tensor m1 * n1 = H * W (only one image in any folder is required)
        [m1, n1] = cv2.imread(image + 'B1/' + filename, cv2.IMREAD_GRAYSCALE).shape
        # creat a 9 dim array to save all channels data
        data = np.zeros([m1, n1, 9])
        data[:, :, 0] = cv2.imread(image + 'B1/' + filename, cv2.IMREAD_GRAYSCALE)
        data[:, :, 1] = cv2.imread(image + 'B2/' + filename, cv2.IMREAD_GRAYSCALE)
        data[:, :, 2] = cv2.imread(image + 'B3/' + filename, cv2.IMREAD_GRAYSCALE)
        data[:, :, 3] = cv2.imread(image + 'B4/' + filename, cv2.IMREAD_GRAYSCALE)
        data[:, :, 4] = cv2.imread(image + 'B5/' + filename, cv2.IMREAD_GRAYSCALE)
        data[:, :, 5] = cv2.imread(image + 'B6/' + filename, cv2.IMREAD_GRAYSCALE)
        data[:, :, 6] = cv2.imread(image + 'B7/' + filename, cv2.IMREAD_GRAYSCALE)
        # data[:, :, 7] = cv2.imread(image + 'B9/' + filename, cv2.IMREAD_GRAYSCALE)
        data[:, :, 7] = cv2.imread(image + 'B10/' + filename, cv2.IMREAD_GRAYSCALE)
        data[:, :, 8] = cv2.imread(image + 'B11/' + filename, cv2.IMREAD_GRAYSCALE)

        [m2, n2] = cv2.imread(gt + 'B1/' + filename, cv2.IMREAD_GRAYSCALE).shape
        label = np.zeros([m2, n2, 9])
        label[:, :, 0] = cv2.imread(gt + 'B1/' + filename, cv2.IMREAD_GRAYSCALE)
        label[:, :, 1] = cv2.imread(gt + 'B2/' + filename, cv2.IMREAD_GRAYSCALE)
        label[:, :, 2] = cv2.imread(gt + 'B3/' + filename, cv2.IMREAD_GRAYSCALE)
        label[:, :, 3] = cv2.imread(gt + 'B4/' + filename, cv2.IMREAD_GRAYSCALE)
        label[:, :, 4] = cv2.imread(gt + 'B5/' + filename, cv2.IMREAD_GRAYSCALE)
        label[:, :, 5] = cv2.imread(gt + 'B6/' + filename, cv2.IMREAD_GRAYSCALE)
        label[:, :, 6] = cv2.imread(gt + 'B7/' + filename, cv2.IMREAD_GRAYSCALE)
        # label[:, :, 7] = cv2.imread(gt + 'B9/' + filename, cv2.IMREAD_GRAYSCALE)
        label[:, :, 7] = cv2.imread(gt + 'B10/' + filename, cv2.IMREAD_GRAYSCALE)
        label[:, :, 8] = cv2.imread(gt + 'B11/' + filename, cv2.IMREAD_GRAYSCALE)

        img_cloud=data
        img_free=label
        name = filename.split(".")[0]
        #split '123456.bmp' into '123456'

        M = np.clip((np.array(img_cloud) - np.array(img_free)).sum(axis=2), 0, 1).astype(np.float32)
        img = img_cloud.transpose([2, 0, 1]).astype(np.float32) / 255
        gt = img_free.transpose([2, 0, 1]).astype(np.float32) / 255
        img = torch.FloatTensor(img)
        gt = torch.FloatTensor(gt)

        if self.use_aug:
            img = self.transform(img)
            gt = self.transform(gt)

            img_aug = self.data_aug(img)  # 5, 9, H, W
            gt_aug = self.data_aug(gt)

            for idx in range(len(img_aug)):
                self.img_list.append(img_aug[idx])
                self.gt_list.append(gt_aug[idx])
                self.M_list.append(M)
                self.name_list.append(name)
        else:
            self.img_list.append(img)
            self.gt_list.append(gt)
            self.M_list.append(M)
            self.name_list.append(name)

        return


class Landsat_multi_scale(Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            self.dir_list = f.readlines()
        self.length = len(self.dir_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        multi_dir = self.dir_list[index].strip('\n').split(' ')

        h5_0 = multi_dir[0]
        h5_1 = multi_dir[1]
        h5_2 = multi_dir[2]

        name = h5_0.split('/')[-1].split('.')[0]

        f0 = h5py.File(h5_0, 'r')
        img_0 = f0['data'][:]
        gt_0 = f0['label'][:]

        f1 = h5py.File(h5_1, 'r')
        img_1 = f1['data'][:]
        gt_1 = f1['label'][:]

        f2 = h5py.File(h5_2, 'r')
        img_2 = f2['data'][:]
        gt_2 = f2['label'][:]

        M_0= np.clip((np.array(img_0) - np.array(gt_0)).sum(axis=2), 0, 1).astype(np.float32)
        img_0_T = img_0.transpose([2, 0, 1]).astype(np.float32) / 255
        gt_0_T = gt_0.transpose([2, 0, 1]).astype(np.float32) / 255

        M_1= np.clip((np.array(img_1) - np.array(gt_1)).sum(axis=2), 0, 1).astype(np.float32)
        img_1_T = img_1.transpose([2, 0, 1]).astype(np.float32) / 255
        gt_1_T = gt_1.transpose([2, 0, 1]).astype(np.float32) / 255

        M_2= np.clip((np.array(img_2) - np.array(gt_2)).sum(axis=2), 0, 1).astype(np.float32)
        img_2_T = img_2.transpose([2, 0, 1]).astype(np.float32) / 255
        gt_2_T = gt_2.transpose([2, 0, 1]).astype(np.float32) / 255

        img_0_T = torch.FloatTensor(img_0_T)
        gt_0_T = torch.FloatTensor(gt_0_T)
        img_1_T = torch.FloatTensor(img_1_T)
        gt_1_T = torch.FloatTensor(gt_1_T)
        img_2_T = torch.FloatTensor(img_2_T)
        gt_2_T = torch.FloatTensor(gt_2_T)

        # return (img_0_T, gt_0_T, M_0, M_1,M_2, name)
        # return (img_0_T, gt_0_T, M_0,img_1_T, gt_1_T, M_1, name)
        return (img_0_T, gt_0_T, M_0,img_1_T, gt_1_T, M_1, img_2_T, gt_2_T, M_2, name)


class Landsat_RGB(Dataset):
    def __init__(self, args, img_dir, gt_dir, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.args = args
        self.unaligned = unaligned
        self.files_X = sorted(glob.glob(os.path.join(img_dir) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(gt_dir) + '/*.*'))
        print(len(self.files_X))

    def __getitem__(self, index):

        img_X = Image.open(self.files_X[index % len(self.files_X)])
        if self.unaligned:
            img_Y = Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)])
        else:
            img_Y = Image.open(self.files_Y[index % len(self.files_Y)])

        M= np.clip((np.array(img_X) - np.array(img_Y)).sum(axis=2), 0, 1).astype(np.float32)
        img_X = self.transform(img_X)
        img_Y = self.transform(img_Y)

        return (img_X,img_Y,M,index)

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))







