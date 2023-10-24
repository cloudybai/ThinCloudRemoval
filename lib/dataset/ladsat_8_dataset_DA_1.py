#!/usr/bin/python3
#coding:utf-8


import numpy as np
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
from PIL import  Image
import random
from torchvision.transforms import functional as F

class RVF(transforms.RandomVerticalFlip):
    def __call__(self,img,gt):
        if random.random() < self.p:
            return (F.vflip(img),F.vflip(gt))
        return [img,gt]
class RHF(transforms.RandomHorizontalFlip):
    def __call__(self,img,gt):
        if random.random() < self.p:
            return (F.hflip(img),F.hflip(gt))
        return [img,gt]

class RRt(transforms.RandomRotation):
    def __call__(self,img,gt):
        angle = self.get_params(self.degrees)
        return [F.rotate(img, angle, self.resample, self.expand, self.center),F.rotate(gt, angle, self.resample, self.expand, self.center)]
class Ttensor(transforms.ToTensor):
    def __call__(self, img,gt):
        return [F.to_tensor(img),F.to_tensor(gt)]

class Comp(transforms.Compose):
    def __call__(self, img,gt):
        for t in self.transforms:
            [img,gt]= t(img,gt)
        return [img,gt]

class RVF2(transforms.RandomVerticalFlip):
    def __call__(self,img1,gt1,img2,gt2):
        if random.random() < self.p:
            return (F.vflip(img1),F.vflip(gt1),F.vflip(img2),F.vflip(gt2))
        return [img1,gt1,img2,gt2]
class RHF2(transforms.RandomHorizontalFlip):
    def __call__(self,img1,gt1,img2,gt2):
        if random.random() < self.p:
            return (F.hflip(img1),F.hflip(gt1),F.hflip(img2),F.hflip(gt2))
        return [img1,gt1,img2,gt2]

class RRt2(transforms.RandomRotation):
    def __call__(self,img1,gt1,img2,gt2):
        angle = self.get_params(self.degrees)
        return [F.rotate(img1, angle, self.resample, self.expand, self.center),F.rotate(gt1, angle, self.resample, self.expand, self.center),
                F.rotate(img2, angle, self.resample, self.expand, self.center),F.rotate(gt2, angle, self.resample, self.expand, self.center)]
class Ttensor2(transforms.ToTensor):
    def __call__(self,img1,gt1,img2,gt2):
        return [F.to_tensor(img1),F.to_tensor(gt1),F.to_tensor(img2),F.to_tensor(gt2)]


from numpy import *
class Landsat(Dataset):
    def __init__(self,txt_path, transforms):
        with open(txt_path,'r') as f:
            self.dir_list=f.readlines()
        self.length=len(self.dir_list)
        self.transforms = transforms

    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        multi_dir = self.dir_list[index].strip('\n')
        f = h5py.File(multi_dir, 'r')
        name = multi_dir.split('/')[-1].split('.')[0]
        img_cloud = f['data'][:]
        img_free= f['label'][:]
        # RGB = np.zeros([256, 256, 3])
        # RGB[:, :, 0] = img_cloud[:, :, 3]
        # RGB[:, :, 1] = img_cloud[:, :, 2]
        # RGB[:, :, 2] = img_cloud[:, :, 1]
        #
        # rgb = Image.fromarray(RGB.astype(np.uint8), mode='RGB')
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.imshow(rgb)
        #
        # RGB_ = np.zeros([256, 256, 3])
        # RGB_[:, :, 0] = img_free[:, :, 3]
        # RGB_[:, :, 1] = img_free[:, :, 2]
        # RGB_[:, :, 2] = img_free[:, :, 1]
        # rgb_ = Image.fromarray(RGB_.astype(np.uint8), mode='RGB')
        # plt.figure()
        # plt.imshow(rgb_)

        if self.transforms:
            # img=img_cloud.transpose([2,0,1])
            # gt=img_free.transpose([2,0,1])

            all_array = concatenate((img_cloud,img_free),axis=2)

            img_ = []
            gt_ = []
            all_img = self.transforms(all_array)
            img = np.zeros([256, 256,9])
            gt = np.zeros([256, 256,9])
            for i in range(8):
                img_arr = np.asarray(all_img[i])
                img[:, :,i] = img_arr
            # RGB_trans = np.zeros([256, 256, 3])
            # RGB_trans[:, :, 0] = img[:, :,3]
            # RGB_trans[:, :, 1] = img[:, :,2]
            # RGB_trans[:, :, 2] = img[:, :,1]
            #
            # rgb_trans = Image.fromarray(RGB_trans.astype(np.uint8), mode='RGB')
            # plt.figure()
            # plt.imshow(rgb_trans)

            for i in range(9,18):
                gt_arr =  np.asarray(all_img[i])
                gt[:, :,i-9] = gt_arr
            # gt_trans = np.zeros([256, 256, 3])
            # gt_trans[:, :, 0] = gt[:, :,3]
            # gt_trans[:, :, 1] = gt[ :, :,2]
            # gt_trans[:, :, 2] = gt[ :, :,1]
            #
            # GT_trans = Image.fromarray(gt_trans.astype(np.uint8), mode='RGB')
            # plt.figure()
            # plt.imshow(GT_trans)
            # plt.show()
            M = np.clip((np.array(img_cloud)-np.array(img_free)).sum(axis=2),0,1).astype(np.float32)
            img = img.transpose([2, 0, 1]).astype(np.float32) / 255
            gt = gt.transpose([2, 0, 1]).astype(np.float32) / 255
            img = torch.FloatTensor(img)
            gt = torch.FloatTensor(gt)

        else:
            M = np.clip((np.array(img_cloud)-np.array(img_free)).sum(axis=2),0,1).astype(np.float32)

            img = img_cloud.transpose([2, 0, 1]).astype(np.float32) / 255
            gt = img_free.transpose([2, 0, 1]).astype(np.float32) / 255
            img = torch.FloatTensor(img)
            gt = torch.FloatTensor(gt)

        return (img,gt,M,name)


class Landsat_multi_scale(Dataset):
    def __init__(self,txt_path):
        with open(txt_path,'r') as f:
            self.dir_list=f.readlines()
        self.length=len(self.dir_list)

    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        multi_dir = self.dir_list[index].strip('\n').split(' ')

        h5_0 = multi_dir[0]
        h5_1 = multi_dir[1]
        h5_2 = multi_dir[2]

        name = h5_0.split('/')[-1].split('.')[0]

        f0 = h5py.File(h5_0, 'r')
        img_0 = f0['data'][:]
        gt_0= f0['label'][:]

        f1 = h5py.File(h5_1, 'r')
        img_1 = f1['data'][:]
        gt_1 = f1['label'][:]

        f2 = h5py.File(h5_2, 'r')
        img_2 = f2['data'][:]
        gt_2 = f2['label'][:]

        img_0_T=img_0.transpose([2,0,1]).astype(np.float32)/255
        gt_0_T=gt_0.transpose([2,0,1]).astype(np.float32)/255

        img_1_T=img_1.transpose([2,0,1]).astype(np.float32)/255
        gt_1_T=gt_1.transpose([2,0,1]).astype(np.float32)/255


        img_2_T=img_2.transpose([2,0,1]).astype(np.float32)/255
        gt_2_T=gt_2.transpose([2,0,1]).astype(np.float32)/255

        img_0_T = torch.FloatTensor(img_0_T)
        gt_0_T = torch.FloatTensor(gt_0_T)
        img_1_T = torch.FloatTensor(img_1_T)
        gt_1_T = torch.FloatTensor(gt_1_T)
        img_2_T = torch.FloatTensor(img_2_T)
        gt_2_T = torch.FloatTensor(gt_2_T)

        return (img_0_T,gt_0_T,img_1_T,gt_1_T,img_2_T,gt_2_T,name)

class Landsat_multi_2_scale(Dataset):
    def __init__(self,txt_path):
        with open(txt_path,'r') as f:
            self.dir_list=f.readlines()
        self.length=len(self.dir_list)

    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        multi_dir = self.dir_list[index].strip('\n').split(' ')

        h5_0 = multi_dir[0]
        h5_1 = multi_dir[1]

        name = h5_0.split('/')[-1].split('.')[0]

        f0 = h5py.File(h5_0, 'r')
        img_0 = f0['data'][:]
        gt_0= f0['label'][:]

        f1 = h5py.File(h5_1, 'r')
        img_1 = f1['data'][:]
        gt_1 = f1['label'][:]

# not data enhance
        '''
        img_0_T=img_0.transpose([2,0,1]).astype(np.float32)/255
        gt_0_T=gt_0.transpose([2,0,1]).astype(np.float32)/255

        img_1_T=img_1.transpose([2,0,1]).astype(np.float32)/255
        gt_1_T=gt_1.transpose([2,0,1]).astype(np.float32)/255

        img_0_T = torch.FloatTensor(img_0_T)
        gt_0_T = torch.FloatTensor(gt_0_T)
        img_1_T = torch.FloatTensor(img_1_T)
        gt_1_T = torch.FloatTensor(gt_1_T)

        return (img_0_T,gt_0_T,img_1_T,gt_1_T,name)'''

 # have data enhance
        img1 = img_0.transpose([2, 0, 1])
        gt1 = gt_0.transpose([2, 0, 1])

        img2 = img_1.transpose([2, 0, 1])
        gt2 = gt_1.transpose([2, 0, 1])

        img_1=[]
        gt_1 = []
        img_2=[]
        gt_2 = []
        for i in range(9):
            img1_rvf,gt1_rvf,img2_rvf,gt2_rvf = RVF2()(Image.fromarray(np.float64(img1[i,:,:])),Image.fromarray(np.float64(gt1[i,:,:])),
                                   Image.fromarray(np.float64(img2[i,:,:])),Image.fromarray(np.float64(gt2[i,:,:])))
            # img_rhf,gt_rhf = RHF()(img_rvf,gt_rvf)1
            # img_rrt,gt_rrt = RRt(90)(img_rhf,gt_rhf)
            img1_,gt1_,img2_,gt2_ = Ttensor2()(img1_rvf,gt1_rvf,img2_rvf,gt2_rvf)
            img_1.append(img1_)
            gt_1.append(gt1_)
            img_2.append(img2_)
            gt_2.append(gt2_)

        img1=torch.cat(img_1,dim=0)
        gt1=torch.cat(gt_1,dim=0)

        img2=torch.cat(img_2,dim=0)
        gt2=torch.cat(gt_2,dim=0)
        img1= img1/255
        gt1 = gt1/255
        img2= img2/255
        gt2 = gt2/255
        return (img1,gt1,img2,gt2,name)

class Landsat_RGB(Dataset):
    def __init__(self, args, img_dir,gt_dir, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.args = args
        self.unaligned = unaligned
        self.files_X = sorted(glob.glob(os.path.join(img_dir) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(gt_dir) + '/*.*'))
        print (len(self.files_X))

    def __getitem__(self, index):

        img_X = Image.open(self.files_X[index % len(self.files_X)])
        if self.unaligned:
            img_Y = Image.open(self.files_Y[random.randint(0, len(self.files_Y)-1)])
        else:
            img_Y = Image.open(self.files_Y[index % len(self.files_Y)] )

        img_X = self.transform(img_X)
        img_Y = self.transform(img_Y)

        # if self.args.input_nc_A == 1:  # RGB to gray
        #     img_X = img_X.convert('L')
        #
        # if self.args.input_nc_B == 1:  # RGB to gray
        #     img_Y = img_Y.convert('L')

        return {'X': img_X, 'Y': img_Y}

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))







