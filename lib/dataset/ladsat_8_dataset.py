#!/usr/bin/python3
#coding:utf-8


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

class Landsat(Dataset):
    def __init__(self,txt_path):
        with open(txt_path,'r') as f:
            self.dir_list=f.readlines()
        self.length=len(self.dir_list)

    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        multi_dir = self.dir_list[index].strip('\n')
        f = h5py.File(multi_dir, 'r')
        name = multi_dir.split('/')[-1].split('.')[0]
        img_cloud = f['data'][:]
        img_free= f['label'][:]

        # not data enhance
        img=img_cloud.transpose([2,0,1]).astype(np.float32)/255
        gt=img_free.transpose([2,0,1]).astype(np.float32)/255
        img = torch.FloatTensor(img)
        gt = torch.FloatTensor(gt)
        '''
        # have data enhance
        img=img_cloud.transpose([2,0,1])
        gt=img_free.transpose([2,0,1])
        # img=(img_cloud.transpose([2,0,1]).astype(np.float32)/255-0.5)/0.5
        # gt=(img_free.transpose([2,0,1]).astype(np.float32)/255-0.5)/0.5

        img_1=[]
        gt_1 = []
        for i in range(9):
            img_rvf,gt_rvf = RVF()(Image.fromarray(np.float64(img[i,:,:])),Image.fromarray(np.float64(gt[i,:,:])))
            # img_rhf,gt_rhf = RHF()(img_rvf,gt_rvf)
            # img_rrt,gt_rrt = RRt(90)(img_rhf,gt_rhf)
            img_,gt_ = Ttensor()(img_rvf,gt_rvf)
            img_1.append(img_)
            gt_1.append(gt_)
        img=torch.cat(img_1,dim=0)
        gt=torch.cat(gt_1,dim=0)
        img= img/255
        gt = gt/255
        '''
        return (img,gt,name)


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







