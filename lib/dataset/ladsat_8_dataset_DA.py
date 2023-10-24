#!/usr/bin/python3
#coding:utf-8



import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from numpy import *
from PIL import Image

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

        '''
        #show
        RGB = np.zeros([256, 256, 3])
        RGB[:, :, 0] = img_cloud[:, :, 3]
        RGB[:, :, 1] = img_cloud[:, :, 2]
        RGB[:, :, 2] = img_cloud[:, :, 1]
        rgb = Image.fromarray(RGB.astype(np.uint8), mode='RGB')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(rgb)
        RGB_ = np.zeros([256, 256, 3])
        RGB_[:, :, 0] = img_free[:, :, 3]
        RGB_[:, :, 1] = img_free[:, :, 2]
        RGB_[:, :, 2] = img_free[:, :, 1]
        rgb_ = Image.fromarray(RGB_.astype(np.uint8), mode='RGB')
        plt.figure()
        plt.imshow(rgb_)
        '''



        # if self.transforms:
        #     all_array = concatenate((img_cloud,img_free),axis=2)
        #     img_ = []
        #     gt_ = []
        #     all_img = self.transforms(all_array)
        #     img = np.zeros([256, 256,9])
        #     gt = np.zeros([256, 256,9])
        #     for i in range(9):
        #         img_arr = np.asarray(all_img[i])
        #         img[:, :,i] = img_arr
        #
        #     #show
        #     RGB_trans = np.zeros([256, 256, 3])
        #     RGB_trans[:, :, 0] = img[:, :,3]
        #     RGB_trans[:, :, 1] = img[:, :,2]
        #     RGB_trans[:, :, 2] = img[:, :,1]
        #
        #     rgb_trans = Image.fromarray(RGB_trans.astype(np.uint8), mode='RGB')
        #     plt.figure()
        #     plt.imshow(rgb_trans)
        #
        #
        #     for i in range(9,18):
        #         gt_arr =  np.asarray(all_img[i])
        #         gt[:, :,i-9] = gt_arr
        #
        #     #show
        #     gt_trans = np.zeros([256, 256, 3])
        #     gt_trans[:, :, 0] = gt[:, :,3]
        #     gt_trans[:, :, 1] = gt[ :, :,2]
        #     gt_trans[:, :, 2] = gt[ :, :,1]
        #     GT_trans = Image.fromarray(gt_trans.astype(np.uint8), mode='RGB')
        #     plt.figure()
        #     plt.imshow(GT_trans)
        #     plt.show()
        #
        #
        #     img = img.transpose([2, 0, 1]).astype(np.float32) / 255
        #     gt = gt.transpose([2, 0, 1]).astype(np.float32) / 255
        #     img = torch.FloatTensor(img)
        #     gt = torch.FloatTensor(gt)

        #add cloud
        if self.transforms:
            img_add_cloud = self.transforms(img_free)
            img = np.zeros([256, 256,9])
            for i in range(9):
                img_arr = np.asarray(img_add_cloud[i])
                img[:, :,i] = img_arr
            '''
            #show
            RGB_trans = np.zeros([256, 256, 3])
            RGB_trans[:, :, 0] = img[:, :,3]
            RGB_trans[:, :, 1] = img[:, :,2]
            RGB_trans[:, :, 2] = img[:, :,1]

            rgb_trans = Image.fromarray(RGB_trans.astype(np.uint8), mode='RGB')
            plt.figure()
            plt.imshow(rgb_trans)
            plt.show()
            '''


            img = img.transpose([2, 0, 1]).astype(np.float32) / 255
            gt = img_free.transpose([2, 0, 1]).astype(np.float32) / 255
            img = torch.FloatTensor(img)
            gt = torch.FloatTensor(gt)

        else:
            M = np.clip((np.array(img_cloud)-np.array(img_free)).sum(axis=2),0,1).astype(np.float32)
            img = img_cloud.transpose([2, 0, 1]).astype(np.float32) / 255
            gt = img_free.transpose([2, 0, 1]).astype(np.float32) / 255
            img = torch.FloatTensor(img)
            gt = torch.FloatTensor(gt)
        return (img,gt,M,name)
        # return (img,gt,name)







