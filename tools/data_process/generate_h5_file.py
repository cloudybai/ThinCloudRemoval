#!/usr/bin/python3
#coding:utf-8



import cv2
import h5py
import numpy as np
import os
"""
save data to h5file

image:the data of cloudy
gt:the data of free
"""

def generate_h5(image, gt, h5Path):
    """

    :param image: the data of cloud-contaminate
    :param gt: the data of cloud-free
    :param h5Path:
    :return:
    """
    os.makedirs(h5Path,exist_ok=True)
    # get all file in this scene
    image_list = os.listdir(gt + 'B1/')
    for i in range(len(image_list)):
        filename = image_list[i]
        print(filename)
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

        h5_name = '{}.h5'.format(filename.split('.')[0])
        os.makedirs(h5Path,exist_ok=True)
        h5_file = h5Path + h5_name
        f = h5py.File(h5_file, 'w')
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)
        f.close()


if __name__ == '__main__':
    # scale = '0.5/'
    free = 'real_test_free/'
    cloud = 'real_test_cloud/'
    h5Path = '/home/bbd/Desktop/remove_cloud/cloud/test/hd5/'
    image = '/home/bbd/Desktop/remove_cloud/cloud/test/' + cloud
    gt = '/home/bbd/Desktop/remove_cloud/cloud/test/' + free

    # h5Path = '/media/userdisk2/xqzhou/cloud/multi_test/' + scale
    # image = '/media/userdisk2/xqzhou/cloud/multi/' + cloud + scale
    # gt = '/media/userdisk2/xqzhou/cloud/multi/' + free + scale

    # h5Path = '/media/omnisky/data3/xiesong/dataset/cloud/RSC_data_h5/mytest/'
    # image = '/media/omnisky/data3/xiesong/dataset/cloud/RSC_data/mytest_cloud/'
    # gt = '/media/omnisky/data3/xiesong/dataset/cloud/RSC_data/mytest_free/'

    #
    free = 'train_free/'
    cloud = 'train_cloud/'
    h5Path = '/home/bbd/Desktop/remove_cloud/cloud/new_data/hd5/'
    image = '/home/bbd/Desktop/remove_cloud/cloud/new_data/' + cloud
    gt = '/home/bbd/Desktop/remove_cloud/cloud/new_data/' + free
    # h5Path = '/media/userdisk2/xqzhou/cloud/new_data/hd5/'
    # image = '/media/userdisk2/xqzhou/cloud/new_data/' + cloud
    # gt = '/media/userdisk2/xqzhou/cloud/new_data/' + free


    generate_h5(image,gt,h5Path)













