#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
import numpy as np
from PIL import Image
import cv2
import os
def split(origin_img,filename):
    b2_dir = '/home/xqzhou/xscode/传统/ERT/B2/'
    b3_dir = '/home/xqzhou/xscode/传统/ERT/B3/'
    b4_dir = '/home/xqzhou/xscode/传统/ERT/B4/'

    image = cv2.imread(origin_img)
    (B, G, R) = cv2.split(image)

    # cv2.imshow("Red", R)
    # cv2.imshow("Green", G)
    # cv2.imshow("Blue", B)
    cv2.imwrite(os.path.join(b4_dir,filename),R)
    cv2.imwrite(os.path.join(b3_dir,filename),G)
    cv2.imwrite(os.path.join(b2_dir,filename),B)

if __name__ == '__main__':
    img_dir = '/home/xqzhou/xscode/传统/ERT/RGB/'
    img_list = os.listdir(img_dir)
    print(len(img_list))
    for img in img_list:
        img_path = os.path.join(img_dir,img)
        split(img_path,img)

