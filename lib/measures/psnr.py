#!/usr/bin/python3
# coding:utf-8


# from skimage.measure import compare_psnr

from skimage.metrics import peak_signal_noise_ratio
import math
import numpy as np

# def psnr(img, gt):
#     '''
#
#     :param img:
#     :param gt:
#     :return:
#     '''
#     mse = np.mean((img - gt) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr(img,gt):
    return peak_signal_noise_ratio(gt,img)













