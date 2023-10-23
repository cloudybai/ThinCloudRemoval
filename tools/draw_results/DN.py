from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from glob import glob
from PIL import Image
import time
import math
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage import img_as_float
import argparse
import sys
import os


def draw_hist(origin, target, result, AHF, HOT, ERT, CloudGAN, RSC, savepath, savename):
    origin_array = origin*255
    result_array = result*255
    target_array = target*255
    AHF_array = AHF*255
    IHOT_array = HOT*255
    ERT_array = ERT*255
    Cloud_GAN = CloudGAN*255
    RSC = RSC*255

    num = 25
    # num = 10
    x = np.arange(0, 255, num)
    hist_o, bin_edges_o = np.histogram(origin_array, bins=255, density=True)
    hist_r, bin_edges_r = np.histogram(result_array, bins=255, density=True)
    hist_t, bin_edges_t = np.histogram(target_array, bins=255, density=True)
    hist_h, bin_edges_h = np.histogram(AHF_array, bins=255, density=True)
    hist_i, bin_edges_i = np.histogram(IHOT_array, bins=255, density=True)
    hist_e, bin_edges_e = np.histogram(ERT_array, bins=255, density=True)
    hist_c, bin_edges_c = np.histogram(Cloud_GAN, bins=255, density=True)
    hist_d, bin_edges_d = np.histogram(RSC, bins=255, density=True)


    y_origin = hist_o[1:][::num]
    y_result = hist_r[1:][::num]
    y_target = hist_t[1:][::num]
    y_AHF = hist_h[1:][::num]
    y_IHOT = hist_i[1:][::num]
    y_ERT = hist_e[1:][::num]
    y_CloudGAN = hist_c[1:][::num]
    y_RSC = hist_d[1:][::num]


    '''n = len(y_origin)
    for i in range(1, n-1):
        if y_origin[i] == 0:
            y_origin[i] = (y_origin[i-1] + y_origin[i+1]) / 2
        if y_result[i] == 0:
            y_result[i] = (y_result[i-1] + y_result[i+1]) / 2
        if y_target[i] == 0:
            y_target[i] = (y_target[i-1] + y_target[i+1]) / 2'''

    plt.figure()
    plt.plot(x, y_origin, 'g-')
    plt.plot(x, y_AHF, 'y:')
    plt.plot(x, y_IHOT, 'c:')
    plt.plot(x, y_ERT, 'm:')
    plt.plot(x, y_CloudGAN, color='orangered', linestyle=':')
    plt.plot(x, y_RSC, color='lime', linestyle=':')
    plt.plot(x, y_result, 'b-')
    plt.plot(x, y_target, 'r-')

    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.legend(['cloud', 'AHF', 'HOT', 'ERT', 'Cloud-GAN', 'RSC-Net', 'SPAR-Net', 'clear'],loc = 'upper center')
    plt.title('the frequency of DN values')
    plt.savefig(savepath + savename + '.png')
    plt.show()
    plt.close()



if __name__ == '__main__':
    path = '/home/bbd/Desktop/remove_cloud/tools/draw_results/results/'

    name = '120037_289'
    filename = '120037_289_DN'
    origin = mimg.imread(path + name + '_cloud.bmp')
    target = mimg.imread(path + name + '_free.bmp')
    result = mimg.imread(path + name + '_sparnet.bmp')
    HOT = mimg.imread(path + name + '_HOT.bmp')
    ERT = mimg.imread(path + name + '_ERT.bmp')
    AHF = mimg.imread(path + name + '_AHF.bmp')
    cloudgan = mimg.imread(path + name + '_cloudgan.png')
    rsc = mimg.imread(path + name + '_rsc.bmp')
    draw_hist(origin, target, result, AHF, HOT, ERT, cloudgan, rsc, path, filename)





