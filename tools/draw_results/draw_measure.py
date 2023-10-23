#!/usr/bin/python3
#coding:utf-8


'draw psnr and ssim loss in tensorboard to select the best model'
import os

from RSC.ssim import get_mean_ssim
from RSC.psnr import get_mean_psnr
from RSC.options import TrainOptions
from tensorboardX import SummaryWriter

args = TrainOptions().parse()
writer = SummaryWriter(comment='scalar11')
def get_best_model(img_dir):
    result_list = os.listdir(img_dir)
    result_list = sorted(result_list, key=lambda x: os.path.getmtime(os.path.join(img_dir, x)))
    print(result_list)

    for i in range(len(result_list)):
        result_path = img_dir + str(result_list[i]) +'/result/'
        print(result_path)
        target_path = '/home/bbd/Desktop/remove_cloud/experiments/UNet/target/'
        ssim_ = get_mean_ssim(result_path, target_path)
        writer.add_scalar('scalar/ssim', ssim_,i)
        psnr_ = get_mean_psnr(result_path, target_path)
        writer.add_scalar('scalar/psnr', psnr_,i)
    writer.close()

#img_dir = '/home/bbd/Desktop/remove_cloud/experiments/UNet/save_result_img_unet'
img_dir = '/home/zhh6/cloud/remove_cloud_codes/experiments/UNet/save_result_img_unet/'
get_best_model(img_dir)

