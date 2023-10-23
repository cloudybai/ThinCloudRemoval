#!/usr/bin/python3
#coding:utf-8
import os
import sys
sys.path.append('../..')
import cv2
from lib.measures.ssim import ssim
from lib.measures.psnr import psnr
import xlwt

def get_psnr_ssim_9(img_path, gt_path, xl_name):

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(xl_name)
    image_list = os.listdir(img_path + 'B1/')

    m = 1
    for i in range(len(image_list)):
        filename = image_list[i]
        print(filename)
        headlist_index = [filename, 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
        # write cols
        for index, cols in enumerate(headlist_index):
            sheet.write(m, index, cols)

        for j in range(0,9):
            if j ==7 or j ==8:
                band = j + 3
            else:
                band = j + 1

            img = cv2.imread(img_path + 'B'+ str(band) + '/' + filename, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path + 'B'+ str(band) + '/' + filename, cv2.IMREAD_GRAYSCALE)

            psnr_value = psnr(img, gt)
            ssim_value = ssim(img, gt)

            # write train db
            print(psnr_value)
            print(ssim_value)

            sheet.write(m+1, j+1, psnr_value)
            sheet.write(m+2, j+1, ssim_value)

        m = m+3


    workbook.save('{}/{}.xls'.format(img_path,xl_name))

def get_psnr_ssim_3(img_path, gt_path,xl_name):

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(xl_name)
    image_list = os.listdir(img_path + 'B2/')

    m = 1
    for i in range(len(image_list)):
        filename = image_list[i]
        # cyclegan
        # file_name_tuple = filename.split('_')
        # file = file_name_tuple[0] + '_' + file_name_tuple[1] + '.bmp'

        print(filename)
        headlist_index = [filename, 'B2', 'B3', 'B4']
        # write cols
        for index, cols in enumerate(headlist_index):
            sheet.write(m, index, cols)

        for j in range(2, 5):

            img = cv2.imread(img_path + 'B' + str(j) + '/' + filename, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path + 'B' + str(j) + '/' + filename, cv2.IMREAD_GRAYSCALE)
            # gt = cv2.imread(gt_path + 'B' + str(j) + '/' + file, cv2.IMREAD_GRAYSCALE)

            psnr_value = psnr(img, gt)
            ssim_value = ssim(img, gt)

            # write train db

            sheet.write(m + 1, j-1, psnr_value)
            sheet.write(m + 2, j-1, ssim_value)

        m = m + 3

    workbook.save('{}/{}.xls'.format(img_path,xl_name))



def get_psnr_ssim_mean(img_path, gt_path, xl_name):
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(xl_name)
    j = 0
    for (root, dirs, files) in os.walk(img_path):
        for dir in sorted(dirs):
            if dir != 'RGB':
                sum_psnr = 0
                sum_ssim = 0
                dir_path = os.path.join(root, dir)
                img_list = os.listdir(dir_path)
                for img_file in img_list:
                    #
                    # file_name_tuple = img_file.split('_')
                    # file = file_name_tuple[0] + '_' + file_name_tuple[1] + '.bmp'

                    img = cv2.imread(os.path.join(img_path, dir, img_file),cv2.IMREAD_GRAYSCALE)
                    gt = cv2.imread(os.path.join(gt_path, dir, img_file),cv2.IMREAD_GRAYSCALE)
                    # gt = cv2.imread(os.path.join(gt_path, dir, file),cv2.IMREAD_GRAYSCALE)

                    psnr_value = psnr(img,gt)
                    ssim_value = ssim(img,gt)

                    sum_psnr += psnr_value
                    sum_ssim += ssim_value

            mean_psnr = sum_psnr/len(img_list)
            mean_ssim = sum_ssim/len(img_list)
            print(len(img_list))

            print('{band},mean_psnr:{mean_psnr},mean_ssim:{mean_ssim}'.format(band=dir,mean_psnr=mean_psnr,mean_ssim=mean_ssim))
            sheet.write(0, j, dir)
            sheet.write(1, j, mean_psnr)
            sheet.write(2 + 2, j, mean_ssim)
            j = j + 1

    workbook.save('{}/{}_mean.xls'.format(img_path,xl_name))

if __name__ == '__main__':
    # img_path = '/home/xqzhou/xscode/传统/AHF/'
    # img_path = '/home/data1/xiesong/git_repo/removal_thin_cloud/experiments/ssa/evaluation_results/2998/'
    # gt_path = '/media/omnisky/data3/xiesong/dataset/cloud/landsat8/test_free/1/'
    # img_path = '/home/xqzhou/xscode/remove_cloud/experiments/MSPF/MSEF1_430/'
    img_path = '/home/yub3/cloud/remove_cloud_codes/experiments/UNet/unet_2'
    # img_path = '/home/bbd/Desktop/remove_cloud/experiments/SPANet/new/'
    # img_path = '/media/userdisk2/xqzhou/cloud/landsat8/real_test_cloud/'
    gt_path = '/home/yub3/cloud/remove_cloud_codes/cloud/test/real_test_free/'
    xl_name ='unet3d_2023'
    get_psnr_ssim_mean(img_path, gt_path, xl_name)
    # get_psnr_ssim_3('/home/xqzhou/xscode/pytorch-CycleGAN-and-pix2pix-master/results/sparnet_cyclegan/test_latest/',gt_path,'sparnet_cyclegan')
    # get_psnr_ssim_3('/home/xqzhou/xscode/传统/AHF/',gt_path,'AHF')
    # get_psnr_ssim_mean('/home/xqzhou/xscode/传统/AHF/',gt_path,'AHF_mean')