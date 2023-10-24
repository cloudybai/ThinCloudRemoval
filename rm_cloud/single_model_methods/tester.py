#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import torch
from torch.autograd import Variable
from lib.utils.save_each_band import save_each_band
from lib.measures.get_psnr_ssim import get_psnr_ssim_9,get_psnr_ssim_3,get_psnr_ssim_mean


class Testner():
    def __init__(self,model):
        self.model = model.eval()

    def test(self, args, dataloader):
        """

        :param args:
        :param dataloader:
        :return: save the results as img

        """
        if args.kind == 'single':
            for i,(img, gt,M,name) in enumerate(dataloader):
                if torch.cuda.is_available():
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                else:
                    img = Variable(img)
                    gt = Variable(gt)

                gen_img = self.model(img)
                name = list(name)[0]
                if args.model.arch == 'spanet':
                    save_each_band(args, gen_img[1], name)
                elif args.model.arch == 'rsaunet':
                    save_each_band(args, gen_img[1], name)

                elif args.model.arch == 'u2net':
                    save_each_band(args, gen_img[0], name)

                elif args.model.arch == 'Multi3_SPANet':
                    save_each_band(args, gen_img[3], name)

                elif args.model.arch == 'MSPF':
                    save_each_band(args, gen_img[1], name)

                else:
                    save_each_band(args, gen_img, name)



        if args.kind == 'multi':
            for i, (img_0,gt_0,m0,img_1,gt_1,m1,img_2,gt_2,m2,name) in enumerate(dataloader):
                if torch.cuda.is_available():
                    img_0 = Variable(img_0.cuda())
                    img_1 = Variable(img_1.cuda())
                    img_2 = Variable(img_2.cuda())

                else:
                    img_0 = Variable(img_0)

                gen_img = self.model(img_0,img_1,img_2)
                name = list(name)[0]


                if args.model.arch == 'Multi3_SPANet':
                    save_each_band(args, gen_img[3], name)

                elif args.model.arch == 'MSPF':
                    if args.name == 'SS':
                        save_each_band(args, gen_img, name)
                    else:

                        save_each_band(args, gen_img[3], name)
                else:
                    save_each_band(args, gen_img, name)

        if args.kind == 'multi_2':
            for i, (img_0, gt_0, img_1, gt_1, name) in enumerate(dataloader):
                if torch.cuda.is_available():
                    img_0 = Variable(img_0.cuda())
                    gt_0 = Variable(gt_0.cuda())
                    img_1 = Variable(img_1.cuda())
                    gt_1 = Variable(gt_1.cuda())

                else:
                    img_0 = Variable(img_0)
                    gt_0 = Variable(gt_0)
                    img_1 = Variable(img_1)
                    gt_1 = Variable(gt_1)

                gen_imgs = self.model(img_0, img_1)
                name = list(name)[0]
                save_each_band(args, gen_imgs[0], name)


    def eval(self, args):
        """

        :param args:
        :return: get measure
        """
        get_psnr_ssim_9(args.evaluation_dir, args.gt_dir, args.model.arch)







