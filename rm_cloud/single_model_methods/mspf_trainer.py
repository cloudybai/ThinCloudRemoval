#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
from rm_cloud.single_model_methods.trainer import Trainer
from lib.loss.mse_loss import get_mse_loss_function
import torch.nn as nn
from lib.loss.L1_loss import get_l1_loss_function
from lib.loss.loss_ssim import SSIM
class MSPFTrainer(Trainer):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def optimize_multi_strategy(self,args,img,gt,M,img1,gt1,M1,img2,gt2,M2,is_optimizer):
        if args.name == 'MSCF':
            _, _, _, gen, gen1, gen2 = self.model(img, img1, img2)
            L1_loss = get_l1_loss_function()
            loss_l1 = L1_loss(gen, gt)
            loss_l1_1 = L1_loss(gen1, gt1)
            loss_l1_2 = L1_loss(gen2, gt2)
            loss = 100 * loss_l1 + 50 * loss_l1_1 + 25 * loss_l1_2

        else:
            mask, mask1,mask2,gen,gen1,gen2 = self.model(img,img1,img2)
            MSE_loss = get_mse_loss_function()
            att_loss = MSE_loss(mask[:,0,:,:],M)
            att_loss1 = MSE_loss(mask1[:,0,:,:],M1)
            att_loss2 = MSE_loss(mask2[:,0,:,:],M2)
            L1_loss = get_l1_loss_function()
            loss_l1 = L1_loss(gen, gt)
            loss_l1_1 = L1_loss(gen1, gt1)
            loss_l1_2 = L1_loss(gen2, gt2)
            if args.name == 'MSSPF':
                loss = 100* loss_l1 + att_loss
            else:
                loss = 100* loss_l1 + 50 * loss_l1_1 + 25*loss_l1_2 + att_loss + att_loss1 + att_loss2


        if is_optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            # xs change
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
        return gen, loss


    def optimize_strategy_rsaunet(self,change_loss,img,gt,M,is_optimizer):
        # train generator#

        mask,gen_imgs = self.model(img)

        L1_loss = get_l1_loss_function()

        loss_l1 = L1_loss(gen_imgs,gt)

        MSE_loss = get_mse_loss_function()
        att_loss = MSE_loss(mask[:, 0, :, :], M)

        loss= 100 * loss_l1 +att_loss
        #BP
        if is_optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            # xs change
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

        return gen_imgs,loss



