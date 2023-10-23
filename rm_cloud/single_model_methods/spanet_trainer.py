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
class SPANetTrainer(Trainer):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    # def optimize_strategy_rsaunet(self,img,gt,M,is_optimizer):
    #     mask,gen_imgs = self.model(img)
    #     MSE_loss = get_mse_loss_function()
    #     mse_loss = MSE_loss(gen_imgs, gt)
    #     ssim_loss = SSIM().cuda()
    #     ssim_loss = ssim_loss(gen_imgs,gt)
    #
    #
    #     att_loss = MSE_loss(mask[:,0,:,:],M)
    #
    #     L1_loss = get_l1_loss_function()
    #     loss_l1 = L1_loss(gen_imgs, gt)
    #
    #     loss = loss_l1 + att_loss + mse_loss
    #
    #     if is_optimizer:
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         # xs change
    #         nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
    #         self.optimizer.step()
    #     return gen_imgs, loss


    def optimize_strategy_rsaunet(self,change_loss,img,gt,M,is_optimizer):
        # gen_imgs = self.model(img)
        mask, gen_imgs = self.model(img)
        MSE_loss = get_mse_loss_function()
        mse_loss = MSE_loss(gen_imgs, gt)
        ssim_loss = SSIM().cuda()
        ssim_loss = ssim_loss(gen_imgs,gt)
        att_loss = MSE_loss(mask[:, 0, :, :], M)

        L1_loss = get_l1_loss_function()
        loss_l1 = L1_loss(gen_imgs, gt)


        loss = 100 * loss_l1 + att_loss

        if change_loss:
            loss = 100 * mse_loss + att_loss

        if is_optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
        return gen_imgs, loss

