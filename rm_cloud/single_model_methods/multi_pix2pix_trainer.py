#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
from lib.loss.mse_loss import get_mse_loss_function
from rm_cloud.single_model_methods.trainer import Trainer


class Multi_pix2pixTrainer(Trainer):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def optimize_multi_2_strategy(self, img_1, gt_1, img_2, gt_2):
        gen_imgs = self.model(img_1, img_2)
        mse_loss = get_mse_loss_function()

        loss_1 = mse_loss(gen_imgs[0], gt_1)
        loss_2 = mse_loss(gen_imgs[1], gt_2)

        loss =  loss_1 + loss_2

        # BP

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return gen_imgs, loss



