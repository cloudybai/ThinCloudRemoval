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
from lib.loss.L1_loss import get_l1_loss_function
import torch.nn as nn


class U2netTrainer(Trainer):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    # def optimize_strategy(self, img, gt, is_optimizer):
    #     # train generator#
    #
    #     d0, d1, d2, d3, d4, d5, d6 = self.model(img)
    #     mse_loss = get_mse_loss_function()
    #     L1_loss = get_l1_loss_function()
    #
    #     loss0 = mse_loss(d0, gt)
    #     loss1 = mse_loss(d1, gt)
    #     loss2 = mse_loss(d2, gt)
    #     loss3 = mse_loss(d3, gt)
    #     loss4 = mse_loss(d4, gt)
    #     loss5 = mse_loss(d5, gt)
    #     loss6 = mse_loss(d6, gt)
    #
    #     loss10 = L1_loss(d0, gt)
    #     loss11 = L1_loss(d1, gt)
    #     loss12 = L1_loss(d2, gt)
    #     loss13 = L1_loss(d3, gt)
    #     loss14 = L1_loss(d4, gt)
    #     loss15 = L1_loss(d5, gt)
    #     loss16 = L1_loss(d6, gt)
    #     loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6  +loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16
    #
    #     # BP
    #     if is_optimizer:
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
    #         self.optimizer.step()
    #
    #     return d0, loss
    def optimize_strategy(self, img, gt, is_optimizer):
        # train generator#

        d0, d1, d2,d3 = self.model(img)
        mse_loss = get_mse_loss_function()
        L1_loss = get_l1_loss_function()

        loss0 = mse_loss(d0, gt)
        loss1 = mse_loss(d1, gt)
        loss2 = mse_loss(d2, gt)
        loss3 = mse_loss(d3, gt)


        loss10 = L1_loss(d0, gt)
        loss11 = L1_loss(d1, gt)
        loss12 = L1_loss(d2, gt)
        loss13 = L1_loss(d3, gt)


        loss = 10 *(loss0 + loss10) + 2 * (loss2 + loss12 + loss3 + loss13)

        # loss = loss0 + loss1 + loss2  +loss10 + loss11 + loss12 +loss13 +loss3

        # BP
        if is_optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

        return d0, loss


