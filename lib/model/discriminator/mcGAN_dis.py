#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

from lib.model.layers import CBR
from lib.model.models_utils import weights_init, print_network



class _Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        #self.gpu_ids = gpu_ids

        self.c0_0 = CBR(in_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c0_1 = CBR(out_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = nn.Conv2d(512, 1, 3, 1, 1)

    def forward(self, x):
        x_0 = x[:, :4]
        x_1 = x[:, 4:]
        h = torch.cat((self.c0_0(x_0), self.c0_1(x_1)), 1)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        return h


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpu_ids = args.gpu_ids

        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator(args.in_ch, args.out_ch))]))

        self.dis.apply(weights_init)

    def forward(self, x):
        return nn.parallel.data_parallel(self.dis, x, self.gpu_ids)
