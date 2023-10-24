#!/usr/bin/python3
#coding:utf-8


import torch
import torch.nn as nn

def get_softplus():

    if torch.cuda.is_available():
        softplus = nn.Softplus().cuda()
    else:
        softplus = nn.Softplus()
    return softplus

