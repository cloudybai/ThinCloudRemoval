#!/usr/bin/python3
#coding:utf-8


import torch

def get_mse_loss_function():
    """

    :return: pixelwise_loss
    """
    mse_loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        mse_loss.cuda()
    return mse_loss