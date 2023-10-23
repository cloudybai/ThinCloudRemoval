#!/usr/bin/python3
#coding:utf-8

import torch

def get_sgd_optimizer(args, model):
    opimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=1e-4)
    return opimizer