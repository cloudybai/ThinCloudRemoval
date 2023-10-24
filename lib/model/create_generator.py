#!/usr/bin/python3
#coding:utf-8


import torch
import torch.nn as nn
import math

def print_network(model):
    """

    :param model:
    :return: print network structure
    """
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(model)
    print('Total number of parameters: %d' % num_params)



# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm2d') != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weight_init_Xavier(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels * (1 + 0.2 * 0.2)
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels * (1 + 0.2 * 0.2)
            m.weight.data.normal_(0, math.sqrt(2. / n))

def create_generator(args,generator):

    print_network(generator)
    if torch.cuda.is_available():
        generator =generator.cuda()
    if args.mGPU:
        generator = nn.DataParallel(generator,args.device_ids)
    if args.weight_init == 'Xavier':
        weight_init_Xavier(generator)
    if args.weight_init == 'mean':
        weights_init(generator)

    if args.start_epoch != 0:
        generator.load_state_dict(torch.load(args.checkpoint))

    return generator