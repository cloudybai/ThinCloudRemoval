#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

def conv(inplanes, outplanes, s=1, k=1, p=0):
    return nn.Conv2d(inplanes, outplanes, kernel_size=k, stride=s, padding=p, bias=True)

def deconv(inplanes, outplanes, s=1, k=1, p=0):
    return nn.ConvTranspose2d(inplanes, outplanes, kernel_size=k, stride=s, padding=p, bias=True)


def residual_conv(inplanes, outplanes):
    conv_block = nn.Sequential(
        nn.Conv2d(inplanes, inplanes, kernel_size=3, stride= 1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True),
    )
    return conv_block

def residual_deconv(inplanes, outplanes):
    deconv_block = nn.Sequential(
        nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, stride= 1, padding=1, bias=True),
        nn.ReLU(),
        nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True),
    )
    return deconv_block

class Block_skip(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = residual_conv(channels, channels)
        self.conv2 = residual_conv(channels, channels)
        self.conv3 = residual_conv(channels, channels)
        self.conv4 = residual_conv(channels, channels)
        self.conv5 = residual_conv(channels, channels)

        self.map = conv(channels, channels,s=1,k=1,p=0)

        self.deconv1 = residual_deconv(channels, channels)
        self.deconv2 = residual_deconv(2 * channels,channels)
        self.deconv3 = residual_deconv(2 * channels,channels)
        self.deconv4 = residual_deconv(2 * channels,channels)
        self.deconv5 = residual_deconv(2 * channels,channels)

    def forward(self, x):
        l_1 = self.conv1(x)
        residual1 = F.relu(torch.add(x, l_1))


        l_2 = self.conv2(residual1)
        residual2 = F.relu(torch.add(residual1, l_2))


        l_3 = self.conv3(residual2)
        residual3 = F.relu(torch.add(residual2, l_3))


        l_4 = self.conv4(residual3)
        residual4 = F.relu(torch.add(residual3, l_4))


        l_5 = self.conv5(residual4)
        residual5 = F.relu(torch.add(residual4, l_5))

        map = F.relu(self.map(residual5))

        r_1 = self.deconv1(map)
        residual6 = F.relu(torch.add(map, r_1))
        com1 = torch.cat((residual6, residual5), 1)

        r_2 = self.deconv2(com1)
        residual7 = F.relu(torch.add(residual6, r_2))
        com2 = torch.cat((residual7, residual4), 1)

        r_3 = self.deconv3(com2)
        residual8 = F.relu(torch.add(residual7, r_3))
        com3 = torch.cat((residual8, residual3), 1)

        r_4 = self.deconv4(com3)
        residual9 = F.relu(torch.add(residual8, r_4))
        com4 = torch.cat((residual9, residual2), 1)


        r_5 = self.deconv5(com4)
        residual10 = F.relu(torch.add(residual9, r_5))
        com5 = torch.cat((residual10, residual1), 1)
        return com5

class Multi_pix2pix(nn.Module):
    def __init__(self):
        #input 256*256
        super(Multi_pix2pix, self).__init__()
        self.input1 = conv(9, 16,s=1, k=3, p=1)
        self.downsample1 = conv(16,16,s=2,k=3,p=1)
        self.block1 = Block_skip(16)
        self.output1 = deconv(32, 9, s=2, k=4, p=1)

        #input 128*128
        self.input2 = conv(9, 16,s=1, k=3, p=1)
        self.downsample2 = conv(16,16,s=2,k=3,p=1)
        self.block2 = Block_skip(16)
        self.increace = conv(9, 16,s=1,k=1,p=0)
        self.output2 = deconv(32, 9, s=2, k=4, p=1)

    def forward(self, x1,x2):

        # input 128*128
        x2 = F.relu(self.input2(x2))
        downsample_2 = F.relu(self.downsample2(x2))
        l2= self.block2(downsample_2)
        o2 = self.output2(l2)

        # input 256*256
        x = F.relu(self.input1(x1))
        downsample_1 = F.relu(torch.add(self.downsample1(x),self.increace(o2)))
        l1 = self.block1(downsample_1)
        o1 = self.output1(l1)

        return o1 , o2

'''
#bn/leakyrelu
import torch.nn as nn
import torch
import torch.nn.functional as F

def conv(inplanes, outplanes, s=1, k=1, p=0):
    return nn.Conv2d(inplanes, outplanes, kernel_size=k, stride=s, padding=p, bias=True)

def deconv(inplanes, outplanes, s=1, k=1, p=0):
    return nn.ConvTranspose2d(inplanes, outplanes, kernel_size=k, stride=s, padding=p, bias=True)

def bn(channles):
    return nn.BatchNorm2d(channles,affine=True)

def residual_conv(inplanes, outplanes):
    conv_block = nn.Sequential(
        nn.Conv2d(inplanes, inplanes, kernel_size=3, stride= 1, padding=1, bias=True),
        nn.BatchNorm2d(inplanes, affine=True),
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(outplanes, affine=True),

    )
    return conv_block

def residual_deconv(inplanes, outplanes):
    deconv_block = nn.Sequential(
        nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, stride= 1, padding=1, bias=True),
        nn.BatchNorm2d(inplanes, affine=True),
        nn.LeakyReLU(0.2, True),
        nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(outplanes, affine=True),

    )
    return deconv_block

class Block_skip(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = residual_conv(channels, channels)
        self.conv2 = residual_conv(channels, channels)
        self.conv3 = residual_conv(channels, channels)
        self.conv4 = residual_conv(channels, channels)
        self.conv5 = residual_conv(channels, channels)

        self.map = conv(channels, channels,s=1,k=1,p=0)
        self.reduce = conv(2 * channels, channels,s=1,k=1,p=0)

        self.bn = nn.BatchNorm2d(channels)

        self.deconv1 = residual_deconv(channels, channels)
        self.deconv2 = residual_deconv(2 * channels,channels)
        self.deconv3 = residual_deconv(2 * channels,channels)
        self.deconv4 = residual_deconv(2 * channels,channels)
        self.deconv5 = residual_deconv(2 * channels,channels)

    def forward(self, x):
        l_1 = self.conv1(x)
        residual1 = F.leaky_relu(torch.add(x, l_1),0.2,True)

        l_2 = self.conv2(residual1)
        residual2 = F.leaky_relu(torch.add(residual1, l_2),0.2,True)

        l_3 = self.conv3(residual2)
        residual3 = F.leaky_relu(torch.add(residual2, l_3),0.2,True)

        l_4 = self.conv4(residual3)
        residual4 = F.leaky_relu(torch.add(residual3, l_4),0.2,True)


        l_5 = self.conv5(residual4)
        residual5 = F.leaky_relu(torch.add(residual4, l_5),0.2,True)

        map = F.leaky_relu(self.bn(self.map(residual5)),0.2,True)

        r_1 = self.deconv1(map)
        residual6 = F.leaky_relu(torch.add(map, r_1),0.2,True)
        com1 = torch.cat((residual6, residual5), 1)

        r_2 = self.deconv2(com1)
        residual7 = F.leaky_relu(torch.add(self.reduce(com1), r_2),0.2,True)
        com2 = torch.cat((residual7, residual4), 1)

        r_3 = self.deconv3(com2)
        residual8 = F.leaky_relu(torch.add(self.reduce(com2), r_3),0.2,True)
        com3 = torch.cat((residual8, residual3), 1)

        r_4 = self.deconv4(com3)
        residual9 = F.leaky_relu(torch.add(self.reduce(com3), r_4),0.2,True)
        com4 = torch.cat((residual9, residual2), 1)

        r_5 = self.deconv5(com4)
        residual10 = F.leaky_relu(torch.add(self.reduce(com4), r_5),0.2,True)
        com5 = torch.cat((residual10, residual1), 1)
        return com5

class Multi_pix2pix(nn.Module):
    def __init__(self):
        #input 256*256
        super(Multi_pix2pix, self).__init__()
        self.input1 = conv(9, 16,s=1, k=3, p=1)
        self.bn1 = bn(16)
        self.downsample1 = conv(16,16,s=2,k=3,p=1)
        self.block1 = Block_skip(16)
        self.output1_1 = deconv(32, 16, s=2, k=4, p=1)
        self.output2_1 = conv(16, 9, s=1, k=3, p=1)

        #input 128*128
        self.input2 = conv(9, 16,s=1, k=3, p=1)
        self.bn2 = bn(16)
        self.downsample2 = conv(16,16,s=2,k=3,p=1)
        self.block2 = Block_skip(16)
        self.increace = conv(9, 16,s=1,k=1,p=0)
        self.output1_2 = deconv(32, 16, s=2, k=4, p=1)
        self.output2_2 = conv(16, 9, s=1, k=3, p=1)

    def forward(self, x1,x2):

        # input 128*128
        x2 = self.input2(x2)
        x2 = F.leaky_relu(self.bn2(x2),0.2,True)
        downsample_2 = self.downsample2(x2)
        downsample_2 = F.leaky_relu(self.bn2(downsample_2),0.2,True)
        l2= self.block2(downsample_2)
        o2 = self.output2_2(self.output1_2(l2))

        # input 256*256
        x = self.input1(x1)
        x = F.leaky_relu(self.bn1(x),0.2,True)
        downsample_1 = self.downsample1(x)
        downsample_1 = F.leaky_relu(self.bn1(downsample_1 + self.increace(o2)),0.2,True)
        l1 = self.block1(downsample_1)
        o1 = self.output2_1(self.output1_1(l1))

        return o1 + x1, o2

'''