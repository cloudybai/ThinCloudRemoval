# #!/usr/bin/python3
# # coding:utf-8
#
# """
#     Author: Xie Song
#     Email: 18406508513@163.com
#
#     Copyright: Xie Song
#     License: MIT
# """
#RSAUNet
'''
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class REBNCONV(nn.Module):
    def __init__(self,in_ch=9,out_ch=9,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
class REBNDECONV(nn.Module):
    def __init__(self,in_ch=16,out_ch=16):
        super(REBNDECONV,self).__init__()

        self.deconv_s1 = nn.ConvTranspose2d(in_ch,out_ch,4,2,1)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.deconv_s1(hx)))
        return xout


class irnn_layer(nn.Module):
    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)

    def forward(self, x):
        _, _, H, W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:, :, :, 1:] = F.relu(self.left_weight(x)[:, :, :, :W - 1] + x[:, :, :, 1:], inplace=False)
        top_right[:, :, :, :-1] = F.relu(self.right_weight(x)[:, :, :, 1:] + x[:, :, :, :W - 1], inplace=False)
        top_up[:, :, 1:, :] = F.relu(self.up_weight(x)[:, :, :H - 1, :] + x[:, :, 1:, :], inplace=False)
        top_down[:, :, :-1, :] = F.relu(self.down_weight(x)[:, :, 1:, :] + x[:, :, :H - 1, :], inplace=False)
        return (top_up, top_right, top_down, top_left)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask


class RSAUNet(nn.Module):

    def __init__(self, in_ch=9, mid_ch=16, out_ch=9):
        super(RSAUNet,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)


        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)


        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)


        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)


        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.SAM1 = SAM(16,16,1)


        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv6dup = REBNDECONV(16,16)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5dup = REBNDECONV(16,16)


        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv4dup = REBNDECONV(16,16)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv3dup = REBNDECONV(16,16)

        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv2dup = REBNDECONV(16,16)

        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)


        Attention1 = self.SAM1(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6 * Attention1),1))
        hx6dup = self.rebnconv6dup(hx6d)

        Attention2= self.SAM1(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5 * Attention2),1))
        hx5dup = self.rebnconv5dup(hx5d)

        Attention3= self.SAM1(hx4)


        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4*Attention3),1))
        hx4dup = self.rebnconv4dup(hx4d)

        Attention4= self.SAM1(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3 * Attention4),1))
        hx3dup = self.rebnconv3dup(hx3d)

        Attention5= self.SAM1(hx2)


        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2 * Attention5),1))
        hx2dup = self.rebnconv2dup(hx2d)

        Attention6= self.SAM1(hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1 * Attention6),1))

        return hx1d + hxin
'''

'''
#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""

# RSAUNet2
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict


def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class REBNCONV(nn.Module):
    def __init__(self,in_ch=9,out_ch=9,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
class REBNDECONV(nn.Module):
    def __init__(self,in_ch=16,out_ch=16):
        super(REBNDECONV,self).__init__()

        self.deconv_s1 = nn.ConvTranspose2d(in_ch,out_ch,4,2,1)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.deconv_s1(hx)))
        return xout


class irnn_layer(nn.Module):
    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)

    def forward(self, x):
        _, _, H, W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:, :, :, 1:] = F.relu(self.left_weight(x)[:, :, :, :W - 1] + x[:, :, :, 1:], inplace=False)
        top_right[:, :, :, :-1] = F.relu(self.right_weight(x)[:, :, :, 1:] + x[:, :, :, :W - 1], inplace=False)
        top_up[:, :, 1:, :] = F.relu(self.up_weight(x)[:, :, :H - 1, :] + x[:, :, 1:, :], inplace=False)
        top_down[:, :, :-1, :] = F.relu(self.down_weight(x)[:, :, 1:, :] + x[:, :, :H - 1, :], inplace=False)
        return (top_up, top_right, top_down, top_left)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out

class RSAUNet(nn.Module):

    def __init__(self, in_ch=9, mid_ch=16, out_ch=9):
        super(RSAUNet,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)


        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)


        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)


        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.Conv2d(mid_ch,mid_ch,3,stride=2,padding=1)


        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.SAM1 = SAM(16,16,1)


        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv6dup = REBNDECONV(16,16)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5dup = REBNDECONV(16,16)


        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv4dup = REBNDECONV(16,16)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv3dup = REBNDECONV(16,16)

        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv2dup = REBNDECONV(16,16)

        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
        
        self.res_block4 = Bottleneck(16,16)
        self.res_block5 = Bottleneck(16,16)
        self.res_block6 = Bottleneck(16,16)
        self.res_block7 = Bottleneck(16,16)
        self.res_block8 = Bottleneck(16,16)
        self.res_block9 = Bottleneck(16,16)
        self.res_block10 = Bottleneck(16,16)
        self.res_block11 = Bottleneck(16,16)
        self.res_block12 = Bottleneck(16,16)
        self.res_block13 = Bottleneck(16,16)
        self.res_block14 = Bottleneck(16,16)
        self.res_block15 = Bottleneck(16,16)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        Attention1 = self.SAM1(hx6)
        out = F.relu(self.res_block4(hx6) * Attention1 + hx6)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)

        Attention2 = self.SAM1(out)
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)

        Attention3 = self.SAM1(out)
        out = F.relu(self.res_block10(out) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12(out) * Attention3 + out)

        Attention4 = self.SAM1(out)
        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)

        hx7 = self.rebnconv7(out)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6 ),1))
        hx6dup = self.rebnconv6dup(hx6d)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5 ),1))
        hx5dup = self.rebnconv5dup(hx5d)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = self.rebnconv4dup(hx4d)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3 ),1))
        hx3dup = self.rebnconv3dup(hx3d)


        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2 ),1))
        hx2dup = self.rebnconv2dup(hx2d)


        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1 ),1))

        return hx1d + hxin
'''


#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""

# RSAUNet3

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict


def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class REBNCONV(nn.Module):
    def __init__(self,in_ch=9,out_ch=9,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
class REBNDECONV(nn.Module):
    def __init__(self,in_ch=16,out_ch=16):
        super(REBNDECONV,self).__init__()

        self.deconv_s1 = nn.ConvTranspose2d(in_ch,out_ch,3,1,1)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.deconv_s1(hx)))
        return xout


class irnn_layer(nn.Module):
    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)

    def forward(self, x):
        _, _, H, W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:, :, :, 1:] = F.relu(self.left_weight(x)[:, :, :, :W - 1] + x[:, :, :, 1:], inplace=False)
        top_right[:, :, :, :-1] = F.relu(self.right_weight(x)[:, :, :, 1:] + x[:, :, :, :W - 1], inplace=False)
        top_up[:, :, 1:, :] = F.relu(self.up_weight(x)[:, :, :H - 1, :] + x[:, :, 1:, :], inplace=False)
        top_down[:, :, :-1, :] = F.relu(self.down_weight(x)[:, :, 1:, :] + x[:, :, :H - 1, :], inplace=False)
        return (top_up, top_right, top_down, top_left)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out

class RSAUNet(nn.Module):

    def __init__(self, in_ch=9, mid_ch=16, out_ch=9):
        super(RSAUNet,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.Conv2d(mid_ch,mid_ch,3,stride=1,padding=1)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.Conv2d(mid_ch,mid_ch,3,stride=1,padding=1)


        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.Conv2d(mid_ch,mid_ch,3,stride=1,padding=1)


        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.Conv2d(mid_ch,mid_ch,3,stride=1,padding=1)


        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.Conv2d(mid_ch,mid_ch,3,stride=1,padding=1)


        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.SAM1 = SAM(16,16,1)


        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv6dup = REBNDECONV(16,16)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5dup = REBNDECONV(16,16)


        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv4dup = REBNDECONV(16,16)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv3dup = REBNDECONV(16,16)

        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)

        self.rebnconv2dup = REBNDECONV(16,16)

        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

        self.res_block4 = Bottleneck(16,16)
        self.res_block5 = Bottleneck(16,16)
        self.res_block6 = Bottleneck(16,16)
        self.res_block7 = Bottleneck(16,16)
        self.res_block8 = Bottleneck(16,16)
        self.res_block9 = Bottleneck(16,16)
        self.res_block10 = Bottleneck(16,16)
        self.res_block11 = Bottleneck(16,16)
        self.res_block12 = Bottleneck(16,16)
        self.res_block13 = Bottleneck(16,16)
        self.res_block14 = Bottleneck(16,16)
        self.res_block15 = Bottleneck(16,16)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        Attention1 = self.SAM1(hx6)
        out = F.relu(self.res_block4(hx6) * Attention1 + hx6)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)

        Attention2 = self.SAM1(out)
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)

        Attention3 = self.SAM1(out)
        out = F.relu(self.res_block10(out) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12(out) * Attention3 + out)

        Attention4 = self.SAM1(out)
        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)

        hx7 = self.rebnconv7(out)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6 ),1))
        hx6dup = self.rebnconv6dup(hx6d)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5 ),1))
        hx5dup = self.rebnconv5dup(hx5d)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = self.rebnconv4dup(hx4d)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3 ),1))
        hx3dup = self.rebnconv3dup(hx3d)


        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2 ),1))
        hx2dup = self.rebnconv2dup(hx2d)


        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1 ),1))

        return Attention1,hx1d


