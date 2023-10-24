import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from lib.model.models_utils import weights_init, print_network


# import common

###### Layer
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv(in_channels, out_channels, stride = 2):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

def deconv(in_channels, out_channels, stride = 2,padding=1):
    return nn.ConvTranspose2d(in_channels,out_channels,kernel_size = 4,
        stride =stride, padding=padding,bias=False)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


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



def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return src
#MSPF_new
class MSPFNet(nn.Module):
    def __init__(self):
        super(MSPFNet, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.conv_in_128 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )
        self.conv_in_64 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.SAM1 = SAM(32, 32, 1)
        self.SAM12 = SAM(32, 32, 1)
        self.SAM13 = SAM(32, 32, 1)
        self.res_block1 = Bottleneck(32, 32)
        self.res_block12 = Bottleneck(32, 32)
        self.res_block123 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block22 = Bottleneck(32, 32)
        self.res_block223 = Bottleneck(32, 32)
        self.res_block3 = Bottleneck(32, 32)
        self.res_block32 = Bottleneck(32, 32)
        self.res_block323 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block42 = Bottleneck(32, 32)
        self.res_block423 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block52 = Bottleneck(32, 32)
        self.res_block523 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)
        self.res_block62 = Bottleneck(32, 32)
        self.res_block623 = Bottleneck(32, 32)
        self.res_block7 = Bottleneck(32, 32)
        self.res_block72 = Bottleneck(32, 32)
        self.res_block723 = Bottleneck(32, 32)
        self.res_block8 = Bottleneck(32, 32)
        self.res_block82 = Bottleneck(32, 32)
        self.res_block823 = Bottleneck(32, 32)
        self.res_block9 = Bottleneck(32, 32)
        self.res_block92 = Bottleneck(32, 32)
        self.res_block923 = Bottleneck(32, 32)
        self.res_block10 = Bottleneck(32, 32)
        self.res_block102 = Bottleneck(32, 32)
        self.res_block1023 = Bottleneck(32, 32)
        self.res_block11 = Bottleneck(32, 32)
        self.res_block112 = Bottleneck(32, 32)
        self.res_block1123 = Bottleneck(32, 32)
        self.res_block12_ = Bottleneck(32, 32)
        self.res_block122 = Bottleneck(32, 32)
        self.res_block1223 = Bottleneck(32, 32)
        self.res_block13 = Bottleneck(32, 32)
        self.res_block132 = Bottleneck(32, 32)
        self.res_block1323 = Bottleneck(32, 32)
        self.res_block14 = Bottleneck(32, 32)
        self.res_block142 = Bottleneck(32, 32)
        self.res_block1423 = Bottleneck(32, 32)
        self.res_block15 = Bottleneck(32, 32)
        self.res_block152 = Bottleneck(32, 32)
        self.res_block1523 = Bottleneck(32, 32)
        self.res_block16 = Bottleneck(64, 32)
        self.res_block162 = Bottleneck(64, 32)
        self.res_block1623 = Bottleneck(64, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.res_block172 = Bottleneck(32, 32)
        self.res_block1723 = Bottleneck(32, 32)
        self.conv_out_128 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_64 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_256 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )

        self.deconv_out_128 = nn.Sequential(
            deconv(9, 9, 2),
            nn.ReLU(True)
        )
        self.deconv_out_64 = nn.Sequential(
            deconv(9,9,4,0),
            nn.ReLU(True)
        )

        self.conv_out = nn.Sequential(
            conv3x3(27, 9)
        )


    def forward(self, x1,x2,x3):

        out_in = self.conv_in(x1)#32 * 256 *256
        out2_in = self.conv_in_128(x2)  #32 * 128 *128
        out3_in = self.conv_in_64(x3)  #32 * 64 *64

        out3 = F.relu(self.res_block123(out3_in) + out3_in)
        out2 = F.relu(self.res_block12(out2_in+_upsample_like(out3_in,out2_in)) + out2_in)
        out = F.relu(self.res_block1(out_in + _upsample_like(out2_in,out_in) + _upsample_like(out3_in,out_in)) + out_in)

        out3_C = F.relu(self.res_block223(out3) + out3)
        out2_C = F.relu(self.res_block22(out2+_upsample_like(out3,out2)) + out2)
        out_C = F.relu(self.res_block2(out + _upsample_like(out2,out) + _upsample_like(out3,out)) + out)

        # out3 = F.relu(self.res_block323(out3) + out3)
        # out2 = F.relu(self.res_block32(out2+_upsample_like(out3,out2)) + out2)
        # out = F.relu(self.res_block3(out + _upsample_like(out2,out) + _upsample_like(out3,out)) + out)

        Attention1 = self.SAM1(out_C)
        Attention12 = self.SAM12(out2_C)
        Attention123 = self.SAM13(out3_C)
        out3 = F.relu(self.res_block423(out3_C) * Attention123 + out3_C)
        out3 = F.relu(self.res_block523(out3) * Attention123 + out3)
        out3 = F.relu(self.res_block623(out3) * Attention123 + out3)

        out2 = F.relu(self.res_block42(out2_C + _upsample_like(out3, out2)) * Attention12 + out2_C)
        out2 = F.relu(self.res_block52(out2) * Attention12 + out2)
        out2 = F.relu(self.res_block62(out2) * Attention12 + out2)

        out = F.relu(self.res_block4(out_C + _upsample_like(out2,out) + _upsample_like(out3,out)) * Attention1 + out_C)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)




        Attention2 = self.SAM1(out)
        Attention22 = self.SAM12(out2)
        Attention223 = self.SAM13(out3)

        out3 = F.relu(self.res_block723(out3) * Attention223 + out3)
        out3 = F.relu(self.res_block823(out3) * Attention223 + out3)
        out3 = F.relu(self.res_block923(out3) * Attention223 + out3)

        out2 = F.relu(self.res_block72(out2 + _upsample_like(out3, out2)) * Attention22 + out2)
        out2 = F.relu(self.res_block82(out2) * Attention22 + out2)
        out2 = F.relu(self.res_block92(out2) * Attention22 + out2)

        out = F.relu(self.res_block7(out + _upsample_like(out2,out) + _upsample_like(out3,out)) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)

        Attention3 = self.SAM1(out)
        Attention32 = self.SAM12(out2)
        Attention323 = self.SAM13(out3)

        out3 = F.relu(self.res_block1023(out3) * Attention323 + out3)
        out3 = F.relu(self.res_block1123(out3) * Attention323 + out3)
        out3 = F.relu(self.res_block1223(out3) * Attention323 + out3)

        out2 = F.relu(self.res_block102(out2 + _upsample_like(out3, out2)) * Attention32 + out2)
        out2 = F.relu(self.res_block112(out2) * Attention32 + out2)
        out2 = F.relu(self.res_block122(out2) * Attention32 + out2)

        out = F.relu(self.res_block10(out + _upsample_like(out2,out) + _upsample_like(out3,out)) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12_(out) * Attention3 + out)

        Attention4 = self.SAM1(out)
        Attention42 = self.SAM12(out2)
        Attention423 = self.SAM13(out3)

        out3 = F.relu(self.res_block1323(out3) * Attention423 + out3)
        out3 = F.relu(self.res_block1423(out3) * Attention423 + out3)
        out3 = F.relu(self.res_block1523(out3) * Attention423 + out3)

        out2 = F.relu(self.res_block132(out2 + _upsample_like(out3, out2)) * Attention42 + out2)
        out2 = F.relu(self.res_block142(out2) * Attention42 + out2)
        out2 = F.relu(self.res_block152(out2) * Attention42 + out2)

        out = F.relu(self.res_block13(out + _upsample_like(out2,out) + _upsample_like(out3,out)) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)


       #RM

        out3_com = torch.cat((out3_C, out3), 1)
        out2_com = torch.cat((out2_C, out2), 1)
        out_com = torch.cat((out_C, out), 1)

        out3 = F.relu(self.res_block1623(out3_com) + out3)
        out2 = F.relu(self.res_block162(out2_com) + out2)
        out = F.relu(self.res_block16(out_com) + out)

        out3 = F.relu(self.res_block1723(out3) + out3)
        out2 = F.relu(self.res_block172(out2+_upsample_like(out3,out2)) + out2)
        out = F.relu(self.res_block17(out + _upsample_like(out2,out) + _upsample_like(out3,out)) + out)


        out2_ = self.conv_out_128(out2 + out2_in)  #9*256*256
        out3_ = self.conv_out_64(out3 + out3_in)   #9*256*256

        out1_com = torch.cat((self.conv_out_256(out+out_in) , self.deconv_out_128(out2_) ,self.deconv_out_64(out3_)),1)

        out1 = self.conv_out(out1_com)

        return Attention4, Attention42,Attention423, out1,out2_,out3_

#MSEF fusion in end
class MSEFNet(nn.Module):
    def __init__(self):
        super(MSEFNet, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.conv_in_128 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )
        self.conv_in_64 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.SAM1 = SAM(32, 32, 1)
        self.SAM12 = SAM(32, 32, 1)
        self.SAM13 = SAM(32, 32, 1)
        self.res_block1 = Bottleneck(32, 32)
        self.res_block12 = Bottleneck(32, 32)
        self.res_block123 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block22 = Bottleneck(32, 32)
        self.res_block223 = Bottleneck(32, 32)
        self.res_block3 = Bottleneck(32, 32)
        self.res_block32 = Bottleneck(32, 32)
        self.res_block323 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block42 = Bottleneck(32, 32)
        self.res_block423 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block52 = Bottleneck(32, 32)
        self.res_block523 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)
        self.res_block62 = Bottleneck(32, 32)
        self.res_block623 = Bottleneck(32, 32)
        self.res_block7 = Bottleneck(32, 32)
        self.res_block72 = Bottleneck(32, 32)
        self.res_block723 = Bottleneck(32, 32)
        self.res_block8 = Bottleneck(32, 32)
        self.res_block82 = Bottleneck(32, 32)
        self.res_block823 = Bottleneck(32, 32)
        self.res_block9 = Bottleneck(32, 32)
        self.res_block92 = Bottleneck(32, 32)
        self.res_block923 = Bottleneck(32, 32)
        self.res_block10 = Bottleneck(32, 32)
        self.res_block102 = Bottleneck(32, 32)
        self.res_block1023 = Bottleneck(32, 32)
        self.res_block11 = Bottleneck(32, 32)
        self.res_block112 = Bottleneck(32, 32)
        self.res_block1123 = Bottleneck(32, 32)
        self.res_block12_ = Bottleneck(32, 32)
        self.res_block122 = Bottleneck(32, 32)
        self.res_block1223 = Bottleneck(32, 32)
        self.res_block13 = Bottleneck(32, 32)
        self.res_block132 = Bottleneck(32, 32)
        self.res_block1323 = Bottleneck(32, 32)
        self.res_block14 = Bottleneck(32, 32)
        self.res_block142 = Bottleneck(32, 32)
        self.res_block1423 = Bottleneck(32, 32)
        self.res_block15 = Bottleneck(32, 32)
        self.res_block152 = Bottleneck(32, 32)
        self.res_block1523 = Bottleneck(32, 32)
        self.res_block16 = Bottleneck(32, 32)
        self.res_block162 = Bottleneck(32, 32)
        self.res_block1623 = Bottleneck(32, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.res_block172 = Bottleneck(32, 32)
        self.res_block1723 = Bottleneck(32, 32)
        self.conv_out_128 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_64 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_256 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )

        self.deconv_out_128 = nn.Sequential(
            deconv(9, 9, 2),
            nn.ReLU(True)
        )
        self.deconv_out_64 = nn.Sequential(
            deconv(9,9,4,0),
            nn.ReLU(True)
        )

        self.conv_out = nn.Sequential(
            conv3x3(27, 9)
        )


    def forward(self, x1,x2,x3):

        out_in = self.conv_in(x1)#32 * 256 *256
        out2_in = self.conv_in_128(x2)  #32 * 128 *128
        out3_in = self.conv_in_64(x3)  #32 * 64 *64

        out3 = F.relu(self.res_block123(out3_in) + out3_in)
        out2 = F.relu(self.res_block12(out2_in) + out2_in)
        out = F.relu(self.res_block1(out_in ) + out_in)

        out3_C = F.relu(self.res_block223(out3) + out3)
        out2_C = F.relu(self.res_block22(out2) + out2)
        out_C = F.relu(self.res_block2(out) + out)



        Attention1 = self.SAM1(out_C)
        Attention12 = self.SAM12(out2_C)
        Attention123 = self.SAM13(out3_C)
        out3 = F.relu(self.res_block423(out3_C) * Attention123 + out3_C)
        out3 = F.relu(self.res_block523(out3) * Attention123 + out3)
        out3 = F.relu(self.res_block623(out3) * Attention123 + out3)

        out2 = F.relu(self.res_block42(out2_C ) * Attention12 + out2_C)
        out2 = F.relu(self.res_block52(out2) * Attention12 + out2)
        out2 = F.relu(self.res_block62(out2) * Attention12 + out2)

        out = F.relu(self.res_block4(out_C ) * Attention1 + out_C)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)




        Attention2 = self.SAM1(out)
        Attention22 = self.SAM12(out2)
        Attention223 = self.SAM13(out3)

        out3 = F.relu(self.res_block723(out3) * Attention223 + out3)
        out3 = F.relu(self.res_block823(out3) * Attention223 + out3)
        out3 = F.relu(self.res_block923(out3) * Attention223 + out3)

        out2 = F.relu(self.res_block72(out2 ) * Attention22 + out2)
        out2 = F.relu(self.res_block82(out2) * Attention22 + out2)
        out2 = F.relu(self.res_block92(out2) * Attention22 + out2)

        out = F.relu(self.res_block7(out ) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)

        Attention3 = self.SAM1(out)
        Attention32 = self.SAM12(out2)
        Attention323 = self.SAM13(out3)

        out3 = F.relu(self.res_block1023(out3) * Attention323 + out3)
        out3 = F.relu(self.res_block1123(out3) * Attention323 + out3)
        out3 = F.relu(self.res_block1223(out3) * Attention323 + out3)

        out2 = F.relu(self.res_block102(out2 ) * Attention32 + out2)
        out2 = F.relu(self.res_block112(out2) * Attention32 + out2)
        out2 = F.relu(self.res_block122(out2) * Attention32 + out2)

        out = F.relu(self.res_block10(out ) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12_(out) * Attention3 + out)

        Attention4 = self.SAM1(out)
        Attention42 = self.SAM12(out2)
        Attention423 = self.SAM13(out3)

        out3 = F.relu(self.res_block1323(out3) * Attention423 + out3)
        out3 = F.relu(self.res_block1423(out3) * Attention423 + out3)
        out3 = F.relu(self.res_block1523(out3) * Attention423 + out3)

        out2 = F.relu(self.res_block132(out2 ) * Attention42 + out2)
        out2 = F.relu(self.res_block142(out2) * Attention42 + out2)
        out2 = F.relu(self.res_block152(out2) * Attention42 + out2)

        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)


       #RM


        out3 = F.relu(self.res_block1623(out3) + out3)
        out2 = F.relu(self.res_block162(out2) + out2)
        out = F.relu(self.res_block16(out) + out)

        out3 = F.relu(self.res_block1723(out3) + out3)
        out2 = F.relu(self.res_block172(out2+_upsample_like(out3,out2)) + out2)
        out = F.relu(self.res_block17(out + _upsample_like(out2,out) + _upsample_like(out3,out)) + out)


        out2_ = self.conv_out_128(out2 + out2_in)  #9*256*256
        out3_ = self.conv_out_64(out3 + out3_in)   #9*256*256

        out1_com = torch.cat((self.conv_out_256(out+out_in) , self.deconv_out_128(out2_) ,self.deconv_out_64(out3_)),1)

        out1 = self.conv_out(out1_com)

        return Attention4, Attention42,Attention423, out1,out2_,out3_

#SS single scale
class SSNet(nn.Module):
    def __init__(self):
        super(SSNet, self).__init__()

        self.conv_in = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )
        self.SAM1 = SAM(32, 32, 1)
        self.res_block1 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block3 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)
        self.res_block7 = Bottleneck(32, 32)
        self.res_block8 = Bottleneck(32, 32)
        self.res_block9 = Bottleneck(32, 32)
        self.res_block10 = Bottleneck(32, 32)
        self.res_block11 = Bottleneck(32, 32)
        self.res_block12 = Bottleneck(32, 32)
        self.res_block13 = Bottleneck(32, 32)
        self.res_block14 = Bottleneck(32, 32)
        self.res_block15 = Bottleneck(32, 32)
        self.res_block16 = Bottleneck(64, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.conv_out = nn.Sequential(
            conv3x3(32, 9)
        )

    def forward(self, x):
        out_in = self.conv_in(x)
        out = F.relu(self.res_block1(out_in) + out_in)
        out_C = F.relu(self.res_block2(out) + out)

        Attention1 = self.SAM1(out_C)
        out = F.relu(self.res_block4(out_C) * Attention1 + out_C)
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

        out_com = torch.cat((out_C, out), 1)

        out = F.relu(self.res_block16(out_com) + out)
        out = F.relu(self.res_block17(out) + out)
        out = self.conv_out(out + out_in)

        return Attention4, out


#OSPF threee scale is same
class OSPFNet(nn.Module):
    def __init__(self):
        super(OSPFNet, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.conv_in_128 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )
        self.conv_in_64 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.SAM1 = SAM(32, 32, 1)
        self.SAM12 = SAM(32, 32, 1)
        self.SAM13 = SAM(32, 32, 1)
        self.res_block1 = Bottleneck(32, 32)
        self.res_block12 = Bottleneck(32, 32)
        self.res_block123 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block22 = Bottleneck(32, 32)
        self.res_block223 = Bottleneck(32, 32)
        self.res_block3 = Bottleneck(32, 32)
        self.res_block32 = Bottleneck(32, 32)
        self.res_block323 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block42 = Bottleneck(32, 32)
        self.res_block423 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block52 = Bottleneck(32, 32)
        self.res_block523 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)
        self.res_block62 = Bottleneck(32, 32)
        self.res_block623 = Bottleneck(32, 32)
        self.res_block7 = Bottleneck(32, 32)
        self.res_block72 = Bottleneck(32, 32)
        self.res_block723 = Bottleneck(32, 32)
        self.res_block8 = Bottleneck(32, 32)
        self.res_block82 = Bottleneck(32, 32)
        self.res_block823 = Bottleneck(32, 32)
        self.res_block9 = Bottleneck(32, 32)
        self.res_block92 = Bottleneck(32, 32)
        self.res_block923 = Bottleneck(32, 32)
        self.res_block10 = Bottleneck(32, 32)
        self.res_block102 = Bottleneck(32, 32)
        self.res_block1023 = Bottleneck(32, 32)
        self.res_block11 = Bottleneck(32, 32)
        self.res_block112 = Bottleneck(32, 32)
        self.res_block1123 = Bottleneck(32, 32)
        self.res_block12_ = Bottleneck(32, 32)
        self.res_block122 = Bottleneck(32, 32)
        self.res_block1223 = Bottleneck(32, 32)
        self.res_block13 = Bottleneck(32, 32)
        self.res_block132 = Bottleneck(32, 32)
        self.res_block1323 = Bottleneck(32, 32)
        self.res_block14 = Bottleneck(32, 32)
        self.res_block142 = Bottleneck(32, 32)
        self.res_block1423 = Bottleneck(32, 32)
        self.res_block15 = Bottleneck(32, 32)
        self.res_block152 = Bottleneck(32, 32)
        self.res_block1523 = Bottleneck(32, 32)
        self.res_block16 = Bottleneck(64, 32)
        self.res_block162 = Bottleneck(64, 32)
        self.res_block1623 = Bottleneck(64, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.res_block172 = Bottleneck(32, 32)
        self.res_block1723 = Bottleneck(32, 32)
        self.conv_out_128 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_64 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_256 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )

        self.deconv_out_128 = nn.Sequential(
            deconv(9, 9, 2),
            nn.ReLU(True)
        )
        self.deconv_out_64 = nn.Sequential(
            deconv(9,9,4,0),
            nn.ReLU(True)
        )

        self.conv_out = nn.Sequential(
            conv3x3(27, 9)
        )


    def forward(self, x1,x2,x3):

        out_in = self.conv_in(x1)#32 * 256 *256
        out2_in = self.conv_in_128(x2)  #32 * 128 *128
        out3_in = self.conv_in_64(x3)  #32 * 64 *64

        out3 = F.relu(self.res_block123(out3_in) + out3_in)
        out2 = F.relu(self.res_block12(out2_in+out3_in) + out2_in)
        out = F.relu(self.res_block1(out_in + out2_in +out3_in) + out_in)

        out3_C = F.relu(self.res_block223(out3) + out3)
        out2_C = F.relu(self.res_block22(out2 + out3) + out2)
        out_C = F.relu(self.res_block2(out + out2 + out3) + out)

        Attention1 = self.SAM1(out_C)
        Attention12 = self.SAM12(out2_C)
        Attention123 = self.SAM13(out3_C)
        out3 = F.relu(self.res_block423(out3_C) * Attention123 + out3_C)
        out3 = F.relu(self.res_block523(out3) * Attention123 + out3)
        out3 = F.relu(self.res_block623(out3) * Attention123 + out3)

        out2 = F.relu(self.res_block42(out2_C +out3) * Attention12 + out2_C)
        out2 = F.relu(self.res_block52(out2) * Attention12 + out2)
        out2 = F.relu(self.res_block62(out2) * Attention12 + out2)

        out = F.relu(self.res_block4(out_C +out2 +out3) * Attention1 + out_C)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)




        Attention2 = self.SAM1(out)
        Attention22 = self.SAM12(out2)
        Attention223 = self.SAM13(out3)

        out3 = F.relu(self.res_block723(out3) * Attention223 + out3)
        out3 = F.relu(self.res_block823(out3) * Attention223 + out3)
        out3 = F.relu(self.res_block923(out3) * Attention223 + out3)

        out2 = F.relu(self.res_block72(out2 + out3) * Attention22 + out2)
        out2 = F.relu(self.res_block82(out2) * Attention22 + out2)
        out2 = F.relu(self.res_block92(out2) * Attention22 + out2)

        out = F.relu(self.res_block7(out + out2 + out3) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)

        Attention3 = self.SAM1(out)
        Attention32 = self.SAM12(out2)
        Attention323 = self.SAM13(out3)

        out3 = F.relu(self.res_block1023(out3) * Attention323 + out3)
        out3 = F.relu(self.res_block1123(out3) * Attention323 + out3)
        out3 = F.relu(self.res_block1223(out3) * Attention323 + out3)

        out2 = F.relu(self.res_block102(out2 + out3) * Attention32 + out2)
        out2 = F.relu(self.res_block112(out2) * Attention32 + out2)
        out2 = F.relu(self.res_block122(out2) * Attention32 + out2)

        out = F.relu(self.res_block10(out + out2 + out3) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12_(out) * Attention3 + out)

        Attention4 = self.SAM1(out)
        Attention42 = self.SAM12(out2)
        Attention423 = self.SAM13(out3)

        out3 = F.relu(self.res_block1323(out3) * Attention423 + out3)
        out3 = F.relu(self.res_block1423(out3) * Attention423 + out3)
        out3 = F.relu(self.res_block1523(out3) * Attention423 + out3)

        out2 = F.relu(self.res_block132(out2 + out3) * Attention42 + out2)
        out2 = F.relu(self.res_block142(out2) * Attention42 + out2)
        out2 = F.relu(self.res_block152(out2) * Attention42 + out2)

        out = F.relu(self.res_block13(out +out2 + out3) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)


       #RM

        out3_com = torch.cat((out3_C, out3), 1)
        out2_com = torch.cat((out2_C, out2), 1)
        out_com = torch.cat((out_C, out), 1)

        out3 = F.relu(self.res_block1623(out3_com) + out3)
        out2 = F.relu(self.res_block162(out2_com) + out2)
        out = F.relu(self.res_block16(out_com) + out)

        out3 = F.relu(self.res_block1723(out3) + out3)
        out2 = F.relu(self.res_block172(out2+out3) + out2)
        out = F.relu(self.res_block17(out +out2 + out3) + out)


        out2_ = self.conv_out_128(out2 + out2_in)  #9*256*256
        out3_ = self.conv_out_64(out3 + out3_in)   #9*256*256

        out1_com = torch.cat((self.conv_out_256(out+out_in) , out2_ ,out3_),1)

        out1 = self.conv_out(out1_com)

        return Attention4, Attention42,Attention423, out1,out2_,out3_


# remove CFM --MSFF-Net
class MSFFNet(nn.Module):
    def __init__(self):
        super(MSFFNet, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.conv_in_128 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )
        self.conv_in_64 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.SAM1 = SAM(32, 32, 1)
        self.SAM12 = SAM(32, 32, 1)
        self.SAM13 = SAM(32, 32, 1)

        self.res_block3 = Bottleneck(32, 32)
        self.res_block32 = Bottleneck(32, 32)
        self.res_block323 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block42 = Bottleneck(32, 32)
        self.res_block423 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block52 = Bottleneck(32, 32)
        self.res_block523 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)
        self.res_block62 = Bottleneck(32, 32)
        self.res_block623 = Bottleneck(32, 32)
        self.res_block7 = Bottleneck(32, 32)
        self.res_block72 = Bottleneck(32, 32)
        self.res_block723 = Bottleneck(32, 32)
        self.res_block8 = Bottleneck(32, 32)
        self.res_block82 = Bottleneck(32, 32)
        self.res_block823 = Bottleneck(32, 32)
        self.res_block9 = Bottleneck(32, 32)
        self.res_block92 = Bottleneck(32, 32)
        self.res_block923 = Bottleneck(32, 32)
        self.res_block10 = Bottleneck(32, 32)
        self.res_block102 = Bottleneck(32, 32)
        self.res_block1023 = Bottleneck(32, 32)
        self.res_block11 = Bottleneck(32, 32)
        self.res_block112 = Bottleneck(32, 32)
        self.res_block1123 = Bottleneck(32, 32)
        self.res_block12_ = Bottleneck(32, 32)
        self.res_block122 = Bottleneck(32, 32)
        self.res_block1223 = Bottleneck(32, 32)
        self.res_block13 = Bottleneck(32, 32)
        self.res_block132 = Bottleneck(32, 32)
        self.res_block1323 = Bottleneck(32, 32)
        self.res_block14 = Bottleneck(32, 32)
        self.res_block142 = Bottleneck(32, 32)
        self.res_block1423 = Bottleneck(32, 32)
        self.res_block15 = Bottleneck(32, 32)
        self.res_block152 = Bottleneck(32, 32)
        self.res_block1523 = Bottleneck(32, 32)
        self.res_block16 = Bottleneck(32, 32)
        self.res_block162 = Bottleneck(32, 32)
        self.res_block1623 = Bottleneck(32, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.res_block172 = Bottleneck(32, 32)
        self.res_block1723 = Bottleneck(32, 32)
        self.conv_out_128 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_64 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_256 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )

        self.deconv_out_128 = nn.Sequential(
            deconv(9, 9, 2),
            nn.ReLU(True)
        )
        self.deconv_out_64 = nn.Sequential(
            deconv(9,9,4,0),
            nn.ReLU(True)
        )

        self.conv_out = nn.Sequential(
            conv3x3(27, 9)
        )


    def forward(self, x1,x2,x3):

        out_in = self.conv_in(x1)#32 * 256 *256
        out2_in = self.conv_in_128(x2)  #32 * 128 *128
        out3_in = self.conv_in_64(x3)  #32 * 64 *64



        Attention1 = self.SAM1(out_in)
        Attention12 = self.SAM12(out2_in)
        Attention123 = self.SAM13(out3_in)
        out3 = F.relu(self.res_block423(out3_in) * Attention123 + out3_in)
        out3 = F.relu(self.res_block523(out3) * Attention123 + out3)
        out3 = F.relu(self.res_block623(out3) * Attention123 + out3)

        out2 = F.relu(self.res_block42(out2_in + _upsample_like(out3, out2_in)) * Attention12 + out2_in)
        out2 = F.relu(self.res_block52(out2) * Attention12 + out2)
        out2 = F.relu(self.res_block62(out2) * Attention12 + out2)

        out = F.relu(self.res_block4(out_in + _upsample_like(out2,out_in) + _upsample_like(out3,out_in)) * Attention1 + out_in)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)




        Attention2 = self.SAM1(out)
        Attention22 = self.SAM12(out2)
        Attention223 = self.SAM13(out3)

        out3 = F.relu(self.res_block723(out3) * Attention223 + out3)
        out3 = F.relu(self.res_block823(out3) * Attention223 + out3)
        out3 = F.relu(self.res_block923(out3) * Attention223 + out3)

        out2 = F.relu(self.res_block72(out2 + _upsample_like(out3, out2)) * Attention22 + out2)
        out2 = F.relu(self.res_block82(out2) * Attention22 + out2)
        out2 = F.relu(self.res_block92(out2) * Attention22 + out2)

        out = F.relu(self.res_block7(out + _upsample_like(out2,out) + _upsample_like(out3,out)) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)

        Attention3 = self.SAM1(out)
        Attention32 = self.SAM12(out2)
        Attention323 = self.SAM13(out3)

        out3 = F.relu(self.res_block1023(out3) * Attention323 + out3)
        out3 = F.relu(self.res_block1123(out3) * Attention323 + out3)
        out3 = F.relu(self.res_block1223(out3) * Attention323 + out3)

        out2 = F.relu(self.res_block102(out2 + _upsample_like(out3, out2)) * Attention32 + out2)
        out2 = F.relu(self.res_block112(out2) * Attention32 + out2)
        out2 = F.relu(self.res_block122(out2) * Attention32 + out2)

        out = F.relu(self.res_block10(out + _upsample_like(out2,out) + _upsample_like(out3,out)) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12_(out) * Attention3 + out)

        Attention4 = self.SAM1(out)
        Attention42 = self.SAM12(out2)
        Attention423 = self.SAM13(out3)

        out3 = F.relu(self.res_block1323(out3) * Attention423 + out3)
        out3 = F.relu(self.res_block1423(out3) * Attention423 + out3)
        out3 = F.relu(self.res_block1523(out3) * Attention423 + out3)

        out2 = F.relu(self.res_block132(out2 + _upsample_like(out3, out2)) * Attention42 + out2)
        out2 = F.relu(self.res_block142(out2) * Attention42 + out2)
        out2 = F.relu(self.res_block152(out2) * Attention42 + out2)

        out = F.relu(self.res_block13(out + _upsample_like(out2,out) + _upsample_like(out3,out)) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)


       #RM

        out3 = F.relu(self.res_block1623(out3) + out3)
        out2 = F.relu(self.res_block162(out2) + out2)
        out = F.relu(self.res_block16(out) + out)

        out3 = F.relu(self.res_block1723(out3) + out3)
        out2 = F.relu(self.res_block172(out2+_upsample_like(out3,out2)) + out2)
        out = F.relu(self.res_block17(out + _upsample_like(out2,out) + _upsample_like(out3,out)) + out)


        out2_ = self.conv_out_128(out2 + out2_in)  #9*256*256
        out3_ = self.conv_out_64(out3 + out3_in)   #9*256*256

        out1_com = torch.cat((self.conv_out_256(out+out_in) , self.deconv_out_128(out2_) ,self.deconv_out_64(out3_)),1)

        out1 = self.conv_out(out1_com)

        return Attention4, Attention42,Attention423, out1,out2_,out3_

#remove FFM --MSF-Net
class MSCFNet(nn.Module):
    def __init__(self):
        super(MSCFNet, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.conv_in_128 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )
        self.conv_in_64 = nn.Sequential(
            conv3x3(9, 32),
            nn.ReLU(True)
        )

        self.SAM1 = SAM(32, 32, 1)
        self.SAM12 = SAM(32, 32, 1)
        self.SAM13 = SAM(32, 32, 1)
        self.res_block1 = Bottleneck(32, 32)
        self.res_block12 = Bottleneck(32, 32)
        self.res_block123 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block22 = Bottleneck(32, 32)
        self.res_block223 = Bottleneck(32, 32)

        self.res_block16 = Bottleneck(32, 32)
        self.res_block162 = Bottleneck(32, 32)
        self.res_block1623 = Bottleneck(32, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.res_block172 = Bottleneck(32, 32)
        self.res_block1723 = Bottleneck(32, 32)
        self.conv_out_128 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_64 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )
        self.conv_out_256 = nn.Sequential(
            conv3x3(32, 9),
            nn.ReLU(True)
        )

        self.deconv_out_128 = nn.Sequential(
            deconv(9, 9, 2),
            nn.ReLU(True)
        )
        self.deconv_out_64 = nn.Sequential(
            deconv(9,9,4,0),
            nn.ReLU(True)
        )

        self.conv_out = nn.Sequential(
            conv3x3(27, 9)
        )


    def forward(self, x1,x2,x3):

        out_in = self.conv_in(x1)#32 * 256 *256
        out2_in = self.conv_in_128(x2)  #32 * 128 *128
        out3_in = self.conv_in_64(x3)  #32 * 64 *64

        out3 = F.relu(self.res_block123(out3_in) + out3_in)
        out2 = F.relu(self.res_block12(out2_in+_upsample_like(out3_in,out2_in)) + out2_in)
        out = F.relu(self.res_block1(out_in + _upsample_like(out2_in,out_in) + _upsample_like(out3_in,out_in)) + out_in)

        out3 = F.relu(self.res_block223(out3) + out3)
        out2 = F.relu(self.res_block22(out2+_upsample_like(out3,out2)) + out2)
        out = F.relu(self.res_block2(out + _upsample_like(out2,out) + _upsample_like(out3,out)) + out)



        out3 = F.relu(self.res_block1623(out3) + out3)
        out2 = F.relu(self.res_block162(out2) + out2)
        out = F.relu(self.res_block16(out) + out)

        out3 = F.relu(self.res_block1723(out3) + out3)
        out2 = F.relu(self.res_block172(out2+_upsample_like(out3,out2)) + out2)
        out = F.relu(self.res_block17(out + _upsample_like(out2,out) + _upsample_like(out3,out)) + out)


        out2_ = self.conv_out_128(out2 + out2_in)  #9*256*256
        out3_ = self.conv_out_64(out3 + out3_in)   #9*256*256

        out1_com = torch.cat((self.conv_out_256(out+out_in) , self.deconv_out_128(out2_) ,self.deconv_out_64(out3_)),1)

        out1 = self.conv_out(out1_com)

        return out1,out2_,out3_,out1,out2_,out3_


