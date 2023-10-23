import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from einops import rearrange

#######focus linear attention#######
########CLA FLA HLA WLA#############
########matrix order corrected#####

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv3d(in_channels, out_channels, stride = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size = (3, 3, 3),
        stride = stride, padding=1, bias=False)

#unet_origin_conv
class REBNCONV(nn.Module):
    #dirate是空洞卷积
    def __init__(self, in_ch=1, out_ch=1, dirate=1): #in_ch=16?
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv3d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        #这里的3是3x3x3
        # self.bn_s1 = nn.BatchNorm3d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        # xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        xout = self.relu_s1(self.conv_s1(hx))
        return xout

class REBNDECONV(nn.Module):
    def __init__(self, in_ch=16, out_ch=16):
        super(REBNDECONV, self).__init__()

        self.deconv_s1 = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.bn_s1 = nn.BatchNorm3d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        # xout = self.relu_s1(self.bn_s1(self.deconv_s1(hx)))
        xout = self.relu_s1(self.deconv_s1(hx))
        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src

# U 型
class UNet(nn.Module):

    def __init__(self, in_ch=1, mid_ch=16, out_ch=1):
        super(UNet, self).__init__()

        # self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconvin = REBNCONV(in_ch, mid_ch, dirate=1)

        # self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv1_1 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv1_2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch*2, dirate=1)
        # self.pool2 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch*2, mid_ch*2, dirate=1)
        self.pool3 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch*2, mid_ch*4, dirate=1)
        # self.pool4 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch*4, mid_ch*4, dirate=1)
        self.pool5 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch*4, mid_ch*8, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch*8, mid_ch*8, dirate=2)

        # self.channel_attention = ChannelAttention()
        # self.spectral_attention = SpectralAttention()
        # self.spatial_attention = SpatialAttention()


        # self.transformer_channel_attention = TransformerChannelAttention(dim=128, heads=2)
        # self.transformer_spectral_attention = TransformerSpectralAttention(dim=9, heads=3)
        # self.transformer_spatial_attention = WindowAttention(dim=1152, num_heads=3)

        self.focusedlinear_channel_attention = FocusedLinearChannelAttention(dim=128, heads=2, flat_token=9*32*32)
        self.focusedlinear_spectral_attention = FocusedLinearSpectralAttention(dim=9, heads=3, flat_token=128*32*32)
        # self.focusedlinear_height_attention = FocusedLinearHeightAttention(dim=32, heads=4, flat_token=128*9*32)
        # self.focusedlinear_width_attention = FocusedLinearWidthAttention(dim=32, heads=4, flat_token=128*9*32)
        # self.focusedlinear_row_attention = FocusedLinearRowAttention(dim=1152, heads=3)
        # self.focusedlinear_col_attention = FocusedLinearColAttention(dim=1152, heads=3)


        self.rebnconv6d = REBNCONV(mid_ch*2*8, mid_ch*8, dirate=1)
        self.rebnconv6dup = REBNDECONV(128, 64)
        #这个16是什么？

        self.rebnconv5d = REBNCONV(mid_ch*2*4, mid_ch*4, dirate=1)
        # self.rebnconv5dup = REBNDECONV(16, 16)

        self.rebnconv4d = REBNCONV(mid_ch*2*4, mid_ch*4, dirate=1)
        self.rebnconv4dup = REBNDECONV(64, 32)

        self.rebnconv3d = REBNCONV(mid_ch*2*2, mid_ch*2, dirate=1)
        # self.rebnconv3dup = REBNDECONV(16, 16)

        self.rebnconv2d = REBNCONV(mid_ch*2*2, mid_ch*2, dirate=1)
        self.rebnconv2dup = REBNDECONV(32, 16)

        # self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.conv_out = nn.Sequential(
            conv3d(mid_ch*2, 1),
            # 这里应该也要改
            # add new
            # nn.BatchNorm3d(1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        hx = x.unsqueeze(1)
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1_1(hxin)
        hx1 = self.rebnconv1_2(hx1)

        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        # hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx2)

        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)

        # hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx4)

        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        # hx7 = hx7 * self.channel_attention(hx7)
        # hx7 = hx7 * self.spectral_attention(hx7)
        # hx7 = hx7 * self.spatial_attention(hx7)

        # hx7 = self.transformer_channel_attention(hx7)   # [2 16 9 32 32]
        # hx7 = self.transformer_spectral_attention(hx7)  # [2 16 9 32 32]

        hx7 = hx7 * self.focusedlinear_channel_attention(hx7)
        hx7 = hx7 * self.focusedlinear_spectral_attention(hx7)
        # hx7 = hx7 * self.focusedlinear_height_attention(hx7)
        # hx7 = hx7 * self.focusedlinear_width_attention(hx7)




        # hx7 = hx7_3.reshape(B, C//9, 9, H, W)
        # hx7 = hx7 * shifted_x

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))

        hx6dup = self.rebnconv6dup(hx6d)
        hx5d =  self.rebnconv5d(torch.cat((hx6dup, hx5), 1))

        # hx5dup = self.rebnconv5dup(hx5d)
        hx4d = self.rebnconv4d(torch.cat((hx5d, hx4), 1))

        hx4dup = self.rebnconv4dup(hx4d)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))

        # hx3dup = self.rebnconv3dup(hx3d)
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))

        hx2dup = self.rebnconv2dup(hx2d)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return self.conv_out(torch.cat([hx1d, hxin], dim=1)).squeeze(1)+x # "+x" indicates the residual learning


class TransformerChannelAttention(nn.Module): #define what is channelattention
    def __init__(
            self,
            dim=16,
            heads=2,
    ):
        super().__init__()
        # self.gp1 = nn.AdaptiveMaxPool3d((None, None, 1))
        self.num_heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        # """
        # x_in: [b,c,s,h,w]
        # x_in: [b,h,w,c]_
        # return out: [b,h,w,c]
        # """
        # x_in = x_in.permute([0, 3, 4, 1, 2])     # b h w c s [2 32 32 128 9]
        # x_in = self.gp1(x_in)                    # [2 32 32 128 1]
        # x_in = x_in.squeeze(dim=4)               # [2 32 32 128]
        #
        # b, h, w, c = x_in.shape
        # x = x_in.reshape(b, h*w, c)
        # q_inp = self.to_q(x)
        # k_inp = self.to_k(x)
        # v_inp = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
        #                         (q_inp, k_inp, v_inp))
        # # q: b,heads,hw,c
        # q = q.transpose(-2, -1)
        # k = k.transpose(-2, -1)
        # v = v.transpose(-2, -1)
        # q = F.normalize(q, dim=-1, p=2)
        # k = F.normalize(k, dim=-1, p=2)
        # attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        # attn = attn * self.rescale
        # attn = attn.softmax(dim=-1)
        # x = attn @ v   # b,heads,d,hw
        # x = x.permute(0, 3, 1, 2)    # Transpose
        # x = x.reshape(b, h * w, self.dim)
        # out_c = self.proj(x).view(b, h, w, c)
        # out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # out = out_c + out_p
        # out = out.permute(0, 3, 1, 2)

        """
        x_in: [b,c,s,h,w]
        x_in: [b,h,w,c]_
        return out: [b,h,w,c]
        """
        x_in = x_in.permute([0, 3, 4, 2, 1])     # b h w c s [2 32 32 9 128]


        b, h, w, s, c = x_in.shape
        x = x_in.reshape(b, h*w*s, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w * s, self.dim)
        out_c = self.proj(x).view(b, h, w, s, c)
        out_c = out_c.permute(0, 1, 2, 4, 3)   # [b, h, w, c, s]
        out_c = out_c.permute(0, 3, 4, 1, 2)   # [b, c, s, h, w]
        return out_c

class FocusedLinearChannelAttention(nn.Module):
    def __init__(
            self,
            dim=16,
            heads=2,
            flat_token=1,  # should be S*H*W
            focusing_factor=3,
            kernel_size=5
    ):
        super().__init__()
        # self.gp1 = nn.AdaptiveMaxPool3d((None, None, 1))
        self.num_heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

        head_dim = dim // heads
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv3d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, flat_token, dim)))
        # print('Linear Attention sr_ratio{} f{} kernel{}'.
        #       format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x_in):
        """
        x_in: [b,c,s,h,w]
        x_in: [b,h,w,c]_
        return out: [b,h,w,c]
        """
        # x_in = x_in.permute([0, 3, 4, 1, 2])     #b h w c s
        x_in = x_in.permute([0, 2, 3, 4, 1])  # b s h w c
        # x_in = self.gp1(x_in)
        # x_in = x_in.squeeze(dim=4)

        B, S, H, W, C = x_in.shape
        x = x_in.reshape(B, S*H*W, C)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
        #                         (q_inp, k_inp, v_inp))
        q,k,v=q_inp,k_inp,v_inp


        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int((v.shape[1]/S) ** 0.5)
        feature_map = rearrange(v, "b (s w h) c -> b c s w h", s=S, w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c s w h -> b (s w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c) ", h=self.num_heads)


        x = self.proj(x)
        x = x.reshape(B, S, H, W, C)
        x = x.permute(0, 4, 1, 2, 3) #b c s h w



        return x

class TransformerSpectralAttention(nn.Module):
    def __init__(
            self,
            dim=9,
            heads=3,
    ):
        super().__init__()
        self.num_heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,c,s,h,w]
        x_in: [b,h,w,c]_
        return out: [b,h,w,c]
        """
        # x_in = x_in.permute([0, 3, 4, 2, 1])  # b h w s c
        # x_in = self.gp2(x_in)
        # x_in = x_in.squeeze(dim=4)
        #
        # b, h, w, c = x_in.shape
        # x = x_in.reshape(b, h * w, c)
        # q_inp = self.to_q(x)
        # k_inp = self.to_k(x)
        # v_inp = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
        #               (q_inp, k_inp, v_inp))
        # # q: b,heads,hw,c
        # q = q.transpose(-2, -1)
        # k = k.transpose(-2, -1)
        # v = v.transpose(-2, -1)
        # q = F.normalize(q, dim=-1, p=2)
        # k = F.normalize(k, dim=-1, p=2)
        # attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        # attn = attn * self.rescale
        # attn = attn.softmax(dim=-1)
        # x = attn @ v  # b,heads,d,hw
        # x = x.permute(0, 3, 1, 2)  # Transpose
        # x = x.reshape(b, h * w, self.dim)
        # out_c = self.proj(x).view(b, h, w, c)
        # out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # out = out_c + out_p
        # out = out.permute(0, 3, 1, 2)

        x_in = x_in.permute([0, 3, 4, 1, 2])  # b h w c s


        b, h, w, c, s = x_in.shape
        x = x_in.reshape(b, h * w * c, s)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w * c, self.dim)
        out_c = self.proj(x).view(b, h, w, c, s)
        out_c = out_c.permute(0, 3, 4, 1, 2)


        return out_c

class FocusedLinearSpectralAttention(nn.Module):
    def __init__(
            self,
            dim=9,
            heads=3,
            flat_token=1,  # should be C*H*W
            focusing_factor=3,
            kernel_size=5
    ):
        super().__init__()
        self.gp1 = nn.AdaptiveMaxPool3d((None, None, 1))
        self.num_heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

        head_dim = dim // heads
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv3d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, flat_token, dim)))
        # print('Linear Attention sr_ratio{} f{} kernel{}'.
        #       format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x_in):
        """
        x_in: [b,c,s,h,w]
        x_in: [b,h,w,c]_
        return out: [b,h,w,c]
        """
        x_in = x_in.permute([0, 1, 3, 4, 2])     #b c h w s
        # x_in = self.gp1(x_in)
        # x_in = x_in.squeeze(dim=4)

        B, C, H, W, S = x_in.shape
        x = x_in.reshape(B, C*H*W, S)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
        #                         (q_inp, k_inp, v_inp))
        q,k,v=q_inp,k_inp,v_inp

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)


        num = int((v.shape[1]/C) ** 0.5)
        feature_map = rearrange(v, "b (c w h) s -> b s c w h", c=C,w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b s c w h -> b (c w h) s")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c) ", h=self.num_heads)


        x = self.proj(x)
        x = x.reshape(B, C, H, W, S)
        x = x.permute(0, 1, 4, 2, 3)  # b c s h w


        return x

class FocusedLinearHeightAttention(nn.Module):
    def __init__(
            self,
            dim=32,
            heads=4,
            flat_token=1,  # should be C*S*W
            focusing_factor=3,
            kernel_size=5
    ):
        super().__init__()
        # self.gp1 = nn.AdaptiveMaxPool3d((None, None, 1))
        self.num_heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

        head_dim = dim // heads
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv3d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, flat_token, dim)))
        # print('Linear Attention sr_ratio{} f{} kernel{}'.
        #       format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x_in):
        """
        x_in: [b,c,s,h,w]
        return out: [b,c,s,h,w]
        """
        x_in = x_in.permute([0, 1, 2, 4, 3])     #b c s w h
        # x_in = self.gp1(x_in)
        # x_in = x_in.squeeze(dim=4)

        B, C, S, W, H = x_in.shape
        x = x_in.reshape(B, C*S*W, H)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
        #                         (q_inp, k_inp, v_inp))
        q,k,v=q_inp,k_inp,v_inp

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)


        feature_map = rearrange(v, "b (c s w) h -> b h c s w", c=C,w=W)
        feature_map = rearrange(self.dwc(feature_map), "b h c s w -> b (c s w) h")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c) ", h=self.num_heads)

        x = self.proj(x)
        x = x.reshape(B, C, S, W, H)
        x = x.permute(0, 1, 2, 4, 3)  # b c s h w


        return x

class FocusedLinearWidthAttention(nn.Module):
    def __init__(
            self,
            dim=32,
            heads=4,
            flat_token=1, #should be C*S*H
            focusing_factor=3,
            kernel_size=5
    ):
        super().__init__()
        # self.gp1 = nn.AdaptiveMaxPool3d((None, None, 1))
        self.num_heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

        head_dim = dim // heads
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv3d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, flat_token, dim)))  # ？？
        # print('Linear Attention sr_ratio{} f{} kernel{}'.
        #       format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x_in):
        """
        x_in: [b,c,s,h,w]
        return out: [b,c,s,h,w]
        """
        # x_in = x_in.permute([0, 1, 2, 4, 3])
        # x_in = self.gp1(x_in)
        # x_in = x_in.squeeze(dim=4)

        B, C, S, H, W = x_in.shape
        x = x_in.reshape(B, C*S*H, W)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
        #                         (q_inp, k_inp, v_inp))
        q,k,v=q_inp,k_inp,v_inp

        k = k + self.positional_encoding  # why k? what does this nn.Parameter(torch.zeros(size=(1, 1, dim))) means
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)


        feature_map = rearrange(v, "b (c s h) w -> b w c s h", c=C,h=H)
        feature_map = rearrange(self.dwc(feature_map), "b w c s h -> b (c s h) w")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c) ", h=self.num_heads)

        x = self.proj(x)
        x = x.reshape(B, C, S, H, W)
        # x = x.permute(0, 1, 2, 4, 3)  # b c s h w


        return x


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, window_size=(8, 8), qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape  # 2 16 9 32 32

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



if __name__=="__main__":
    net = UNet().cuda()
    x = torch.rand([2, 9, 256, 256]).cuda()
    x_hat = net(x)
    print(x_hat.shape)