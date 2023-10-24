#!/usr/bin/python3
#coding:utf-8



from torch.utils.data  import DataLoader
from lib.dataset.ladsat_8_dataset_DA import Landsat
import torchvision.transforms as transforms
import random
import numpy as np
from theconf import Config as C

# operate_list = ['Rotate', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'AutoContrast', 'Invert', 'Equalize','Solarize', 'Posterize',
#                 'Contrast', 'Color', 'Brightness', 'Sharpness']


operate_list = ['Rotate', 'Mirror', 'Flip']
# operate_list = ['Addcloud']
operate_set = []
for operate in operate_list:
    for m in np.arange(0,1.1,0.1):
        operate_set.append([operate,1,m])

from PIL import Image
class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies
        self.num = 0

    def __call__(self, img):
        ret_img = []
        ret_img2 = []
        for _ in range(1):
            policy = random.choice(self.policies)
            for j in range(9):
                name = policy[0]
                level = policy[2]
                # img_ = apply_augment(Image.fromarray(np.uint8(img[:, :, j]),'L'), name, level)
                img_ = apply_augment(img[:, :, j], name, level)
                ret_img.append(img_)
                # print(policy)

        return ret_img
# class Augmentation(object):
#     def __init__(self, policies):
#         self.policies = policies
#         self.num = 0
#
#     def __call__(self, img):
#         ret_img = []
#         ret_img2 = []
#         for _ in range(1):
#             policy = random.choice(self.policies)
#             for j in range(18):
#                 name = policy[0]
#                 level = policy[2]
#                 img_ = apply_augment(Image.fromarray(np.uint8(img[:, :, j]),'L'), name, level)
#                 ret_img.append(img_)
#                 # print(policy)
#             policy = random.choice(self.policies)
#             for j in range(18):
#                 name = policy[0]
#                 level = policy[2]
#                 img_1 = apply_augment(ret_img[j], name, level)
#                 ret_img2.append(img_1)
#                 # print(policy)
#         return ret_img2

def Get_dataloader(enhance):
    if enhance:
        transform_train = transforms.Compose([
        ])
        transform_train.transforms.insert(0, Augmentation(operate_set))
    else:
        transform_train = None

    train1 = Landsat('/media/userdisk2/xqzhou/cloud/datalist/cloud/train.txt', transform_train)
    # train2 = Landsat('/media/userdisk2/xqzhou/cloud/datalist/cloud/train.txt', None)
    # train = torch.utils.data.ConcatDataset([train1,train2])

    test = Landsat('/media/userdisk2/xqzhou/cloud/datalist/cloud/test.txt', None)
    train_dataloader = DataLoader(train1, batch_size=4, shuffle=True, num_workers=1, drop_last=True)
    test_dataloader = DataLoader(test, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
    return train_dataloader, test_dataloader

import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from imgaug import augmenters as iaa

def Addcloud(img,_):
    # aug = iaa.Clouds(seed=np.random.RandomState([1,2,3]))
    aug = iaa.CloudLayer(intensity_mean=(196, 255),
                intensity_freq_exponent=(-2.5, -2.0),
                intensity_coarse_scale=10,
                alpha_min=0,
                alpha_multiplier=(0.25, 0.75),
                alpha_size_px_max=(2, 8),
                alpha_freq_exponent=(-2.5, -2.0),
                sparsity=(0.8, 1.0),
                density_multiplier=(0.5, 1.0),seed=1)
    img_=aug.augment_image(img)
    return Image.fromarray(img_)

def Flip(img, _):
    print('-----')
    return PIL.ImageOps.flip(img)

def Mirror(img, _):
    return PIL.ImageOps.mirror(img)


def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (Flip, -30, 30),
        (Mirror, -30, 30),
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Addcloud, 0.1, 1.9),

        # (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            # (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l

augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

def get_augment(name):
    return augment_dict[name]

def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

from lib.utils.config import Config
from lib.utils.parse_args import parse_args

# if __name__ =='__main__':
#     config = Config().parse()
#     args = parse_args(config)
#     train,valid = Get_dataloader(True)
#     for i,(img,gt,name) in enumerate(train):
#         print(len(train))
