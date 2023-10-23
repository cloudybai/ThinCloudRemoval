#!/usr/bin/python3
# coding:utf-8


from torch.utils.data import DataLoader
from lib.dataset.dataset import Landsat, Landsat_multi_scale, Landsat_RGB
import torchvision.transforms as transforms


def Get_dataloader(args):
    # kind can choose single or multi
    # trainset = '/media/omnisky/data3/xiesong/datalist/cloud/RSC_data/train.txt'
    # testset = '/media/omnisky/data3/xiesong/datalist/cloud/RSC_data/test.txt'
    # teansforms_ = [
    #     transforms.FiveCrop((224, 224)),
    #     Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ]
    if args.kind == 'single':
        train = Landsat(args.trainset)
        test = Landsat(args.testset)

    if args.kind == 'multi':
        train = Landsat_multi_scale(args.trainset)
        test = Landsat_multi_scale(args.testset)

    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_dataloader = DataLoader(test, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    return train_dataloader, test_dataloader


def Get_dataloader_RGB(args):
    transforms_ = [
        # transforms.Resize(int(args.img_height*1.12), Image.BICUBIC),
        # transforms.RandomCrop((args.img_height, args.img_width)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    imagedataset1 = Landsat_RGB(args, args.train_img_dir, args.train_gt_dir, transforms_=transforms_, unaligned=True)
    train_dataloader = DataLoader(imagedataset1, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                  drop_last=True)
    imagedataset2 = Landsat_RGB(args, args.test_img_dir, args.test_gt_dir, transforms_=transforms_, unaligned=True)
    test_dataloader = DataLoader(imagedataset2, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    return train_dataloader,test_dataloader


#
from lib.utils.config import Config
from lib.utils.parse_args import parse_args

if __name__ == '__main__':
    config = Config().parse()
    args = parse_args(config)
    train, valid = Get_dataloader(args)
    for i, (img, gt, name) in enumerate(train):
        print(len(train))


