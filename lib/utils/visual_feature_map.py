import numpy as np
import cv2
import os
from lib.utils.config import Config
from lib.utils.parse_args import parse_args
from lib.dataset.dataloader import Get_dataloader


def heatmap(img):
    if len(img.shape) == 3:
        b, h, w = img.shape
        heat = np.zeros((b, 3, h, w)).astype('uint8')
        for i in range(b):
            heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, :, :], cv2.COLORMAP_JET), (2, 0, 1))
    else:
        b, c, h, w = img.shape
        heat = np.zeros((b, 3, h, w)).astype('uint8')
        for i in range(b):
            heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, 0, :, :], cv2.COLORMAP_JET), (2, 0, 1))
    return heat


def save_mask(name, img_lists, m=0):
    data, pred, label, mask, mask_label = img_lists
    pred = pred.cpu().data

    mask = mask.cpu().data
    mask_label = mask_label.cpu().data
    data, label, pred, mask, mask_label = data * 255, label * 255, pred * 255, mask * 255, mask_label * 255
    pred = np.clip(pred, 0, 255)

    mask = np.clip(mask.numpy(), 0, 255).astype('uint8')
    mask_label = np.clip(mask_label.numpy(), 0, 255).astype('uint8')
    h, w = pred.shape[-2:]
    mask = heatmap(mask)
    mask_label = heatmap(mask_label)
    gen_num = (1, 1)

    img = np.zeros((gen_num[0] * h, gen_num[1] * 5 * w, 3))
    for img_list in img_lists:
        for i in range(gen_num[0]):
            row = i * h
            for j in range(gen_num[1]):
                idx = i * gen_num[1] + j
                tmp_list = [data[idx], pred[idx], label[idx], mask[idx], mask_label[idx]]
                for k in range(5):
                    col = (j * 5 + k) * w
                    tmp = np.transpose(tmp_list[k], (1, 2, 0))
                    img[row: row + h, col: col + w] = tmp

    img_file = os.path.join('./feature_map', '%s.png' % (name))
    cv2.imwrite(img_file, img)

from lib.model.generator.SPANet import SPANet
from lib.model.create_generator import create_generator
import torch
from torch.autograd import Variable

if __name__ == '__main__':
    config = Config().parse()
    args = parse_args(config)
    train_loader, test_loader = Get_dataloader(args)

    spanet = SPANet()
    model = create_generator(args,spanet)
    model.load_state_dict(torch.load('/home/xqzhou/xscode/remove_cloud/experiments/SPANet/save_models_Res/best_model.pkl'))

    for i, (img, gt, M, name) in enumerate(test_loader):
        img = Variable(img.cuda())
        gt = Variable(gt.cuda())
        M = Variable(M.cuda())
        mask, out = model(img)

        mask = mask.cpu().data
        mask = mask*255
        mask = np.clip(mask.numpy(), 0, 255).astype('uint8')
        mask = heatmap(mask)
        mask = np.transpose(mask[0], (1, 2, 0))
        cv2.imwrite('/home/xqzhou/xscode/remove_cloud/lib/utils/feature/{}.jpg'.format(name),mask)
        print('we')






