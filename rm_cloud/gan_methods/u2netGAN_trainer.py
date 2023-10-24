#!/usr/bin/python3
#coding:utf-8


import os
import numpy as np

import time
from lib.utils.convert_RGB import convert

import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.loss.mse_loss import get_mse_loss_function
from lib.loss.L1_loss import get_l1_loss_function
from lib.utils.utils import ReplayBuffer, LambdaLR
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from lib.dataset.landsat_8_dataloader_DA import Get_dataloader
from tqdm import tqdm

class u2netGANTrainer(object):
    def __init__(self,gen,dis,optimizer_G,optimizer_D):
        self.gen = gen.train()
        self.dis = dis.train()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

    def train(self,args,train_dataloader,test_dataloader,start_epoch,end_epoch):
        begin_time = time.time()
        trainloader, testloader = Get_dataloader(enhance=False)
        patch = (1,args.img_height//(2**args.n_D_layers*4), args.img_width//(2**args.n_D_layers*4))
        fake_img_buffer = ReplayBuffer()
        fake_gt_buffer = ReplayBuffer()

        writer = SummaryWriter(log_dir='{}'.format(args.log_dir),comment='train_loss')
        print("======== begin train model ========")

        best_loss = 3
        os.makedirs(args.results_gen_model_dir, exist_ok=True)
        os.makedirs(args.results_dis_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        os.makedirs(args.results_gt_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        for epoch in range(start_epoch,end_epoch):
            self.gen.train()
            self.dis.train()
            data_loader = tqdm(trainloader)
            data_loader.set_description('[{} {}/{}'.format('train',epoch,args.end_epoch))
            for i, (img,gt,name) in enumerate(data_loader):
                if torch.cuda.is_available():
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                else:
                    img = Variable(img)
                    gt = Variable(gt)
                valid = Variable(torch.FloatTensor(np.ones((img.size(0), *patch))).cuda(), requires_grad=False)
                fake = Variable(torch.FloatTensor(np.zeros((img.size(0), *patch))).cuda(), requires_grad=False)

                ##### Train Generator #######
                self.optimizer_G.zero_grad()
                # identity loss
                L1_loss = get_l1_loss_function()
                d0, d1, d2, d3, d4, d5, d6 = self.gen(img)
                loss_l1 = L1_loss(d0,gt)

                mse_loss = get_mse_loss_function()
                loss0 = mse_loss(d0, gt)
                loss1 = mse_loss(d1, gt)
                loss2 = mse_loss(d2, gt)
                loss3 = mse_loss(d3, gt)
                loss4 = mse_loss(d4, gt)
                loss5 = mse_loss(d5, gt)
                loss6 = mse_loss(d6, gt)
                loss_mse = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                # GAN loss
                fake_gt = self.gen(img)[0]
                pred_fake = self.dis(fake_gt)
                gan_loss = get_mse_loss_function()
                loss_GAN = gan_loss(pred_fake,valid)

                # Tota loss
                loss_G = loss_GAN + args.lambda_id * loss_l1 + loss_mse

                loss_G.backward()
                nn.utils.clip_grad_norm_(self.gen.parameters(), 2.0)

                self.optimizer_G.step()
                batches_done = epoch * len(data_loader) + i

                ####### Train Discriminator  #######

                self.optimizer_D.zero_grad()
                pred_real = self.dis(img)
                loss_real = gan_loss(pred_real,valid)
                fake_gt = fake_gt_buffer.push_and_pop(fake_gt)
                pred_fake = self.dis(fake_gt.detach())
                loss_fake = gan_loss(pred_fake,fake)

                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                nn.utils.clip_grad_norm_(self.dis.parameters(), 2.0)

                self.optimizer_D.step()

                writer.add_scalars('{}_train_loss'.format(args.model.arch),{'loss_G':loss_G.data.cpu(),'loss_D':loss_D.data.cpu()}, batches_done)

                if i % args.interval == 0:
                    print('[epoch %d/%d] [batch %d/%d] [loss: %f]' %(epoch,end_epoch,batches_done,(end_epoch*len(data_loader)),loss_G.item()))

            if epoch % args.interval == 0 or epoch > args.end_epoch -5:
                torch.save(self.gen.state_dict(),
                           args.results_gen_model_dir + '/%d-%d_gen.pkl' % (epoch, batches_done))
                torch.save(self.dis.state_dict(),
                           args.results_dis_model_dir + '/%d-%d_dis.pkl' % (epoch, batches_done))

                fake_gt_RGB = convert(fake_gt[0])
                fake_gt_RGB.save(args.results_gt_dir + '%d-%d.png' % (epoch, batches_done))


        #test
            if epoch % 100 ==0 or epoch > args.end_epoch-5:
                self.gen.eval()
                self.dis.eval()
                total_test_loss = 0
                data_loader = tqdm(testloader)
                data_loader.set_description('[{} {}/{}'.format('test', epoch, args.end_epoch))
                for i, (img, gt, name) in enumerate(data_loader):
                    if torch.cuda.is_available():
                        img = Variable(img.cuda())
                        gt = Variable(gt.cuda())
                    else:
                        img = Variable(img)
                        gt = Variable(gt)
                    valid = Variable(torch.FloatTensor(np.ones((img.size(0), *patch))).cuda(), requires_grad=False)
                    fake = Variable(torch.FloatTensor(np.zeros((img.size(0), *patch))).cuda(), requires_grad=False)


                    # identity loss
                    L1_loss = get_l1_loss_function()
                    loss_l1 = L1_loss(self.gen(img)[0], gt)

                    # GAN loss
                    fake_gt = self.gen(img)[0]
                    pred_fake = self.dis(fake_gt)
                    gan_loss = get_mse_loss_function()
                    loss_GAN = gan_loss(pred_fake, valid)

                    # Tota loss
                    loss_G = loss_GAN + args.lambda_id * loss_l1
                    total_test_loss += loss_G.data.cpu()
                avg_test_loss = total_test_loss/len(testloader)

                if avg_test_loss < best_loss :
                    best_loss = avg_test_loss
                    torch.save(self.gen.state_dict(),args.results_gen_model_dir + '/best_model_G.pkl')
                    torch.save(self.dis.state_dict(),args.results_dis_model_dir + 'best_model_D.pkl')

                    fake_gt_RGB = convert(fake_gt[0])
                    fake_gt_RGB.save(args.results_gt_dir + '%d-%d.png' % (epoch, batches_done))

        writer.close()
        print("total_consum_time = %.2f s" % (time.time()-begin_time))




