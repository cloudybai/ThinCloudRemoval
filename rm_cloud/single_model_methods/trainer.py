#!/usr/bin/python3
#coding:utf-8


import numpy as np
from PIL import Image

import os
import torch
import time
from torch.autograd import Variable
from lib.loss.mse_loss import get_mse_loss_function
from lib.loss.multi_mse import muti_mse_loss_fusion
from lib.utils.convert_RGB import convert
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from lib.utils.log_report import LogReport
from tqdm import tqdm

#no enhance
# from lib.dataset.landsat_8_dataloader_DA import Get_dataloader
# have DA
from lib.dataset.landsat_8_dataloader_DA_1 import Get_dataloader_enhance

from lib.dataset.dataloader import Get_dataloader
from lib.utils.lr_scheduler import adjust_lr
from torch.optim.lr_scheduler import MultiStepLR
class Trainer(object):
    """
        Each trainer should  inherit this base trainer

    """
    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer
        self.sche = MultiStepLR(self.optimizer, milestones=[1000,2000], gamma=0.1)


    def train_single(self,args,trainloader,testloader,start_epoch,end_epoch,):
        if args.enhance == True:
            trainloader,testloader = Get_dataloader_enhance(args)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.results_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        begin_time = time.time()
        print("======== begin train model ========")
        print('data_size:',len(trainloader))
        #xs change
        writers = [SummaryWriter(log_dir='./mylogs/%s/%s' % (args.name, x)) for x in ['train', 'test']]
        best_loss = float('inf')
        change_loss = False

        #train
        for epoch in range(start_epoch,end_epoch):
            self.model.train()
            data_loader = tqdm(trainloader)
            data_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.end_epoch))
            #xs change
            rs = dict()
            total_train_loss = 0
            # xs change
            steps = 0
            total_steps = len(trainloader)
            for i, (img,gt,M,name) in enumerate(data_loader):
                steps += 1
                if torch.cuda.is_available():
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                    M = Variable(M.cuda())
                else:
                    img = Variable(img)
                    gt = Variable(gt)

                # gen_img, loss = self.optimize_strategy(img,gt)
                is_optimizer = True

                if args.model.arch == 'unet':
                    gen_img, rs['train'] = self.optimize_strategy(img,gt,is_optimizer)
                else:
                    gen_img, rs['train'] = self.optimize_strategy_rsaunet(change_loss,img,gt,M,is_optimizer)

                total_train_loss += rs['train'].data.cpu()
                # scheduler.step(epoch + float(steps)/total_steps)
            # self.sche.step(epoch)
            avg_train_loss = total_train_loss / len(trainloader)
            writers[0].add_scalar('{}_train_loss'.format(args.model.arch), avg_train_loss.cpu(), epoch)

            #xs change
            writers[0].add_scalar('lr',self.optimizer.param_groups[0]['lr'],epoch)
            # print(self.optimizer.param_groups[0]['lr'])
            # test
            if epoch % args.evaluation_interval == 0:
                self.model.eval()
                is_optimizer = False
                total_test_loss = 0
                data_loader = tqdm(testloader)
                data_loader.set_description('[%s %04d/%04d]' % ('test', epoch, args.end_epoch))
                for i, (img, gt, M,name) in enumerate(data_loader):
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                    M = Variable(M.cuda())
                    if args.model.arch == 'unet':
                        gen_img, rs['test'] = self.optimize_strategy(img, gt, is_optimizer)
                    else:
                        gen_img, rs['test'] = self.optimize_strategy_rsaunet(change_loss, img, gt, M, is_optimizer)
                    gen_RGB = convert(args,gen_img[0])

                    total_test_loss += rs['test'].data.cpu()
                avg_test_loss = total_test_loss/len(testloader)
                batches_done = epoch*len(data_loader) + i
                gen_RGB.save(args.results_img_dir + '%d-%d—best.png' % (epoch, batches_done))

                if epoch % args.save_model == 0:
                    torch.save(self.model.state_dict(), args.results_model_dir +'/%d_model.pkl'%(epoch))
                # save best model and img
                if avg_test_loss < best_loss :
                    best_loss = avg_test_loss
                    # torch.save(self.model.state_dict(), args.results_model_dir +'/%d-%d_best_model.pkl'%(epoch,batches_done))
                    torch.save(self.model.state_dict(), args.results_model_dir +'/best_model.pkl')
                    gen_RGB.save(args.results_img_dir +'%d-%d—---best.png' %(epoch,batches_done))

                writers[1].add_scalar('{}_test_loss'.format(args.model.arch),avg_test_loss.cpu(), epoch)
                print('[epoch %d/%d] [batch %d/%d] [loss: %f]' % (
                epoch, end_epoch, batches_done, (end_epoch * len(data_loader)), avg_test_loss.item()))

            if epoch == args.stage2_epoch:
                change_loss = True
                trainloader,testloader = Get_dataloader(args)
                self.optimizer.param_groups[0]['lr'] = 0.00004

        print("total_consum_time = %.2f s" % (time.time()-begin_time))

    def train_u2net(self,args,data_loader,start_epoch,end_epoch,):
        trainloader, testloader = Get_dataloader(args)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.results_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        begin_time = time.time()
        print("======== begin train model ========")
        print('data_size:',len(data_loader))
        # writer = SummaryWriter(log_dir='{}'.format(args.log_dir),comment='train_loss')
        #xs change
        writers = [SummaryWriter(log_dir='./mylogs/%s/%s' % (args.name, x)) for x in ['train', 'test']]
        best_loss = float('inf')
        #train
        for epoch in range(start_epoch,end_epoch):
            self.model.train()
            data_loader = tqdm(trainloader)
            data_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.end_epoch))
            #xs change
            rs = dict()
            total_train_loss = 0
            # xs change
            steps = 0
            total_steps = len(trainloader)
            for i, (img,gt,M,name) in enumerate(data_loader):
                steps += 1
                if torch.cuda.is_available():
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                else:
                    img = Variable(img)
                    gt = Variable(gt)
                is_optimizer = True
                gen_img, rs['train'] = self.optimize_strategy(img, gt, is_optimizer)
                total_train_loss += rs['train'].data.cpu()
            avg_train_loss = total_train_loss / len(trainloader)
            writers[0].add_scalar('{}_train_loss'.format(args.model.arch), avg_train_loss.cpu(), epoch)

            #xs change
            writers[0].add_scalar('lr',self.optimizer.param_groups[0]['lr'],epoch)
            # test
            if epoch % args.evaluation_interval == 0:
                self.model.eval()
                is_optimizer = False
                total_test_loss = 0
                data_loader = tqdm(testloader)
                data_loader.set_description('[%s %04d/%04d]' % ('test', epoch, args.end_epoch))
                for i, (img, gt,M, name) in enumerate(data_loader):
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                    gen_img, rs['test'] = self.optimize_strategy(img, gt, is_optimizer)
                    gen_RGB = convert(args,gen_img[0])
                    total_test_loss += rs['test'].data.cpu()
                avg_test_loss = total_test_loss/len(testloader)
                batches_done = epoch*len(data_loader) + i
                gen_RGB.save(args.results_img_dir + '%d-%d.png' % (epoch, batches_done))

                batches_done = epoch*len(data_loader) + i

                # save best model and img
                if epoch % 1000 == 0:
                    torch.save(self.model.state_dict(), args.results_model_dir +'/%d_model.pkl'%(epoch))
                if avg_test_loss < best_loss :
                    best_loss = avg_test_loss
                    # torch.save(self.model.state_dict(), args.results_model_dir +'/%d-%d_best_model.pkl'%(epoch,batches_done))
                    torch.save(self.model.state_dict(), args.results_model_dir +'/best_model.pkl')
                    gen_RGB.save(args.results_img_dir +'%d----%d.png' %(epoch,batches_done))

                writers[1].add_scalar('{}_test_loss'.format(args.model.arch),avg_test_loss.cpu(), epoch)
                print('[epoch %d/%d] [batch %d/%d] [loss: %f]' % (
                epoch, end_epoch, batches_done, (end_epoch * len(data_loader)), avg_test_loss.item()))

        print("total_consum_time = %.2f s" % (time.time()-begin_time))



    def train_single_scale(self,args,trainloader,testloader,start_epoch,end_epoch,):

        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.results_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        begin_time = time.time()
        print("======== begin train model ========")
        print('data_size:',len(trainloader))
        #xs change
        writers = [SummaryWriter(log_dir='./mylogs/%s/%s' % (args.name, x)) for x in ['train', 'test']]
        best_loss = float('inf')
        change_loss = False

        #train
        for epoch in range(start_epoch,end_epoch):
            self.model.train()
            data_loader = tqdm(trainloader)
            data_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.end_epoch))
            #xs change
            rs = dict()
            total_train_loss = 0
            # xs change
            steps = 0
            total_steps = len(trainloader)
            for i, (img,gt,M,name) in enumerate(data_loader):
                steps += 1
                if torch.cuda.is_available():
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                    M = Variable(M.cuda())
                else:
                    img = Variable(img)
                    gt = Variable(gt)

                # gen_img, loss = self.optimize_strategy(img,gt)
                is_optimizer = True


                gen_img, rs['train'] = self.optimize_strategy_rsaunet(change_loss,img,gt,M,is_optimizer)

                total_train_loss += rs['train'].data.cpu()
                # scheduler.step(epoch + float(steps)/total_steps)
            # self.sche.step(epoch)
            avg_train_loss = total_train_loss / len(trainloader)
            writers[0].add_scalar('{}_train_loss'.format(args.model.arch), avg_train_loss.cpu(), epoch)

            #xs change
            writers[0].add_scalar('lr',self.optimizer.param_groups[0]['lr'],epoch)
            # print(self.optimizer.param_groups[0]['lr'])
            # test
            if epoch % args.evaluation_interval == 0:
                self.model.eval()
                is_optimizer = False
                total_test_loss = 0
                data_loader = tqdm(testloader)
                data_loader.set_description('[%s %04d/%04d]' % ('test', epoch, args.end_epoch))
                for i, (img, gt, M,name) in enumerate(data_loader):
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                    M = Variable(M.cuda())

                    gen_img, rs['test'] = self.optimize_strategy_rsaunet(change_loss, img, gt, M, is_optimizer)
                    gen_RGB = convert(args,gen_img[0])

                    total_test_loss += rs['test'].data.cpu()
                avg_test_loss = total_test_loss/len(testloader)
                batches_done = epoch*len(data_loader) + i
                gen_RGB.save(args.results_img_dir + '%d-%d—best.png' % (epoch, batches_done))


                # save best model and img
                if avg_test_loss < best_loss :
                    best_loss = avg_test_loss
                    # torch.save(self.model.state_dict(), args.results_model_dir +'/%d-%d_best_model.pkl'%(epoch,batches_done))
                    torch.save(self.model.state_dict(), args.results_model_dir +'/best_model.pkl')
                    gen_RGB.save(args.results_img_dir +'%d-%d—---best.png' %(epoch,batches_done))

                writers[1].add_scalar('{}_test_loss'.format(args.model.arch),avg_test_loss.cpu(), epoch)
                print('[epoch %d/%d] [batch %d/%d] [loss: %f]' % (
                epoch, end_epoch, batches_done, (end_epoch * len(data_loader)), avg_test_loss.item()))
            if epoch == 250:
                self.optimizer.param_groups[0]['lr'] = 0.00004
            if epoch == 400:
                self.optimizer.param_groups[0]['lr'] = 0.000004

        print("total_consum_time = %.2f s" % (time.time()-begin_time))


    def train_multi_scale(self,args,trainloader,testloader,start_epoch,end_epoch,):
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.results_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        begin_time = time.time()
        print("======== begin train model ========")
        print('data_size:',len(trainloader))
        #xs change
        writers = [SummaryWriter(log_dir='./mylogs/%s/%s' % (args.name, x)) for x in ['train', 'test']]
        best_loss = float('inf')

        #train
        for epoch in range(start_epoch,end_epoch):
            self.model.train()
            data_loader = tqdm(trainloader)
            data_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.end_epoch))
            #xs change
            rs = dict()
            total_train_loss = 0
            # xs change
            steps = 0
            for i, (img0,gt0,M0,img1,gt1,M1,img2,gt2,M2,name) in enumerate(data_loader):
                steps += 1
                if torch.cuda.is_available():
                    img0 = Variable(img0.cuda())
                    gt0 = Variable(gt0.cuda())
                    M0 = Variable(M0.cuda())
                    img1 = Variable(img1.cuda())
                    gt1 = Variable(gt1.cuda())
                    M1 = Variable(M1.cuda())
                    img2 = Variable(img2.cuda())
                    gt2 = Variable(gt2.cuda())
                    M2 = Variable(M2.cuda())
                else:
                    img0 = Variable(img0)
                    gt0 = Variable(gt0)

                # gen_img, loss = self.optimize_strategy(img,gt)
                is_optimizer = True
                gen_img, rs['train'] = self.optimize_multi_strategy(args,img0,gt0,M0,img1,gt1,M1,img2,gt2,M2,is_optimizer)
                total_train_loss += rs['train'].data.cpu()

            avg_train_loss = total_train_loss / len(trainloader)
            writers[0].add_scalar('{}_train_loss'.format(args.model.arch), avg_train_loss.cpu(), epoch)

            #xs change
            writers[0].add_scalar('lr',self.optimizer.param_groups[0]['lr'],epoch)
            # print(self.optimizer.param_groups[0]['lr'])
            # test
            if epoch % 5 == 0:
                self.model.eval()
                is_optimizer = False
                total_test_loss = 0
                data_loader = tqdm(testloader)
                data_loader.set_description('[%s %04d/%04d]' % ('test', epoch, args.end_epoch))
                # for i, (img0, gt0, M0, img1, gt1, M1, name) in enumerate(data_loader):
                for i, (img0,gt0,M0,img1,gt1,M1,img2,gt2,M2,name) in enumerate(data_loader):
                    img0 = Variable(img0.cuda())
                    gt0 = Variable(gt0.cuda())
                    M0 = Variable(M0.cuda())
                    img1 = Variable(img1.cuda())
                    gt1 = Variable(gt1.cuda())
                    M1 = Variable(M1.cuda())
                    img2 = Variable(img2.cuda())
                    gt2 = Variable(gt2.cuda())
                    M2 = Variable(M2.cuda())
                    gen_img, rs['test'] = self.optimize_multi_strategy(args,img0,gt0,M0,img1,gt1,M1,img2,gt2,M2,is_optimizer)
                    gen_RGB = convert(args,gen_img[0])
                    total_test_loss += rs['test'].data.cpu()
                avg_test_loss = total_test_loss/len(testloader)
                batches_done = epoch*len(data_loader) + i
                gen_RGB.save(args.results_img_dir + '%d-%d—best.png' % (epoch, batches_done))

                if epoch % 50 == 0:
                    torch.save(self.model.state_dict(), args.results_model_dir +'/%d_model.pkl'%(epoch))
                # save best model and img
                if avg_test_loss < best_loss :
                    best_loss = avg_test_loss
                    # torch.save(self.model.state_dict(), args.results_model_dir +'/%d-%d_best_model.pkl'%(epoch,batches_done))
                    torch.save(self.model.state_dict(), args.results_model_dir +'/best_model.pkl')
                    gen_RGB.save(args.results_img_dir +'%d-%d—---best.png' %(epoch,batches_done))

                writers[1].add_scalar('{}_test_loss'.format(args.model.arch),avg_test_loss.cpu(), epoch)
                print('[epoch %d/%d] [batch %d/%d] [loss: %f]' % (
                epoch, end_epoch, batches_done, (end_epoch * len(data_loader)), avg_test_loss.item()))

            if epoch == 250:
                self.optimizer.param_groups[0]['lr'] = 0.00004
            if epoch == 400:
                self.optimizer.param_groups[0]['lr'] = 0.000004


        print("total_consum_time = %.2f s" % (time.time()-begin_time))
    def train_same_scale(self,args,trainloader,testloader,start_epoch,end_epoch,):
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.results_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        begin_time = time.time()
        print("======== begin train model ========")
        print('data_size:',len(trainloader))
        #xs change
        writers = [SummaryWriter(log_dir='./mylogs/%s/%s' % (args.name, x)) for x in ['train', 'test']]
        best_loss = float('inf')

        #train
        for epoch in range(start_epoch,end_epoch):
            self.model.train()
            data_loader = tqdm(trainloader)
            data_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.end_epoch))
            #xs change
            rs = dict()
            total_train_loss = 0
            # xs change
            steps = 0
            for i, (img0,gt0,M0,img1,gt1,M1,img2,gt2,M2,name) in enumerate(data_loader):
                steps += 1
                if torch.cuda.is_available():
                    img0 = Variable(img0.cuda())
                    gt0 = Variable(gt0.cuda())
                    M0 = Variable(M0.cuda())
                    img1 = Variable(img1.cuda())
                    gt1 = Variable(gt1.cuda())
                    M1 = Variable(M1.cuda())
                    img2 = Variable(img2.cuda())
                    gt2 = Variable(gt2.cuda())
                    M2 = Variable(M2.cuda())
                else:
                    img0 = Variable(img0)
                    gt0 = Variable(gt0)

                # gen_img, loss = self.optimize_strategy(img,gt)
                is_optimizer = True
                gen_img, rs['train'] = self.optimize_multi_strategy(args,img0,gt0,M0,img0,gt0,M0,img0,gt0,M0,is_optimizer)
                total_train_loss += rs['train'].data.cpu()

            avg_train_loss = total_train_loss / len(trainloader)
            writers[0].add_scalar('{}_train_loss'.format(args.model.arch), avg_train_loss.cpu(), epoch)

            #xs change
            writers[0].add_scalar('lr',self.optimizer.param_groups[0]['lr'],epoch)
            # print(self.optimizer.param_groups[0]['lr'])
            # test
            if epoch % 5 == 0:
                self.model.eval()
                is_optimizer = False
                total_test_loss = 0
                data_loader = tqdm(testloader)
                data_loader.set_description('[%s %04d/%04d]' % ('test', epoch, args.end_epoch))
                # for i, (img0, gt0, M0, img1, gt1, M1, name) in enumerate(data_loader):
                for i, (img0,gt0,M0,img1,gt1,M1,img2,gt2,M2,name) in enumerate(data_loader):
                    img0 = Variable(img0.cuda())
                    gt0 = Variable(gt0.cuda())
                    M0 = Variable(M0.cuda())
                    img1 = Variable(img1.cuda())
                    gt1 = Variable(gt1.cuda())
                    M1 = Variable(M1.cuda())
                    img2 = Variable(img2.cuda())
                    gt2 = Variable(gt2.cuda())
                    M2 = Variable(M2.cuda())
                    gen_img, rs['test'] = self.optimize_multi_strategy(args,img0,gt0,M0,img0,gt0,M0,img0,gt0,M0,is_optimizer)
                    gen_RGB = convert(args,gen_img[0])
                    total_test_loss += rs['test'].data.cpu()
                avg_test_loss = total_test_loss/len(testloader)
                batches_done = epoch*len(data_loader) + i
                gen_RGB.save(args.results_img_dir + '%d-%d—best.png' % (epoch, batches_done))

                if epoch % 50 == 0:
                    torch.save(self.model.state_dict(), args.results_model_dir +'/%d_model.pkl'%(epoch))
                # save best model and img
                if avg_test_loss < best_loss :
                    best_loss = avg_test_loss
                    # torch.save(self.model.state_dict(), args.results_model_dir +'/%d-%d_best_model.pkl'%(epoch,batches_done))
                    torch.save(self.model.state_dict(), args.results_model_dir +'/best_model.pkl')
                    gen_RGB.save(args.results_img_dir +'%d-%d—---best.png' %(epoch,batches_done))

                writers[1].add_scalar('{}_test_loss'.format(args.model.arch),avg_test_loss.cpu(), epoch)
                print('[epoch %d/%d] [batch %d/%d] [loss: %f]' % (
                epoch, end_epoch, batches_done, (end_epoch * len(data_loader)), avg_test_loss.item()))

            if epoch == 250:
                self.optimizer.param_groups[0]['lr'] = 0.00004
            if epoch == 400:
                self.optimizer.param_groups[0]['lr'] = 0.000004


        print("total_consum_time = %.2f s" % (time.time()-begin_time))

    def optimize_strategy(self,img,gt,is_optimizer):
        # train generator#
        gen_imgs = self.model(img)
        mse_loss = get_mse_loss_function()
        loss = mse_loss(gen_imgs,gt)

        #BP
        if is_optimizer:
            self.optimizer_G.zero_grad()
            loss.backward()
            self.optimizer_G.step()

        return gen_imgs,loss

    def optimize_strategy_rsaunet(self,change_loss,img,gt,M,is_optimizer):
        # train generator#

        gen_imgs = self.model(img)
        mse_loss = get_mse_loss_function()
        loss = mse_loss(gen_imgs,gt)

        #BP
        if is_optimizer:
            self.optimizer_G.zero_grad()
            loss.backward()
            self.optimizer_G.step()

        return gen_imgs,loss


    def optimize_multi_strategy(self, args, img, gt, M, img1, gt1, M1, img2, gt2, M2, is_optimizer):

        gen_imgs = self.model(img)
        mse_loss = get_mse_loss_function()
        loss_0 = mse_loss(gen_imgs[2],gt)  #25
        loss = loss_0
        #BP
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return gen_imgs,loss


    def optimize_multi3_strategy(self, img0,gt0,M0,M1,M2,is_optimizer):

        gen_imgs = self.model(img0)
        mse_loss = get_mse_loss_function()
        loss_0 = mse_loss(gen_imgs[2],gt0)  #256

        loss = loss_0

        #BP
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return gen_imgs,loss






