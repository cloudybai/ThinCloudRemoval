#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import os
import numpy as np
from lib.loss.Ganloss import GANLoss
from tqdm import tqdm
from lib.utils.image_pool import ImagePool
import time
from lib.utils.convert_RGB import convert

import torch
from torch.autograd import Variable
from lib.loss.mse_loss import get_mse_loss_function
from lib.loss.L1_loss import get_l1_loss_function
from lib.utils.utils import ReplayBuffer, LambdaLR
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr

# !/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
import os
import numpy as np
from tqdm import tqdm

import time
from lib.utils.convert_RGB import convert

import torch
from torch.autograd import Variable
from lib.loss.mse_loss import get_mse_loss_function
from lib.loss.L1_loss import get_l1_loss_function
from lib.utils.utils import ReplayBuffer, LambdaLR
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from lib.loss.Ganloss import GANLoss


class SpaCycleGANTrainer(object):
    def __init__(self, args,gen_AB, gen_BA, dis_A, dis_B, optimizer_G, optimizer_D_A, optimizer_D_B):
        self.gen_AB = gen_AB.train()
        self.gen_BA = gen_BA.train()
        self.dis_A = dis_A.train()
        self.dis_B = dis_B.train()
        self.optimizer_G = optimizer_G
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B

    def train(self, args, trainloader, testloader, start_epoch, end_epoch):

        # patch = (1, args.img_height // (2 ** args.n_D_layers * 4), args.img_width // (2 ** args.n_D_layers * 4))
        fake_img_buffer = ReplayBuffer(args.pool_size)
        fake_gt_buffer = ReplayBuffer(args.pool_size)

        begin_time = time.time()
        print("======== begin train model ========")
        print('data_size:', len(trainloader))
        # xs change
        writers = [SummaryWriter(log_dir='./mylogs/%s/%s' % (args.name, x)) for x in ['train', 'test']]
        best_loss = float('inf')

        os.makedirs(args.results_gen_model_dir, exist_ok=True)
        os.makedirs(args.results_dis_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        os.makedirs(args.results_gt_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        gan_loss = GANLoss(args.gan_mode).cuda()

        for epoch in range(start_epoch, end_epoch):
            data_loader = tqdm(trainloader)
            data_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.end_epoch))
            rs = dict()
            total_train_loss = 0
            # xs change
            steps = 0

            for i, (img, gt, M, name) in enumerate(data_loader):
                if torch.cuda.is_available():
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                    M = Variable(M.cuda())
                else:
                    img = Variable(img)
                    gt = Variable(gt)



                ##### Train Generator #######
                self.optimizer_G.zero_grad()

                # identity loss
                identity_loss = get_l1_loss_function()
                _,id_img = self.gen_BA(img)
                _,id_gt = self.gen_AB(gt)
                loss_id_img = identity_loss(id_img, img) * args.lambda_B * args.lambda_idt
                loss_id_gt = identity_loss(id_gt, gt) * args.lambda_A * args.lambda_idt
                # loss_identity = (loss_id_gt + loss_id_img) / 2

                # GAN loss
                M_, fake_gt = self.gen_AB(img)
                pred_fake_A = self.dis_A(fake_gt)
                gan_lossA = gan_loss(pred_fake_A,True)

                _, fake_img = self.gen_BA(gt)
                pred_fake_B = self.dis_B(fake_img)
                gan_lossB= gan_loss(pred_fake_B,True)

                # Cycle loss
                _, recov_img = self.gen_BA(fake_gt)
                cycle_loss = get_l1_loss_function()
                loss_cycle_img = cycle_loss(recov_img, img) * args.lambda_A

                _, recov_gt = self.gen_BA(fake_img)
                loss_cycle_gt = cycle_loss(recov_gt, gt) * args.lambda_B

                #attention loss
                MSE_loss = get_mse_loss_function()
                att_loss = MSE_loss(M_[:, 0, :, :], M)

                # Tota loss
                loss_G = loss_id_img + loss_id_gt + gan_lossA + gan_lossB + loss_cycle_img + loss_cycle_gt + att_loss

                loss_G.backward()
                self.optimizer_G.step()
                batches_done = epoch * len(trainloader) + i

                ####### Train Discriminator A #######
                self.optimizer_D_A.zero_grad()
                pred_real = self.dis_A(img)
                loss_real = gan_loss(pred_real, True)
                fake_img = fake_img_buffer.push_and_pop(fake_img)
                pred_fake = self.dis_A(fake_img.detach())
                loss_fake = gan_loss(pred_fake, False)

                loss_D_img = (loss_real + loss_fake) / 2
                loss_D_img.backward()
                self.optimizer_D_A.step()

                ####### Train Discriminator B #######
                self.optimizer_D_B.zero_grad()
                pred_real = self.dis_B(gt)
                loss_real = gan_loss(pred_real, True)
                fake_gt = fake_gt_buffer.push_and_pop(fake_gt)
                pred_fake = self.dis_B(fake_gt.detach())
                loss_fake = gan_loss(pred_fake, False)

                loss_D_gt = (loss_real + loss_fake) / 2
                loss_D_gt.backward()
                self.optimizer_D_B.step()



                ########## save best result ##############

                # if loss_G.data.cpu() < best_loss:
                #     best_loss = loss_G.data.cpu()
                #     torch.save(self.gen_AB.state_dict(), args.results_gen_model_dir + '/%d-%d_gen_AB_best_model.pkl'%(epoch,batches_done))
                #     torch.save(self.gen_BA.state_dict(), args.results_gen_model_dir + '/%d-%d_gen_BA_best_model.pkl'%(epoch,batches_done))
                #     torch.save(self.dis_A.state_dict(), args.results_dis_model_dir + '/%d-%d_dis_A_best_model.pkl'%(epoch,batches_done))
                #     torch.save(self.dis_B.state_dict(), args.results_dis_model_dir + '/%d-%d_dis_B_best_model.pkl'%(epoch,batches_done))
                #     save_image(fake_gt, '%s/%s-%s.bmp' % (args.results_gt_dir, epoch, batches_done), nrow=4,normalize=True)
                #     save_image(fake_img, '%s/%s-%s.bmp' % (args.results_img_dir, epoch, batches_done), nrow=4,normalize=True)

                if i % args.interval == 0:
                    print('[epoch %d/%d] [batch %d/%d] [loss: %f]' % (
                    epoch, end_epoch, batches_done, (end_epoch * len(trainloader)), loss_G.item()))

            if epoch % args.interval == 0 or epoch > args.end_epoch - 5:
                torch.save(self.gen_AB.state_dict(),
                           args.results_gen_model_dir + '/%d-%d_gen_AB.pkl' % (epoch, batches_done))
                torch.save(self.gen_BA.state_dict(),
                           args.results_gen_model_dir + '/%d-%d_gen_BA.pkl' % (epoch, batches_done))
                torch.save(self.dis_A.state_dict(),
                           args.results_dis_model_dir + '/%d-%d_dis_A.pkl' % (epoch, batches_done))
                torch.save(self.dis_B.state_dict(),
                           args.results_dis_model_dir + '/%d-%d_dis_B.pkl' % (epoch, batches_done))

                fake_gt_RGB = convert(fake_gt[0])
                fake_gt_RGB.save(args.results_gt_dir + '%d-%d.png' % (epoch, batches_done))

                fake_img_RGB = convert(fake_img[0])
                fake_img_RGB.save(args.results_img_dir + '%d-%d.png' % (epoch, batches_done))

                # save_image(fake_gt, '%s/%s-%s.bmp' % (args.results_gt_dir, epoch, batches_done), nrow=4, normalize=True)
                # save_image(fake_img, '%s/%s-%s.bmp' % (args.results_img_dir, epoch, batches_done), nrow=4,
                #            normalize=True)

        # f.close()
        writers.close()


# class SpaCycleGANTrainer(object):
    # def __init__(self,args,gen_AB,gen_BA,dis_A,dis_B,optimizer_G,optimizer_D):
    #     self.args = args
    #     self.netG_A = gen_AB.train()
    #     self.netG_B = gen_BA.train()
    #     self.netD_A = dis_A.train()
    #     self.netD_B = dis_B.train()
    #     self.optimizer_G = optimizer_G
    #     self.optimizer_D = optimizer_D
    #     # self.optimizer_D_A = optimizer_D_A
    #     # self.optimizer_D_B = optimizer_D_B
    #     self.criterionGAN = GANLoss(args.gan_mode).cuda()
    #     self.criterionCycle = torch.nn.L1Loss()
    #     self.criterionIdt = torch.nn.L1Loss()
    #     self.MSE_loss = get_mse_loss_function()
    #     self.fake_A_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images
    #     self.fake_B_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images
    #
    #
    # def set_requires_grad(self, nets, requires_grad=False):
    #     """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    #     Parameters:
    #         nets (network list)   -- a list of networks
    #         requires_grad (bool)  -- whether the networks require gradients or not
    #     """
    #     if not isinstance(nets, list):
    #         nets = [nets]
    #     for net in nets:
    #         if net is not None:
    #             for param in net.parameters():
    #                 param.requires_grad = requires_grad
    #
    # def backward_D_basic(self, netD, real, fake):
    #     """Calculate GAN loss for the discriminator
    #
    #     Parameters:
    #         netD (network)      -- the discriminator D
    #         real (tensor array) -- real images
    #         fake (tensor array) -- images generated by a generator
    #
    #     Return the discriminator loss.
    #     We also call loss_D.backward() to calculate the gradients.
    #     """
    #     # Real
    #     pred_real = netD(real)
    #     loss_D_real = self.criterionGAN(pred_real, True)
    #     # Fake
    #     pred_fake = netD(fake.detach())
    #     loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Combined loss and calculate gradients
    #     loss_D = (loss_D_real + loss_D_fake) * 0.5
    #     loss_D.backward()
    #     return loss_D
    #
    # def backward_D_A(self):
    #     """Calculate GAN loss for discriminator D_A"""
    #     fake_B = self.fake_B_pool.query(self.fake_B)
    #     self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    #     return self.loss_D_A
    #
    # def backward_D_B(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     fake_A = self.fake_A_pool.query(self.fake_A)
    #     self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    #     return self.loss_D_B
    #
    # def backward_G(self):
    #     """Calculate the loss for generators G_A and G_B"""
    #     lambda_idt = self.args.lambda_identity
    #     lambda_A = self.args.lambda_A
    #     lambda_B = self.args.lambda_B
    #     # Identity loss
    #     if lambda_idt > 0:
    #         # G_A should be identity if real_B is fed: ||G_A(B) - B||
    #         self.A1,self.idt_A = self.netG_A(self.real_B)
    #         self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
    #         # G_B should be identity if real_A is fed: ||G_B(A) - A||
    #         self.B1,self.idt_B = self.netG_B(self.real_A)
    #         self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    #     else:
    #         self.loss_idt_A = 0
    #         self.loss_idt_B = 0
    #
    #     #attention loss
    #     self.att_loss = self.MSE_loss(self.A1[:, 0, :, :], self.M)
    #
    #     # GAN loss D_A(G_A(A))
    #     self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
    #     # GAN loss D_B(G_B(B))
    #     self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
    #     # Forward cycle loss || G_B(G_A(A)) - A||
    #     self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
    #     # Backward cycle loss || G_A(G_B(B)) - B||
    #     self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
    #     # combined loss and calculate gradients
    #     self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B +self.att_loss
    #     self.loss_G.backward()
    #
    #     return self.loss_idt_A,self.loss_G
    #
    # def train(self,args,trainloader,testloader,start_epoch,end_epoch):
    #
    #     patch = (1,args.img_height//(2**args.n_D_layers*4), args.img_width//(2**args.n_D_layers*4))
    #     fake_img_buffer = ReplayBuffer()
    #     fake_gt_buffer = ReplayBuffer()
    #     begin_time = time.time()
    #
    #     writer = SummaryWriter(log_dir='{}'.format(args.log_dir),comment='train_loss')
    #     print("======== begin train model ========")
    #     print('data_size:',len(trainloader))
    #
    #     best_loss = 3
    #     os.makedirs(args.results_gen_model_dir, exist_ok=True)
    #     os.makedirs(args.results_dis_model_dir, exist_ok=True)
    #     os.makedirs(args.results_img_dir, exist_ok=True)
    #     os.makedirs(args.results_gt_dir, exist_ok=True)
    #     os.makedirs(args.log_dir, exist_ok=True)
    #
    #
    #     for epoch in range(start_epoch,end_epoch):
    #         self.netG_A.train()
    #         data_loader = tqdm(trainloader)
    #         data_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.end_epoch))
    #         total_id_loss = 0
    #         total_G_loss = 0
    #         total_DA_loss = 0
    #         total_DB_loss = 0
    #         steps = 0
    #         writers = [SummaryWriter(log_dir='./mylogs/%s/%s' % (args.name, x)) for x in ['train','test']]
    #
    #         for i, (img,gt,M,name) in enumerate(data_loader):
    #             if torch.cuda.is_available():
    #                 self.real_A = Variable(img.cuda())
    #                 self.real_B = Variable(gt.cuda())
    #                 self.M = Variable(M.cuda())
    #
    #             valid = Variable(torch.FloatTensor(np.ones((img.size(0), *patch))).cuda(), requires_grad=False)
    #             fake = Variable(torch.FloatTensor(np.zeros((img.size(0), *patch))).cuda(), requires_grad=False)
    #
    #             self.A1,self.fake_B = self.netG_A(self.real_A)  # G_A(A)
    #             self.recA1,self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
    #             self.B1,self.fake_A = self.netG_B(self.real_B)  # G_B(B)
    #             self.recB1,self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
    #
    #             self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
    #             self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
    #             loss_id,loss_G = self.backward_G()  # calculate gradients for G_A and G_B
    #
    #             total_id_loss += loss_id.data.cpu()
    #             total_G_loss += loss_G.data.cpu()
    #
    #             self.optimizer_G.step()  # update G_A and G_B's weights
    #             # D_A and D_B
    #             self.set_requires_grad([self.netD_A, self.netD_B], True)
    #             self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
    #             loss_D_A = self.backward_D_A()  # calculate gradients for D_A
    #             loss_D_B = self.backward_D_B()  # calculate graidents for D_B
    #             self.optimizer_D.step()  # update D_A and D_B's weights
    #
    #             total_DA_loss += loss_D_A.data.cpu()
    #             total_DB_loss += loss_D_B.data.cpu()
    #         avg_id_loss = total_id_loss / len(trainloader)
    #         avg_G_loss = total_G_loss / len(trainloader)
    #         avg_DA_loss = total_DA_loss / len(trainloader)
    #         avg_DB_loss = total_DB_loss / len(trainloader)
    #
    #         writers[0].add_scalars('{}_id_loss'.format(args.model.arch), {'id':avg_id_loss.cpu(),'G':avg_G_loss.cpu(),'DA':avg_DA_loss.cpu(),'DB':avg_DB_loss.cpu()}, epoch)
    #
    #
    #         #test
    #         if epoch % args.evaluation_interval == 0:
    #             self.netG_A.eval()
    #             total_test_loss = 0
    #             data_loader = tqdm(testloader)
    #             data_loader.set_description('[%s %04d/%04d]' % ('test', epoch, args.end_epoch))
    #             for i, (img, gt, M, name) in enumerate(data_loader):
    #                 img = Variable(img.cuda())
    #                 gt = Variable(gt.cuda())
    #                 M = Variable(M.cuda())
    #
    #                 A1, fake_B = self.netG_A(img)
    #                 B1, fake_A = self.netG_B(gt)
    #
    #             batches_done = epoch * len(data_loader) + i
    #
    #             gen_RGBB = convert(args, fake_B[0])
    #             gen_RGBA = convert(args, fake_A[0])
    #             gen_RGBB.save(args.results_gt_dir + '%d-%d.png' % (epoch, batches_done))
    #             gen_RGBA.save(args.results_img_dir + '%d-%d.png' % (epoch, batches_done))
    #
    #
    #
    #
    #             # if epoch % args.save_model == 0:
    #             #     torch.save(self.netG_A.state_dict(), args.results_model_dir + '/G_A_%d_model.pkl' % (epoch))
    #             #     torch.save(self.netD_A.state_dict(), args.results_model_dir + '/D_A_%d_model.pkl' % (epoch))
    #             #     torch.save(self.netG_B.state_dict(), args.results_model_dir + '/G_B_%d_model.pkl' % (epoch))
    #             #     torch.save(self.netD_B.state_dict(), args.results_model_dir + '/D_B_%d_model.pkl' % (epoch))
    #             # save best model and img
    #             if epoch % 50 == 0:
    #
    #                 torch.save(self.netG_A.state_dict(), args.results_model_dir + '/G_A_model.pkl')
    #                 torch.save(self.netD_A.state_dict(), args.results_model_dir + '/D_A_model.pkl')
    #                 torch.save(self.netG_B.state_dict(), args.results_model_dir + '/G_B_model.pkl')
    #                 torch.save(self.netD_B.state_dict(), args.results_model_dir + '/D_B_model.pkl')
    #
    #
    #         print("total_consum_time = %.2f s" % (time.time() - begin_time))





