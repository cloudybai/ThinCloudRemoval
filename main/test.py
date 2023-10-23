#!/usr/bin/python3
# coding:utf-8

import os
import sys
sys.path.append('..')

import torch

from lib.dataset.dataloader import Get_dataloader
from lib.model.create_generator import create_generator
from lib.model.generator.ms_net import MS_Net
from lib.model.generator.msa_net import MSA_Net
from lib.model.generator.rsc_net import RSC_Net
from lib.model.generator.ssa_net import SSA_Net
from lib.model.generator.ss_net import SS_Net
from lib.model.generator.unet import UNet
from lib.model.generator.u2net import U2NET
from lib.model.generator.pix2pixhd import Pix2pixhd
from lib.model.generator.multi_pix2pix import Multi_pix2pix
from lib.model.generator.RSAUnet import RSAUNet
from lib.model.generator.SPANet import SPANet,SPNet
from lib.model.generator.pix2pix import Pix2PixModel
from lib.model.generator.MSPF_Net import MSEFNet,MSPFNet,SSNet,OSPFNet,MSFFNet,MSCFNet

from lib.model.generator.cloudGan_gen import GeneratorResNet



from lib.utils.config import Config
from lib.utils.parse_args import parse_args
from rm_cloud.single_model_methods.tester import Testner

def test():
    config = Config().parse()
    args = parse_args(config)
    print(args)
    train_loader, test_loader = Get_dataloader(args)
    checkpoint = args.checkpoint
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    rsc = RSC_Net()
    msa = MSA_Net()
    ssa = SSA_Net()
    ms = MS_Net()
    ss = SS_Net()
    unet = UNet()
    u2net = U2NET()
    rsaunet = RSAUNet()
    pix2pixhd = Pix2pixhd()
    multi_pix2pix = Multi_pix2pix()
    spanet = SPANet()
    spnet = SPNet()


    # pix2pix = Pix2PixModel(opt=args)

    if args.model.arch == 'rsc':
        model = create_generator(args, rsc)
        model.load_state_dict(torch.load(checkpoint))
        rsc = Testner(model)
        rsc.test(args, test_loader)
        rsc.eval(args)

    if args.model.arch == 'msa':
        model = create_generator(args, msa)
        model.load_state_dict(torch.load(checkpoint))
        msa = Testner(model)
        msa.test(args, test_loader)
        msa.eval(args)

    if args.model.arch == 'ss':
        model = create_generator(args, ss)
        model.load_state_dict(torch.load(checkpoint))
        ss = Testner(model)
        ss.test(args, test_loader)
        ss.eval(args)

    if args.model.arch == 'unet':
        model = create_generator(args, unet)
        model.load_state_dict(torch.load(checkpoint))
        unet = Testner(model)
        unet.test(args, test_loader)
        unet.eval(args)

    if args.model.arch == 'u2net':
        model = create_generator(args, u2net)
        model.load_state_dict(torch.load(checkpoint))
        u2net = Testner(model)
        u2net.test(args, test_loader)
        u2net.eval(args)

    if args.model.arch == 'pix2pixhd':
        model = create_generator(args, pix2pixhd)
        model.load_state_dict(torch.load(checkpoint))
        pix2pixhd = Testner(model)
        pix2pixhd.test(args, test_loader)
        pix2pixhd.eval(args)

    if args.model.arch == 'multi_pix2pix':
        model = create_generator(args, multi_pix2pix)
        model.load_state_dict(torch.load(checkpoint))
        multi_pix2pix = Testner(model)
        multi_pix2pix.test(args, test_loader)
        multi_pix2pix.eval(args)


    if args.model.arch == 'ssa':
        model = create_generator(args, ssa)
        model.load_state_dict(torch.load(checkpoint))
        msa = Testner(model)
        msa.test(args, test_loader)
        msa.eval(args)

    if args.model.arch == 'ms':
        model = create_generator(args, ms)
        model.load_state_dict(torch.load(checkpoint))
        ms = Testner(model)
        ms.test(args, test_loader)
        ms.eval(args)
    if args.model.arch == 'msa_':
        model = create_generator(args, msa)
        model.load_state_dict(torch.load(checkpoint))
        msa_ = Testner(model)
        msa_.test(args, test_loader)
        msa_.eval(args)

    if args.model.arch == 'spanet':
        model = create_generator(args, spanet)
        model.load_state_dict(torch.load(checkpoint))
        msa_ = Testner(model)
        msa_.test(args, test_loader)
        msa_.eval(args)

    if args.model.arch == 'spnet':
        model = create_generator(args, spnet)
        model.load_state_dict(torch.load(checkpoint))
        spa = Testner(model)
        spa.test(args, test_loader)
        spa.eval(args)

    if args.model.arch == 'MSPF':
        if args.name == 'MSPF':
            msf_net = MSPFNet()
            model = create_generator(args, msf_net)
            model.load_state_dict(torch.load(checkpoint))
            Mspf = Testner(model)
            Mspf.test(args, test_loader)
            Mspf.eval(args)

        if args.name == 'MSSPF':
            msf_net = MSPFNet()
            model = create_generator(args, msf_net)
            model.load_state_dict(torch.load(checkpoint))
            Mspf = Testner(model)
            Mspf.test(args, test_loader)
            Mspf.eval(args)

        if args.name == 'MSEF':
            msf_net = MSEFNet()
            model = create_generator(args, msf_net)
            model.load_state_dict(torch.load(checkpoint))
            Mspf = Testner(model)
            Mspf.test(args, test_loader)
            Mspf.eval(args)

        if args.name == 'MSFF':
            msf_net = MSFFNet()
            model = create_generator(args, msf_net)
            model.load_state_dict(torch.load(checkpoint))
            Mspf = Testner(model)
            Mspf.test(args, test_loader)
            Mspf.eval(args)

        if args.name == 'MSCF':
            msf_net = MSCFNet()
            model = create_generator(args, msf_net)
            model.load_state_dict(torch.load(checkpoint))
            Mspf = Testner(model)
            Mspf.test(args, test_loader)
            Mspf.eval(args)

        if args.name == 'OSPF':
            msf_net = OSPFNet()
            model = create_generator(args, msf_net)
            model.load_state_dict(torch.load(checkpoint))
            Mspf = Testner(model)
            Mspf.test(args, test_loader)
            Mspf.eval(args)

        if args.name == 'SS':
            msf_net = SSNet()
            model = create_generator(args, msf_net)
            model.load_state_dict(torch.load(checkpoint))
            Mspf = Testner(model)
            Mspf.test(args, test_loader)
            Mspf.eval(args)


    # if args.model.arch == 'Multi3_SPANet':
    #     model = create_generator(args, multi3_spanet)
    #     model.load_state_dict(torch.load(checkpoint))
    #     multi3_spanet = Testner(model)
    #     multi3_spanet.test(args, test_loader)
    #     multi3_spanet.eval(args)

    if args.model.arch == 'cloudGan':
        cloudGan_G_AB = GeneratorResNet(args)
        model = create_generator(args,cloudGan_G_AB)
        model.load_state_dict(torch.load(checkpoint))
        cloudgan = Testner(model)
        cloudgan.test(args, test_loader)
        cloudgan.eval(args)

    if args.model.arch == 'rsaunet':

        model = create_generator(args, rsaunet)
        model.load_state_dict(torch.load(checkpoint))
        rasunet = Testner(model)
        rasunet.test(args, test_loader)
        rasunet.eval(args)


    # if args.model.arch == 'pix2pix':
    #
    #     model = create_generator(args, pix2pix)
    #     model.load_state_dict(torch.load(checkpoint))
    #     pix2pix = Testner(model)
    #     pix2pix.test(args, test_loader)
    #     pix2pix.eval(args)



if __name__ == '__main__':

    test()