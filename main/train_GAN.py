#!/usr/bin/python3
# coding:utf-8



def train():
    config = Config().parse()
    args = parse_args(config)
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    if args.in_channels == 9:
        train_loader,test_loader = Get_dataloader(args)
    else:
        train_loader, test_loader = Get_dataloader_RGB(args)

    if args.model.arch == 'SpaCycleGAN':
        SpaCycleGAN_G_AB = SPANet()
        SpaCycleGAN_gen_AB = create_generator(args, SpaCycleGAN_G_AB)

        SpaCycleGAN_D_B = PixelDiscriminator(input_nc=args.in_channels)
        SpaCycleGAN_dis_B = create_discriminator(args, SpaCycleGAN_D_B)


        SpaCycleGAN_G_BA = SPANet()
        SpaCycleGAN_gen_BA = create_generator(args, SpaCycleGAN_G_BA)

        SpaCycleGAN_D_A = PixelDiscriminator(input_nc=args.in_channels)
        SpaCycleGAN_dis_A = create_discriminator(args, SpaCycleGAN_D_A)

        opt_gen = get_adam_optimizer_chain(args, SpaCycleGAN_gen_AB, SpaCycleGAN_gen_BA)
        # opt_dis = get_adam_optimizer_chain(args, SpaCycleGAN_dis_A, SpaCycleGAN_dis_B)
        opt_dis_A = get_adam_optimizer(args, SpaCycleGAN_dis_A)
        opt_dis_B = get_adam_optimizer(args, SpaCycleGAN_dis_B)
        # SpaCycleGAN = SpaCycleGANTrainer(args,SpaCycleGAN_gen_AB, SpaCycleGAN_gen_BA, SpaCycleGAN_dis_A, SpaCycleGAN_dis_B, opt_gen, opt_dis)
        SpaCycleGAN = SpaCycleGANTrainer(args,SpaCycleGAN_gen_AB, SpaCycleGAN_gen_BA, SpaCycleGAN_dis_A, SpaCycleGAN_dis_B, opt_gen, opt_dis_A,opt_dis_B)
        SpaCycleGAN.train(args, train_loader, test_loader, args.start_epoch, args.end_epoch)


if __name__ == '__main__':
    train()

