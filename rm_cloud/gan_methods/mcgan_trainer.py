#!/usr/bin/python3
#coding:utf-8


from rm_cloud.gan_methods.trainer import Trainer

class McganTrainer(Trainer):

    def __init__(self,dis, gen , D_optimizer, G_optimizer):
        super().__init__(dis, gen , D_optimizer, G_optimizer)
