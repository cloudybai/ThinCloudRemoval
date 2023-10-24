import torch
from torch.optim.lr_scheduler import MultiStepLR


def adjust_lr( optimizer):
    return MultiStepLR_HotFix(optimizer,[1500,2500])

class MultiStepLR_HotFix(MultiStepLR):
    def __init__(self,optimizer,milestones,gamma=0.1,last_epoch=-1):
        super(MultiStepLR_HotFix,self).__init__(optimizer,milestones,gamma,last_epoch)
        self.milestones = list(milestones)


