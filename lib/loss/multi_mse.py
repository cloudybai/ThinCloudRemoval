
import torch.nn as nn

mse_loss = nn.MSELoss()

def muti_mse_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = mse_loss(d0,labels_v)
	loss1 = mse_loss(d1,labels_v)
	loss2 = mse_loss(d2,labels_v)
	loss3 = mse_loss(d3,labels_v)
	loss4 = mse_loss(d4,labels_v)
	loss5 = mse_loss(d5,labels_v)
	loss6 = mse_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))

	return loss0, loss