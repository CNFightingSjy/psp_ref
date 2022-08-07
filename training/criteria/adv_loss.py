import argparse
import math
import random

from torch.nn import functional as F
from torch import nn

class AdvLoss(nn.Module):

    def __init__(self):
        super(AdvLoss, self).__init__()

    # def d_logistic_loss(real_pred, fake_pred):
    #     real_loss = F.softplus(-real_pred)
    #     fake_loss = F.softplus(fake_pred)

    #     return real_loss.mean() + fake_loss.mean()

    # def g_nonsaturating_loss(fake_pred):
    #     loss = F.softplus(-fake_pred).mean()

    #     return loss

    def forward(self, real ,fake, train_g):
        loss = 0
        real_loss = 0
        fake_loss = 0
        count = 0
        for i in range(len(fake)):
            if not train_g:
                real_loss = F.softplus(-real[i])
                fake_loss = F.softplus(fake[i])
                loss = real_loss.mean() + fake_loss.mean()
            elif train_g:
                loss = F.softplus(-fake[i]).mean()
            count += 1
        
        return loss/count