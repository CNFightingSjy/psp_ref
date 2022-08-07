import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.functional as F



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def affine_transformation(X, alpha, beta):
    x = X.clone()
    mean, std = calc_mean_std(x)
    mean = mean.expand_as(x)
    std = std.expand_as(x)
    return alpha * ((x-mean)/std) + beta


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True)):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation)
    def build_conv_block(self, dim, padding_type, norm_layer, activation):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x)
        return out





class SPADE_change(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
    # 分别是  input guide
    def forward(self, a_image, b_image):  # 这几步做消融实验，
        # Part 1.a_image generate parameter-free normalized
        # activationsa_image计算x的std和mean并将两个参数合并
        # normalized = self.param_free_norm(a_image)  # 基于x维度没变
        normalized = a_image
        mean, std = calc_mean_std(normalized)
        mean = mean.expand_as(normalized)
        std = std.expand_as(normalized)

        # Part 2.b_image produce scaling and bias conditioned on semantic map
        b_image = F.interpolate(b_image, size=a_image.size()[2:], mode='nearest')  # 基于x维度没变
        actv = self.mlp_shared(b_image)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)


        # part4: a and b
        result = gamma * ((normalized-mean)/std) + beta

        return result








# 现在用这个网络去进行创新性训练
class Resnet_Spade(nn.Module):
    def __init__(self, input_nc, guide_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d,
                 padding_type='reflect', bottleneck_depth=100):
        super(Resnet_Spade, self).__init__()

        self.activation = nn.ReLU(True)

        n_downsampling = 3

        self.spade_1 = SPADE_change(ngf, ngf)
        self.spade_2 = SPADE_change(ngf * 2, ngf * 2)
        self.spade_3 = SPADE_change(ngf * 4, ngf * 4)
        self.spade_4 = SPADE_change(ngf * 8, ngf * 8)



        ## input
        padding_in = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)]
        self.padding_in = nn.Sequential(*padding_in)
        self.conv1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)

        ## guide
        padding_g = [nn.ReflectionPad2d(3), nn.Conv2d(guide_nc, ngf, kernel_size=7, padding=0)]
        self.padding_g = nn.Sequential(*padding_g)
        self.conv1_g = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        self.conv2_g = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.conv3_g = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)


        resnet = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=self.activation, norm_layer=norm_layer)]
        self.resnet = nn.Sequential(*resnet)
        decoder = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                           output_padding=1),
                        norm_layer(int(ngf * mult / 2)), self.activation]
        self.pre_decoder = nn.Sequential(*decoder)
        self.decoder = nn.Sequential(
            *[nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()])

    def bottleneck_layer(self, nc, bottleneck_depth):
        return nn.Sequential(*[nn.Conv2d(nc, bottleneck_depth, kernel_size=1), self.activation,
                               nn.Conv2d(bottleneck_depth, nc, kernel_size=1)])

    def forward(self, input, guidance):
        input = self.padding_in(input)
        guidance = self.padding_g(guidance)

        input = self.spade_1(input,guidance)
        guidance = self.spade_1(guidance,input)


        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv1(input)
        guidance = self.conv1_g(guidance)


        input = self.spade_2(input, guidance)
        guidance = self.spade_2(guidance, input)

        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv2(input)
        guidance = self.conv2_g(guidance)


        input = self.spade_3(input, guidance)
        guidance = self.spade_3(guidance, input)

        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv3(input)
        guidance = self.conv3_g(guidance)



        input = self.spade_4(input, guidance)



        input = self.activation(input)

        input = self.resnet(input)
        input = self.pre_decoder(input)
        output = self.decoder(input)
        return output