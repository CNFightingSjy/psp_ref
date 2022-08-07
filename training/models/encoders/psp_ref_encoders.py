import imp
from re import M
from telnetlib import SE
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear
from .spade import SPADE_change

'''
当前考虑该模块为map2style模块？
convs中包含了num_pools个数的卷积+LeakyReLU模块，通过卷积后进入一个输入输出同等大小的全连接层
'''
class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))# 通过spatial计算所需要的modules个数
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)# 生成512个向量送入StyleGAN
        return x

'''

'''
class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        # 获得对应的resnet
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.ref_input_layer = Sequential(Conv2d(opts.ref_input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        # 新增ref的特征金字塔
        modules = []
        modules_ref = []
        for block in blocks:
            # print(blocks)
            for bottleneck in block:
                # print(bottleneck)
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
                modules_ref.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        # 用一个特征金字塔
        self.body = Sequential(*modules)
        self.body_ref = Sequential(*modules_ref)
        self.spade_1 = SPADE_change(128, 128)
        self.spade_2 = SPADE_change(256, 256)
        self.spade_3 = SPADE_change(512, 512)
        # print(modules)

        # 根据所需要的style数获取style
        self.styles = nn.ModuleList()
        self.styles_ref = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        '''
        初始化所需要的所有map2style模块，训练一个小的映射网络从相应的特征图中提取所学的风格，style存储了获得的风格：
        风格0-2从小的特征图中产生
        风格3-6从中等的特征图中产生
        风格7-18从最大的特征图中产生
        新增增对ref的map2style模块
        '''
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
                style_ref = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
                style_ref = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
                style_ref = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
            self.styles_ref.append(style_ref)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, ref):
        x = self.input_layer(x)
        ref = self.ref_input_layer(ref)

        latents = []
        # 所有的map2style模块
        
        modulelist = list(self.body._modules.values())
        modulelist_ref = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            # 记录每个尺度下最后一层输出
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for i, l in enumerate(modulelist_ref):
            ref = l(ref)
            if i == 6:
                cf1 = ref
            elif i == 20:
                cf2 = ref
            elif i == 23:
                cf3 = ref

        # 在此处添加将不同尺度的特征使用spade进行融合
        c3 = self.spade_3(c3, cf3)
        c2 = self.spade_2(c2, cf2)
        c1 = self.spade_1(c1, cf1)
        '''
        self.coarse_ind = 3
        self.middle_ind = 7
        styles为各个map2style模块，用不同尺度的特征送入不同的map2style中获得对应的latent
        '''
        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))# 以c3作为作为styles中的第j个Module的输入，将结果存入latents中

        p2 = self._upsample_add(c3, self.latlayer1(c2))# layer1将256特征转为512
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))# layer2将128特征转为512
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        # 获得对应所需的resnet
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x
