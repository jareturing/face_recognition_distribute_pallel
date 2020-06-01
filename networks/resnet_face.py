# -*- coding: utf-8 -*-
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module, Parameter,AdaptiveMaxPool2d,Softmax
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
"""
Resnet designed for face
"""
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks




class ChannelAttention(Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.max_pool = AdaptiveMaxPool2d(1)

        self.fc1   = Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = ReLU()
        self.fc2   = Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        #assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #padding = 3 if kernel_size == 7 else 1
        padding = 1
        self.conv1 = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        self.stride = stride
        if stride == 2:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride ,bias=False),
                                             BatchNorm2d(depth,eps=2e-5,momentum=0.9))

            self.res_layer = Sequential(
                BatchNorm2d(in_channel,eps=2e-5,momentum=0.9),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False),BatchNorm2d(depth,eps=2e-5,momentum=0.9),
                PReLU(depth),Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth,eps=2e-5,momentum=0.9))
        else:
            self.res_layer = Sequential(
                BatchNorm2d(in_channel,eps=2e-5,momentum=0.9),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False),BatchNorm2d(depth,eps=2e-5,momentum=0.9),
                PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth,eps=2e-5,momentum=0.9))

    def forward(self, x):
        if self.stride == 2:
            shortcut = self.shortcut_layer(x)
            res = self.res_layer(x)
            return res + shortcut
        else:
            res = self.res_layer(x)
            return res+x

class bottleneck_CBAM(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_CBAM, self).__init__()
        self.stride = stride
        self.res_layer = Sequential(
            BatchNorm2d(in_channel, eps=2e-5, momentum=0.9),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), BatchNorm2d(depth, eps=2e-5, momentum=0.9),
            PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth, eps=2e-5, momentum=0.9))
        self.ca_layer = ChannelAttention(depth)
        self.sa_layer = SpatialAttention()

        if stride == 2:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride ,bias=False),
                                             BatchNorm2d(depth,eps=2e-5,momentum=0.9))

    def forward(self, x):
        res = self.res_layer(x)
        res = self.ca_layer(res) * res
        res = self.sa_layer(res) * res

        if self.stride == 2:
            shortcut = self.shortcut_layer(x)
            return res + shortcut
        else:
            res = self.res_layer(x)
            return res+x

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DANetHead(Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 2   #256
        self.danet_conv5a = Sequential(Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    ReLU())

        self.danet_conv5c = Sequential(Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.danet_conv51 = Sequential(Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    ReLU())
        self.danet_conv52 = Sequential(Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    ReLU())

        self.danet_conv6 = Sequential(Dropout2d(0.1, False), Conv2d(256, out_channels, 1))
        self.danet_conv7 = Sequential(Dropout2d(0.1, False), Conv2d(256, out_channels, 1))
        self.danet_conv8 = Sequential(Dropout2d(0.1, False), Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.danet_conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.danet_conv51(sa_feat)
        sa_output = self.danet_conv6(sa_conv)

        feat2 = self.danet_conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.danet_conv52(sc_feat)
        sc_output = self.danet_conv7(sc_conv)

        #feat_sum = sa_conv + sc_conv
        feat_sum = sa_output + sc_output

        sasc_output = self.danet_conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        # return tuple(output)
        return sasc_output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False),
                BatchNorm2d(depth,eps=2e-5,momentum=0.9))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel,eps=2e-5,momentum=0.9),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth,eps=2e-5,momentum=0.9),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Backbone(Module):
    def __init__(self, num_layers=50, drop_ratio=0.6, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se','cbam','danet'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            print("THe mode is IR")
            unit_module = bottleneck_IR
        elif mode == 'cbam':
            print("The mode is CBAM")
            unit_module = bottleneck_CBAM
        elif mode == 'danet':
            print("The mode is danet")
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64, eps=2e-5, momentum=0.9),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512, eps=2e-5, momentum=0.9),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 256),
                                       BatchNorm1d(256, eps=2e-5, momentum=0.9, affine=False))
        modules = []
        item = 0
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
                # if item == 0:  # 修改第一个block的blockneck的结构
                #     modules[0].shortcut_layer = Sequential(
                #         Conv2d(bottleneck.in_channel, bottleneck.depth, (1, 1), bottleneck.stride, bias=False),
                #         BatchNorm2d(bottleneck.depth, eps=2e-5, momentum=0.9))
                # item += 1

        if mode == "danet":
            modules.append(DANetHead(512,512,BatchNorm2d))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

def resnet50(drop_ratio=0.4, mode='ir'):
    model = Backbone(num_layers=50,drop_ratio=drop_ratio,mode=mode)
    return model
def resnet100(drop_ratio=0.4, mode='ir'):
    model = Backbone(num_layers=100,drop_ratio=drop_ratio,mode=mode)
    return model
def resnet152(drop_ratio=0.4, mode='ir'):
    model = Backbone(num_layers=152,drop_ratio=drop_ratio,mode=mode)
    return model

