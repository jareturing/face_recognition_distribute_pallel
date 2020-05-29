# -*- coding: utf-8 -*-
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter,init
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb

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




class Backbone(Module):
    def __init__(self, num_layers=50, drop_ratio=0.4, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR


        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64, eps=2e-5, momentum=0.9),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512, eps=2e-5, momentum=0.9),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 256),
                                       BatchNorm1d(256, eps=2e-5, momentum=0.9,affine=True))
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
        self.body = Sequential(*modules)

    def forward(self, x):

        x = self.input_layer(x)

        x = self.body(x)
        #print(x)

        x = self.output_layer(x)

        return l2_norm(x)

        # x = self.body(x)
        # x = self.output_layer(x)
        # return l2_norm(x)
class Arcface_MV(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, conf,embedding_size=512,t=0.2, classnum=51332,  s=64., m=0.5,easy_margin=True):
        super(Arcface_MV, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        #self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        init.xavier_uniform_(self.kernel)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        self.conf =conf
        self.easy_margin = easy_margin
        self.t = t
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)

        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        gt = cos_theta[torch.arange(0, nB), label].view(-1, 1)  # ground truth score

        sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
        gt_cos_theta =  gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)
        mask = cos_theta > gt_cos_theta
        hard_vector = cos_theta[mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        output[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive

        if self.easy_margin:
            final_gt = torch.where(gt > 0, gt_cos_theta, gt)
        else:
            final_gt = cos_theta_m
        output.scatter_(1, label.data.view(-1, 1), final_gt)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        loss = self.conf.ce_loss(output, label)
        return loss
class CircleLossLikeCE(Module):
    def __init__(self, conf,m=0.25, gamma=256,embedding_size=256,classnum=51332):
        super(CircleLossLikeCE, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = conf.ce_loss
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        init.xavier_uniform_(self.kernel)
    def forward(self, embbedings,label) :
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)

        inp = torch.mm(embbedings, kernel_norm)
        a = torch.clamp_min(inp + self.m, min=0).detach()
        src = torch.clamp_min(
            - inp.gather(dim=1, index=label.unsqueeze(1).long()) + 1 + self.m,
            min=0,
        ).detach()

        a.scatter_(dim=1, index=label.unsqueeze(1).long(), src=src)

        sigma = torch.ones_like(inp, device=inp.device, dtype=inp.dtype) * self.m
        src = torch.ones_like(label.unsqueeze(1), dtype=inp.dtype, device=inp.device) - self.m
        sigma.scatter_(dim=1, index=label.unsqueeze(1).long(), src=src)
        #print(a * (inp - sigma) * self.gamma,label.long())
        return self.loss(a * (inp - sigma) * self.gamma, label.long())
class CurricularFace(Module):
    def __init__(self, conf,in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        init.normal_(self.kernel, std=0.01)
        self.conf = conf
    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        output[mask] = hard_example * (self.t + hard_example)
        output.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = output * self.s
        loss = self.conf.ce_loss(output, label)
        #return output, origin_cos * self.s
        return loss