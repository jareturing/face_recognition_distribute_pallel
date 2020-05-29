# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> modules.py
@Date : Created on 2020/5/14 20:15
@author: wanghao
========================"""
import argparse
from importlib import import_module
import os
import numpy as np
import pickle
from torch import optim
import datetime
import math
import os
import pdb
import shutil
from utils import init_func_and_tag
import time
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch.autograd import Variable
import tqdm
import argparse
from importlib import import_module
from torch import optim,nn
from torch.optim.lr_scheduler import StepLR,MultiStepLR
from torchvision.models import resnet18
from torch.nn import Sequential



class Header(nn.Module):
    """
        Pairwise header or Classifier header
    """
    def __init__(self,
                 pairwise = True,
                 net_tag = "resnet50",
                 embedding_dim = 256,
                 num_classes = 1000,
                 config = None):
        super(Header,self).__init__()
        self.pairwise = pairwise
        self.metric_func = None
        self.miner = None
        self.xbm = None
        self.loss = None
        self.modules = []
        self.learnable_modules = [] # saves [module,tag,model_weight_path,optimizer,hypers]
        self.tag = net_tag
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        # initialize
        if pairwise:
            self.pairwiser_init(config)
        else:
            self.classifier_init(config)

    def init_func_and_tag(self,func_name,func_kwargs,implement=True):
        if implement:
            func,tag = init_func_and_tag(func_name,func_kwargs)
            self.modules.append(func)
            return func,tag
        else:
            return None,"None"

    def to(self,device):
        self.modules = [module.to(device) for module in self.modules if module]
    def parallel(self,gpu_ids):
        self.modules = [nn.DataParallel(module, device_ids = gpu_ids) for module in self.modules if module]

    def load(self):
        for (module,_,weight_path,_,_) in self.learnable_modules:
            if weight_path:
                module.load_state_dict(torch.load(weight_path))
    def save(self,epoch,save_weight_path):
        for (module,tag,_,_,_) in self.learnable_modules:
            save_dir = os.path.join(save_weight_path,"epoch{}_{}.pth".format(epoch,tag))
            torch.save(self.net.module.state_dict(), save_dir)

    def pairwiser_init(self,pairwiser):
        self.loss,loss_tag = self.init_func_and_tag(pairwiser.loss,pairwiser.loss_kwargs,True)
        # miner
        self.miner, miner_tag = self.init_func_and_tag(pairwiser.miner.miner,pairwiser.miner.miner_kwargs,pairwiser.miner.use_miner)
        # cross batch memory
        xbm_kwargs = {"loss":self.loss,"embedding_size":self.embedding_dim,"miner":self.miner}
        xbm_kwargs.update(pairwiser.xbm.xbm_kwargs)
        self.xbm, xbm_tag = self.init_func_and_tag(pairwiser.xbm.xbm,xbm_kwargs,pairwiser.xbm.use_xbm)
        self.tag = "_".join([self.tag,miner_tag,xbm_tag,loss_tag])
    def classifier_init(self,classifier):
        self.loss, loss_tag = self.init_func_and_tag(classifier.loss,classifier.loss_kwargs,True)
        kwargs = {"in_features":self.embedding_dim, "out_features":self.num_classes}
        kwargs.update(classifier.metric.metric_kwargs)
        self.metric_func, metric_tag = self.init_func_and_tag(classifier.metric.metric_func,kwargs,True)
        self.tag = "_".join([self.tag, metric_tag,loss_tag])
        # add metric func into learnable modules
        self.learnable_modules.append([self.metric_func,
                                       metric_tag,
                                       classifier.metric.model_weight_path,
                                       classifier.metric.optimizer,
                                       classifier.metric.hypers])
    def forward(self,embeddings,labels,**kwargs):
        outputs = None
        if self.pairwise:
            if self.xbm:
                loss = self.xbm(embeddings,labels)
            else:
                loss = self.loss(embeddings,labels)
        else:
            outputs = self.metric_func(embeddings,labels)
            loss = self.loss(outputs,labels)
        return loss, outputs



class OptimFactory():
    def __init__(self,params,rigid_lr=True,milestones=[10,20,30],**kwargs):
        """
            Ensample multi optimizers and schedulers
        """
        self.optimizers = []
        self.rigid_lr = rigid_lr
        self.milestones = milestones
        self.schedulers = []
        self.add_optim(params, **kwargs)

    def lr_step(self,epoch=None,lr_decay=None):
        if self.rigid_lr: # rigid lr
            epoch = str(epoch)
            if epoch in lr_decay:
                lr_group = lr_decay[epoch]
                for i, optimizer in enumerate(self.optimizers):
                    if i < len(lr_group):
                        optimizer.param_groups[0]["lr"] = lr_group[i]
        else: # smooth lr
            for scheduler in self.schedulers:
                scheduler.step()
    def reset(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()
    def add_optim(self,
                  params,
                  optim_name="sgd",
                  lr=0.001,
                  weight_decay=0.95,
                  momentum=0.9,
                  **kwargs):
        if optim_name == "rmsp":
            optimizer = optim.RMSprop(
                params=params, weight_decay=weight_decay, lr=lr, momentum=momentum)
        elif optim_name == "adam":
            optimizer = optim.Adam(
                params=params, weight_decay=weight_decay, lr=lr)
        else:
            optimizer = optim.SGD(
                params=params, weight_decay=weight_decay, lr=lr, momentum=momentum)
        self.optimizers.append(optimizer)
        if not self.rigid_lr:
            # self.schedulers.append(StepLR(optimizer=optimizer, **kwargs)) #  step_size, gamma
            self.schedulers.append(MultiStepLR(optimizer=optimizer, milestones=self.milestones, **kwargs))


class IOFactory():
    def __init__(self, save_path,worker_rank, tag, vis=True, log_name="train"):
        """
            Make paths for all useful files and enables logging
        """
        self.tag = tag + "_" + str(datetime.datetime.today())[:10]+"_rank"+str(worker_rank)
        self.save_path = os.path.join(save_path, self.tag)
        # model weight path
        self.save_weight_path = os.path.join(self.save_path, "model_weights")
        self.make_path(self.save_weight_path)
        # log file
        self.get_log(log_name)
        # tensorboard file
        if vis:
            self.vis_dir = os.path.join(self.save_path, "tensorboard")
            self.writer = SummaryWriter(self.vis_dir)
        else:
            self.vis_dir = None
            self.writer = None

    def flush(self):
        self.log_file.flush()
        if self.writer: self.writer.flush()
    def close(self):
        self.log_file.close()
        if self.writer: self.writer.close()
    def make_path(self, p):
        if not os.path.exists(p):
            os.makedirs(p)
    def get_log(self, name="train"):
        self.log_file = open(os.path.join(self.save_path, "{}.txt".format(name)), "w")
        time_str = time.asctime(time.localtime(time.time() + 8 * 3660))
        self.log_file.write(time_str + "\n")
    def logging(self, *args):
        for item in args:
            self.log_file.writelines(item)
            self.log_file.write("\n")
            # print(item)
        self.log_file.flush()





