# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> solver.py
@Date : Created on 2020/4/18 11:36
@author: wanghao
========================"""
import torch
from torch import nn
from torch import optim
import datetime
import math
import os
import pdb
import shutil
import gc
import time
from PIL import Image
from tensorboardX import SummaryWriter
import seaborn as sbn
from matplotlib import pyplot as plt
from verifacation import evaluate
plt.switch_backend("agg")
import numpy as np
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
import argparse
from utils import get_func, parse_args, separate_fc_params, init_func_and_tag,AverageMeter,l2_norm,gen_plot,get_time
from modules import IOFactory,OptimFactory,Header
from torchvision import transforms as trans
from torch.optim.lr_scheduler import StepLR
data_loader = None
import mpu
import apex
from pathlib import Path
from  apex import amp
from data_loaders.data_pipe import  get_val_data,get_val_data_own
class Solver(nn.Module):
    def __init__(self, opt=None):
        super(Solver,self).__init__()
        self.net = get_func(opt.backbone)
        self.backbone_kwargs = opt.backbone_kwargs
        self.net_tag = opt.backbone.split(".")[-1]
        self.embedding_dim = opt.embedding_dim
        self.num_classes = opt.num_classes


        self.batch_size = opt.batch_size
        # self.print_freq = int(opt.print_freq)
        self.val_interval = opt.val_interval

        self.board_loss_freq=opt.board_loss_freq
        self.evaluate_freq=opt.evaluate_freq
        self.save_freq=opt.save_freq
        ####################################
        self.device =opt.device
        self.gpu= opt.gpu
        self.use_fp16=opt.use_fp16
        self.world_size=opt.world_size
        self.optimizer = opt.optimizer
        self.epoch = opt.epoch
        self.start_epoch = opt.start_epoch
        # lr and other hyper params
        self.rigid_lr = opt.rigid_lr
        self.lr_decay = opt.lr_decay
        self.milestones=opt.milestones
        self.backbone_hypers = opt.backbone_hypers
        self.arccos_hypers = opt.arccos_hypers

        self.dataset_loader = get_func(opt.dataset_loader)
        self.train_data_kwargs = opt.train_data_kwargs
        self.validate = opt.validate

        self.train_data = opt.train_data
        self.valdata_folder = opt.valdata_folder
        # self.valdata_folder = Path(opt.valdata_folder)

        self.visualize = opt.visualize
        self.vis_kwargs = opt.vis_kwargs
        self.pretrained = opt.pretrained
        self.save_path = opt.save_path+get_time()
        self.model_weight_path = opt.model_weight_path
        self.date_tag = str(datetime.datetime.today())[:10]  # to mark save files
        self.epoch_time = AverageMeter('Time', ':6.3f')
        self.FPS = AverageMeter('FPS', ':6.3f')
        #########################################################################
        print("MODEL BUILDING UP . . .")
        self.build_model()

    def init_model(self):
        # backbone init
        torch.cuda.set_device(self.gpu)
        self.net = self.net(**self.backbone_kwargs).cuda(self.gpu)
        self.worker_rank = mpu.get_model_parallel_rank()
        print('DistributeDataParallel worker rank', self.worker_rank)

        if self.use_fp16:
            self.net =  apex.parallel.convert_syncbn_model(self.net)
        else:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.header=mpu.ArcfaceColumnParallelLinear(embedding_size=self.embedding_dim, output_classs_size=self.num_classes, bias=False).cuda(self.device)
        print('model parallel heads generated :class',self.num_classes)
        self.header.tag="ArcfacePallelheader_"+str(self.worker_rank)
        # optimizer init
        self.optim_fac = OptimFactory(
            params = self.net.parameters(),
            rigid_lr=self.rigid_lr,
            milestones=self.milestones,
            **self.backbone_hypers
        )
        # io factory init
        self.io_fac = IOFactory(
            save_path = self.save_path,
            worker_rank=self.worker_rank,
            tag = self.header.tag,
            vis=self.visualize,
            log_name="train"
        )
    def load_weight(self):
        if self.pretrained:
            if self.model_weight_path.startswith("http"): # load from url
                self.net.load_state_dict(torch.hub.load_state_dict_from_url(self.model_weight_path))
            else:
                model_dict=self.net.state_dict()
                state_dict = torch.load(self.model_weight_path,map_location='cpu')
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                print("ready to update dict item len:", len(pretrained_dict.keys()))
                self.net.load_state_dict(model_dict)
            # self.header.load()
    def build_model(self):
        print("MODULE INITIALIZATION . . .")
        self.init_model()
        self.optim_fac.add_optim(
            params =  self.header.parameters(),
            **self.arccos_hypers
        )
        if self.use_fp16:
            [self.net,self.header],self.optim_fac.optimizers=amp.initialize([self.net,self.header],optimizers=self.optim_fac.optimizers,opt_level="O1")
            self.net=apex.parallel.DistributedDataParallel(self.net)
        else:
            self.net=torch.nn.parallel.DistributedDataParallel(self.net,device_ids=[self.gpu])
        print("loader model weight")
        self.load_weight()
        # data loader init
        print("Data Loader building up . . .")
        self.train_loader = self.dataset_loader(data_path=self.train_data,
                                            batch_size=self.batch_size,worker_rank=self.worker_rank, **self.train_data_kwargs)
        self.epoch_iter_len=self.train_loader.__len__()
        self.board_loss_every =self.epoch_iter_len // self.board_loss_freq
        self.evaluate_every = self.epoch_iter_len // self.evaluate_freq
        self.save_every = self.epoch_iter_len // self.save_freq
        print("loader eval data ...")
        self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
            self.valdata_folder)
      
    def save_state(self,  accuracy, model_only=True,amp_model=False,extra=None):
        checkpoint = None
        torch.save(
            self.net.state_dict(),os.path.join( self.io_fac.save_weight_path ,
            ('model_{}_accuracy_{}_step_{}_{}.pth'.format(get_time(), accuracy, self.steps, extra))))
        if not model_only:
            torch.save(
                self.header.state_dict(), os.path.join( self.io_fac.save_weight_path ,
                ('head_{}_accuracy_{}_step_{}_{}.pth'.format(get_time(), accuracy, self.steps, extra))))
            for opt_idx,optimizer in enumerate(self.optim_fac.optimizers):
                torch.save(
                    optimizer.state_dict(),os.path.join(  self.io_fac.save_weight_path ,
                    ('optimizers_{}_accuracy_{}_step_{}_{}.pth'.format(get_time(), accuracy, self.steps, opt_idx))))
            if self.use_fp16 and amp_model:
                checkpoint = {
                    'net': self.net.state_dict(),
                    'header':self.header.state_dict(),
                    'optimizer': [optimizer.state_dict() for  optimizer in self.optim_fac.optimizers] ,
                    'amp': amp.state_dict()
                }
                torch.save(checkpoint, os.path.join(  self.io_fac.save_weight_path ,
                    ('amp_{}_accuracy_{}_step_{}_{}.pth'.format(get_time(), accuracy, self.steps, extra))))
    def heatmapper(self,embeddings=None,labels=None,steps=None):
        with torch.no_grad():
            fig = plt.figure(figsize=self.vis_kwargs["hm_size"])
            embs = embeddings.data.cpu().numpy()
            labs = labels.data.cpu().numpy()
            mat = np.dot(embs, embs.T)
            data_frame = pd.DataFrame(np.around(mat, decimals=4), columns=labs,index=labs)
            sbn.heatmap(data_frame, annot=False)
            self.io_fac.writer.add_figure("heatmap", figure=fig, global_step=(steps))

    def train(self):
        self.io_fac.logging("Model Loaded from {} ".format(self.model_weight_path))
        self.steps = 0
        self.running_loss=0
        self.company_accuracy=0
        self.io_fac.logging("START TRAINING!!!")
        for epoch in range(self.start_epoch,self.start_epoch+self.epoch):
            self.net.train()
            if epoch !=self.start_epoch:
                self.optim_fac.lr_step(epoch,self.lr_decay)
            epoch_start = time.time()
            idx=0
            for optim in self.optim_fac.optimizers:
                state = optim.state_dict()["param_groups"][0]
                state = ["{} : {}\n".format(k, v) for k, v in state.items() if k !="params"]
                self.io_fac.logging(state)

            # for i,(imgs,labels) in enumerate(self.train_loader.sample_loader):
            fps_start= time.time()
            ###########################if use tfrecord file as data####################################
            #for data in tqdm(self.train_loader.sample_loader):
            #    imgs = data["image"].cuda(self.gpu,non_blocking=True)
            #    labels = torch.squeeze(data["label"],dim=1).cuda(self.gpu,non_blocking=True).long()
            ################################use imgs from Folders##########################  
            for images,label in tqdm(self.train_loader.sample_loader):
                imgs = images.cuda(self.gpu,non_blocking=True)
                labels =label.cuda(self.gpu,non_blocking=True).long()
                ##################################################
                embeddings = self.net(imgs)
                logist, gather_label = self.header(embeddings, labels)
                mpu_loss = mpu.vocab_parallel_cross_entropy(logist.contiguous().float(),
                                                            gather_label)
                loss = torch.sum(mpu_loss.view(-1)) / (self.world_size * self.batch_size)
                loss_val=loss.data.detach()
                self.optim_fac.reset()
                if self.use_fp16:
                    with amp.scale_loss(loss, self.optim_fac.optimizers) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optim_fac.step()
                self.running_loss+=loss_val
                fps_time_cost = time.time() - fps_start
                self.steps+=1
                idx+=1
                if self.steps % self.board_loss_every == 0:
                    self.visual_disp(epoch,idx, loss_val, imgs,embeddings, labels,fps_time_cost)
                if self.steps % self.evaluate_every==0:
                    self.online_val()
                if self.steps%self.save_every==0:
                    self.save_state(self.company_accuracy,True,extra='test')
                fps_start = time.time()
            self.epoch_time.update(time.time() - epoch_start)
            print('epoch: ', epoch,"time:", self.epoch_time)


    def visual_disp(self,epoch,idx,loss_val,imgs,embeddings,labels,fps_time_cost):
        self.io_fac.writer.add_scalar("scalar/batch_loss", loss_val, self.steps)
        self.io_fac.writer.add_scalar("scalar/running_loss", self.running_loss / self.board_loss_every, self.steps)
        self.running_loss = 0
        self.FPS.update(self.batch_size * self.world_size / fps_time_cost)
        # self.io_fac.writer.add_scalar('train_FPS', self.FPS.avg, self.steps)
        self.io_fac.writer.add_scalar('train_FPS', self.FPS.avg, self.steps)
        if self.visualize:
            self.io_fac.writer.add_scalar("scalar/net_lr", self.optim_fac.optimizers[0].param_groups[0]["lr"],
                                          self.steps)
            self.io_fac.writer.add_scalar("scalar/cos_lr", self.optim_fac.optimizers[1].param_groups[0]["lr"],
                                          self.steps)
            if self.steps % (self.board_loss_every * self.vis_kwargs["emb_delay"]) == 0:
                self.heatmapper(embeddings, labels, self.steps)
                self.io_fac.writer.add_embedding(
                    mat=embeddings,
                    metadata=labels,
                    label_img=imgs,
                    global_step=self.steps
                )
                self.io_fac.flush()
            self.io_fac.logging(
                "epoch {} iter {}/{}, loss = {:.4}, time_span = {}". \
                    format(epoch, idx, self.epoch_iter_len, loss_val, self.epoch_time.avg))



    def online_val(self):
        accuracy1, best_threshold, roc_curve_tensor = self.evaluate( self.agedb_30, self.agedb_30_issame)
        self.__board_val('val/agedb_30', accuracy1, best_threshold, roc_curve_tensor)
        accuracy2, best_threshold, roc_curve_tensor = self.evaluate( self.lfw, self.lfw_issame)
        self.__board_val('val/lfw', accuracy2, best_threshold, roc_curve_tensor)
        accuracy3, best_threshold, roc_curve_tensor = self.evaluate(self.cfp_fp, self.cfp_fp_issame)
        self.__board_val('val/cfp_fp', accuracy3, best_threshold, roc_curve_tensor)
      
        self.company_accuracy=accuracy3
        #pass # Online validation is to be added !
    def __board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.io_fac.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.steps)
        self.io_fac.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.steps)
        self.io_fac.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.steps)
    def evaluate(self, carray, issame, nrof_folds = 5, tta = False):
        self.net.eval()
        idx = 0
        embeddings = np.zeros([len(carray), self.embedding_dim])
        with torch.no_grad():
            while idx + self.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + self.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.net(batch.to(self.device)) + self.net(fliped.to(self.device))
                    embeddings[idx:idx + self.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + self.batch_size] = self.net(batch.to(self.device)).cpu()
                idx += self.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.net(batch.to(self.device)) + self.net(fliped.to(self.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.net(batch.to(self.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        self.net.train()
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
if __name__ == "__main__":
    # get config
    from easydict import EasyDict
    import yaml
    args = parse_args()
    # opt = args.cfg[:-3] if args.cfg.endswith("py") else args.cfg
    # opt = get_func(opt)
    opt = EasyDict(yaml.load(open(args.cfg,"r"),Loader = yaml.Loader))
    # specify gpus to use
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # initialize and run up !
    solver = Solver(opt)
    solver.train()

