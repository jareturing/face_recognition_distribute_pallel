# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> evaluate.py
@Date : Created on 2020/4/27 17:17
@author: wanghao
========================"""
import sys
sys.path.append("../")
from data_loaders.data_pipe import de_preprocess, get_train_loader, get_val_data,get_val_data_own,get_val_pair,\
                                   gen_inputs,get_feature_batch

from io_defination import Search,Verify
from util import create_result_excel,create_pairs,create_result_excel_1vs1
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

import glob
from common_metric import evaluate
import pickle
import argparse
import datetime,time

class Evaluate(nn.Module):
    def __init__(self, opt=None):
        super(Evaluate,self).__init__()
        self.net = get_func(opt.backbone)
        self.backbone_kwargs = opt.backbone_kwargs
        self.net_tag = opt.backbone.split(".")[-1]

        self.use_gpu = opt.use_gpu
        self.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")] if isinstance(opt.gpu_ids, str) \
            else [int(x) for x in opt.gpu_ids]
        self.print_freq = opt.print_freq
        self.model_weight_path = opt.model_weight_path
        self.save_log_path = os.path.join(opt.save_log_path,self.model_weight_path.split("/")[-1][:-4])
        self.verify = opt.verify
        self.identify = opt.identify
        self.dataset_loader = get_func(opt.dataset_loader)

        self.data_path_verify = opt.data_path_verify
        self.data_path_query = opt.data_path_query
        self.data_path_gallery = opt.data_path_gallery
        self.verify_data_kwargs = opt.verify_data_kwargs
        self.query_data_kwargs = opt.query_data_kwargs
        self.gallery_data_kwargs = opt.gallery_data_kwargs

        self.date_tag = str(datetime.datetime.today())[:10]  # to mark save files
        #########################################################################
        print("MODEL BUILDING UP ...")
        if not os.path.exists(self.save_log_path): os.makedirs(self.save_log_path)
        self.build_model()
        self.get_loaders()
        self.tag = self.net_tag + "_"  + self.date_tag  # merge all the tags

    def get_log(self,tag):
        file = open(os.path.join(self.save_log_path,"{}_{}.txt".format(tag,self.tag)),"w")
        time_str=time.asctime(time.localtime(time.time()))
        file.write(time_str+"\n")
        return file
    def logging(self,file,*args):
        for item in args:
            file.writelines(item)
            file.write("\n")
            print(item)
        file.flush()
    def get_loaders(self):
        if self.verify:
            self.company_path_verify =  os.path.join(self.data_path_verify,"company1")
            self.pair_path = os.path.join(self.data_path_verify, "company1_pairs.txt")
            self.val_img_paths, self.pair_values = create_pairs(self.pair_path, self.company_path_verify) # 2N img_paths, N pair_values
            self.verify_loader = self.dataset_loader(data_path=self.val_img_paths, **self.verify_data_kwargs)
        if self.identify:
            self.query_loader = self.dataset_loader(data_path=self.data_path_query, **self.query_data_kwargs)
            self.gallery_loader = self.dataset_loader(data_path=self.data_path_gallery, **self.gallery_data_kwargs)

    def build_model(self):
        print("MODULE INITIALIZATION ...")
        self.net = self.net(**self.backbone_kwargs)

        if self.model_weight_path.startswith("http"): # load from url
            self.net.load_state_dict(torch.hub.load_state_dict_from_url(self.model_weight_path))
        else:
            state_dict = torch.load(self.model_weight_path,map_location="cpu")
            self.net.load_state_dict(state_dict)
        # torch.distributed.init_process_group(backend="nccl")
        if not torch.cuda.is_available() or not self.use_gpu:
            self.device = torch.device("cpu")
            self.net = self.net.cpu()
        else:
            self.device = torch.device("cuda:{}".format(self.gpu_ids[0]))
            self.net = self.net.to(self.device)
        self.net.eval()
    def hflip_batch(self,batch):
        idx = torch.arange(111,-1,-1)
        flip_batch = batch[:,:,:,idx]
        return flip_batch
    def eval_identify(self):
        query_img_paths = []
        query_embeddings = []
        gallery_img_paths = []
        gallery_embeddings = []
        with torch.no_grad():
            print("query data inferring . . .")
            for i,(imgs,img_paths) in enumerate(self.query_loader):
                features = self.net(imgs.to(self.device))
                features = F.normalize(features)
                features = features.cpu().numpy()
                query_embeddings.append(features)
                query_img_paths.extend(img_paths)
            print("gallery data inferring . . .")
            tic = time.time()
            for i,(imgs,img_paths) in enumerate(self.gallery_loader):
                if i%self.print_freq == 0:
                    dur = time.time()-tic
                    print("{}/{} step forwarding inferring [{}min {}s]".format(i,len(self.gallery_loader),dur//60,int(dur%60)))
                features = self.net(imgs.to(self.device))
                features = F.normalize(features)
                features = features.cpu().numpy()
                gallery_embeddings.append(features)
                gallery_img_paths.extend(img_paths)

        query_embeddings = np.concatenate(query_embeddings,axis=0)
        gallery_embeddings = np.concatenate(gallery_embeddings,axis=0)
        ##################################################################################################
        # search
        print("searching for result . . .")
        se = Search()
        all_search_reuslt = se.search(query_img_paths, query_embeddings, gallery_img_paths, gallery_embeddings)  # quary_names,
        # write results to csv
        save_dir = os.path.join(self.save_log_path,"1vsN_" + self.tag + ".xls")
        create_result_excel(save_dir, all_search_reuslt, zhengjian_num=se.zhengjian_num,
                            huoti_num=se.huoti_num, gallerynum=len(gallery_img_paths))
        print("table has been saved into {}".format(save_dir))
    def eval_verify(self):
        # get name and feature from query path and gallery path
        embeddings = []
        img_paths = []
        with torch.no_grad():
            print("pair data inferring . . .")
            for i,(imgs,img_paths_) in enumerate(self.verify_loader):
                features = self.net(imgs.to(self.device))
                features = F.normalize(features)
                features = features.cpu().numpy()
                embeddings.append(features)
                img_paths.extend(img_paths_)
        embeddings = np.concatenate(embeddings, axis=0)
        ve = Verify(self.pair_path, self.company_path_verify)
        all_search_reuslt = ve.verify(embeddings, img_paths, self.pair_values)
        save_dir = os.path.join(self.save_log_path,"1vs1_" + self.tag + ".xls")
        # print(result)
        ##################################################################################################
        # write results to csv
        create_result_excel_1vs1(save_dir, all_search_reuslt)
        print("table has been saved into {}".format(save_dir))
    def eval(self):
        eval_log = self.get_log("eval")
        self.logging(eval_log,"Model Loaded from {}".format(self.model_weight_path))
        # scheduler = StepLR(optimizer=self.optimizer,step_size = self.hypers["lr_step"],gamma=self.hypers["gamma"])
        self.logging(eval_log,"START EVALUATION!!!")
        # cfp_fp, lfw,cfp_fp_issame,lfw_issame = get_val_data(args.test_path)
        agedb, shujutang, company, agedb_issame, shujutang_issame, company_issame, jiankong, jiankong_issame = get_val_data_own(
            self.data_path_verify)

        print("agedb dataset inferring...")
        accuracy1, best_threshold1 = self.eval_verify_online(agedb, agedb_issame, nrof_folds=10)
        self.logging(eval_log, "agedb:", "acc:{}; threshold:{}".format(accuracy1, best_threshold1))
        print("shujutang dataset inferring...")
        accuracy2, best_threshold2 = self.eval_verify_online(shujutang, shujutang_issame, nrof_folds=10)
        self.logging(eval_log, "shujutang:", "acc:{}; threshold:{}".format(accuracy2, best_threshold2))
        print("company dataset inferring...")
        accuracy3, best_threshold3 = self.eval_verify_online(company, company_issame, nrof_folds=10)
        self.logging(eval_log, "company:", "acc:{}; threshold:{}".format(accuracy3, best_threshold3))
        print("jiankong dataset inferring...")
        accuracy4, best_threshold4 = self.eval_verify_online(jiankong, jiankong_issame, nrof_folds=6)
        self.logging(eval_log, "jiankong:", "acc:{}; threshold:{}".format(accuracy4, best_threshold4))
        eval_log.close()
        if self.identify:
            self.eval_identify()
        if self.verify:
            self.eval_verify()
    def eval_verify_online(self, carray, issame, embedding_size=256, batch_size=512, nrof_folds=10, tta=False):
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + batch_size])
                # print(batch_size)
                if tta:
                    fliped = self.hflip_batch(batch)
                    emb_batch = self.net(batch.to(self.device)) + self.net(fliped.to(self.device))
                    embeddings[idx:idx + batch_size] = F.normalize(emb_batch)
                else:
                    embeddings[idx:idx + batch_size] = self.net(batch.to(self.device)).cpu()
                idx += batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = self.hflip_batch(batch)
                    emb_batch = self.net(batch.to(self.device)) + self.net(fliped.to(self.device))
                    embeddings[idx:] = F.normalize(emb_batch)
                else:
                    embeddings[idx:] = self.net(batch.to(self.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        return accuracy.mean(), best_thresholds.mean()


def get_func(func_name):
    """An easy call function to get Module by name.
    """
    from importlib import import_module
    if func_name is None:
        return None
    parts = func_name.split('.')
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])
def parse_args():
    """An easy method get config file.
    """
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument(
        '--cfg', help='experiment configure file path', type=str, \
        default="validation.config.Config")
    return parser.parse_args()


if __name__ == "__main__":
    # get config
    args = parse_args()
    opt = args.cfg[:-3] if args.cfg.endswith("py") else args.cfg
    opt = get_func(opt)
    # initialize and run up !
    solver = Evaluate(opt)
    solver.eval()























