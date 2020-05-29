# -*- coding: utf-8 -*-
"""
Created on 20-3-12
@funciotnL:load model
@author: mengzihan
"""
from __future__ import print_function
import os
import cv2
import torch
import numpy as np
import time
from PIL import Image
from torch.nn import DataParallel
from modelInsightFace import Backbone512,Backbone
import glob
import bcolz
from torchvision import transforms as trans
from sklearn import preprocessing

class FaceModel:
    def __init__(self, args):
        self.net_depth  = args.net_depth
        self.drop_ratio = args.drop_ratio
        self.net_mode = args.net_mode
        self.gpu_ids = args.device_ids
        if len(self.gpu_ids) != 0:

            print("gpu", "".join(list(map(str, self.gpu_ids))))
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(list(map(str,self.gpu_ids)))
            self.use_gpu = True
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            self.use_gpu = False
        self.model = self._loadInsightfacemodel(args.model,args.embedding_size)
        self.transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

    def _loadInsightfacemodel(self,ModelPath,embedding_size):
        if embedding_size==256:
            print("model 256")
            model = Backbone(self.net_depth, self.drop_ratio, self.net_mode)
        else:
            print("model 512")
            model = Backbone512(self.net_depth, self.drop_ratio, self.net_mode)
        model_dict = model.state_dict()
        # if self.use_gpu:
        #     pretrained_dict = torch.load(ModelPath,map_location='cpu' )
        # else:
        #     pretrained_dict = torch.load(ModelPath,map_location='cpu')
        pretrained_dict = torch.load(ModelPath, map_location='cpu')
        # print("load dict", pretrained_dict.keys())
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        # print("model dict",model_dict.keys())
        # print("update dict",pretrained_dict)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if self.use_gpu:
            model.cuda()
            model = DataParallel(model )
        # model.train(False)
        model.eval()
        return model
    def get_input(self, face_img):
        face_img = face_img[..., ::-1]
        data = Image.fromarray(face_img)
        data = self.transform(data)
        return data
    def get_feature(self,input):
        if self.use_gpu:
            input = input.cuda()
        else:
            input = input.cpu()
        embedding = self.model(input.unsqueeze(0))
        embedding = embedding.data.cpu().numpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding
    def get_input2(self, face_img):
        # print(type(face_img))
        face_img = face_img[..., ::-1]
        # face_img = face_img / 255.0
        # face_img = face_img - 0.5
        # face_img = face_img / 0.5
        face_img = face_img - 127.5
        face_img = face_img/ 127.5
        data = face_img.transpose((2, 0, 1))
        data = data.astype(np.float32, copy=False)
        data = torch.from_numpy(data)
        return data
    def get_input_batch(self, face_imgs):
        face_imgs = face_imgs[..., ::-1]
        # face_imgs = face_imgs / 255.0
        # face_imgs = face_imgs - 0.5
        # face_imgs = face_imgs / 0.5
        face_imgs = face_imgs - 127.5
        face_imgs = face_imgs / 127.5

        data = face_imgs.transpose((0,3,1,2))
        data = data.astype(np.float32, copy=False)
        data = torch.from_numpy(data)
        return data
    def get_feature_batch(self,inputs):
        if self.use_gpu:
            input = inputs.to(torch.device("cuda")).cuda()
        else:
            input = inputs.to(torch.device("cpu"))
        embedding = self.model(input)
        embedding = embedding.data.cpu().numpy()
        embedding = preprocessing.normalize(embedding)
        return embedding






