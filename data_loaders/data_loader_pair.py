# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> data_loader_pair.py
@Date : Created on 2020/5/12 16:24
@author: wanghao
========================"""

import sys
import os
import torch.utils.data as data
import numpy as np
import cv2
import time
import pdb
from scipy import misc
from numpy.random import randint, shuffle, choice
from PIL import Image
import torchvision.transforms as transforms

import torch
import pickle as pkl
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder
import torch.backends.cudnn as cudnn
sys.path.append("..")
from utils import concur_shuffle, get_all_images, generate_list

class IdentityLabel(object):
    def __init__(self,img_paths,labels,batch_size = 64,p = 0.6,identity_size = 3):
        self.labels = np.array(labels)
        self.img_paths = np.array(img_paths)
        self.batch_size = batch_size
        self.p = p
        self.identity_size = identity_size
        self.labels_squeeze = np.array(list(set(self.labels)))
        self.num_classes = len(self.labels_squeeze)
        self.label2img_paths = None

    def identity_shuffle(self):
        for k in self.label2img_paths:
            np.random.shuffle(self.label2img_paths[k])
    def concur_shuffle(self):
        idx = np.arange(len(self.img_paths))
        np.random.shuffle(idx)
        self.labels = self.labels[idx]
        self.img_paths = self.img_paths[idx]
    def get_label2img_paths(self):
        # get lab-img_paths dict
        self.concur_shuffle()
        self.label2img_paths = dict(zip(self.labels_squeeze, [[] for x in range(self.num_classes)]))
        for i in range(len(self.labels)):
            label = self.labels[i]
            img_path = self.img_paths[i]
            self.label2img_paths[label].append(img_path)

    def gen_batch_line(self):
        """
            Generate batch line according to batch size
        """
        if  self.p * self.batch_size / self.identity_size < 2: # Make sure we can select at least 2 pos pairs
            self.identity_size = 2
        # select pos pairs
        num_pos_pairs = int(self.p * self.batch_size/self.identity_size) + 1
        self.pos_pair_paths = []
        self.pos_labels = []

        np.random.shuffle(self.labels_squeeze)
        # it's too slow to make random choice
        for i,label in enumerate(self.labels_squeeze):
            identity_img_paths = self.label2img_paths[label][:self.identity_size]
            if len(identity_img_paths) < self.identity_size:
                identity_img_paths = np.random.choice(self.label2img_paths[label],self.identity_size ,replace=True)
            self.pos_pair_paths.append(identity_img_paths)
            self.pos_labels.extend([label] * self.identity_size)
        self.pos_pair_paths = np.stack(self.pos_pair_paths,axis=0)
        self.pos_labels  = np.array(self.pos_labels).reshape(-1,self.identity_size)

        # batch line length alignment
        if len(self.pos_pair_paths) % num_pos_pairs != 0:
            overflow = num_pos_pairs - len(self.pos_pair_paths) % num_pos_pairs
            self.pos_pair_paths = np.concatenate([self.pos_pair_paths,self.pos_pair_paths[:overflow]],axis=0)
            self.pos_labels = np.concatenate([self.pos_labels,self.pos_labels[:overflow]],axis=0)

        self.pos_pair_paths = self.pos_pair_paths.reshape(-1,num_pos_pairs * self.identity_size)
        self.pos_labels = self.pos_labels.reshape(-1,num_pos_pairs * self.identity_size)

        # select neg samples
        num_neg_samples = (self.batch_size - num_pos_pairs * self.identity_size) * len(self.pos_pair_paths)
        neg_idx = np.random.randint(low=0,high=len(self.img_paths),size=(num_neg_samples))
        self.neg_img_paths = self.img_paths[neg_idx].reshape(len(self.pos_pair_paths),-1)
        self.neg_labels = self.labels[neg_idx].reshape(len(self.pos_pair_paths),-1)
        # mix up
        self.mix_img_paths = np.concatenate([self.pos_pair_paths,self.neg_img_paths],axis = 1).reshape(-1)
        self.mix_labels = np.concatenate([self.pos_labels,self.neg_labels],axis = 1).reshape(-1)



class Dataset(data.Dataset):
    def __init__(self,data_path=None, transform=None, batch_size = 64,p = 0.6,identity_size = 3):
        super(Dataset).__init__()
        self.data_path = data_path
        self.transform = transform
        if data_path.endswith("pkl"):
            data = pkl.load(open(data_path,"rb"))
            img_paths = data["img_paths"]
            labels = data["labels"]
        else:
            img_paths,_ = get_all_images(data_path)
            labels = self.get_label(img_paths)
        self.identity_label = IdentityLabel(img_paths,labels,batch_size,p,identity_size )
        self.num_classes = self.identity_label.num_classes
        self.identity_label.concur_shuffle()
        self.identity_label.get_label2img_paths()
        self.identity_label.gen_batch_line()

    def __getitem__(self,index):
        img_path = self.identity_label.mix_img_paths[index]
        label = self.identity_label.mix_labels[index]
        img = Image.open(img_path)
        # img = Image.fromarray(img)
        if len(img.split()) != 3:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.identity_label.mix_img_paths)
    def get_label(self,img_paths):
        dir_names = [x.split("/")[-2] for x in img_paths]
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(dir_names)
        return labels

class Data_Loader():
    def __init__(self,data_path=None,batch_size=64,p = 0.6,identity_size = 3,img_size=112,crop_size=100,
                rot_angle=10,num_workers=4,pin_memory=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.p = p
        self.identity_size = identity_size
        self.img_size = img_size
        self.crop_size = crop_size
        self.rot_angle = rot_angle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(rot_angle),
            transforms.RandomCrop(crop_size),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])])
        self.build_dataset()
        self.build_loader()
        self.num_classes = self.sample_dataset.num_classes
    def reload(self):
        print("Batchline Regenerating . . .")
        self.sample_dataset.identity_label.get_label2img_paths()
        self.sample_dataset.identity_label.gen_batch_line()
        print("Rebuilding loader . . .")
        self.build_loader()
    def __len__(self):
        return self.sample_dataset.__len__()/self.batch_size
    def build_dataset(self):
        self.sample_dataset = Dataset(data_path=self.data_path,
                                      transform=self.data_transform,
                                      batch_size=self.batch_size,
                                      p=self.p,
                                      identity_size=self.identity_size
                                      )
    def build_loader(self):
        self.sample_loader = torch.utils.data.DataLoader(dataset=self.sample_dataset,
                                                         batch_size=self.batch_size,
                                                         shuffle=False,
                                                         pin_memory=self.pin_memory,
                                                         num_workers=self.num_workers,
                                                         )



