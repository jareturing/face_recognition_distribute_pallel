# -*- coding: utf-8 -*-
"""========================

@Project -> File : projects -> data_loader.py
@Date : Created on 2020/4/21 9:53
@author: wanghao
========================"""
import sys
import os
import torch.utils.data as data
import numpy as np
import cv2
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

def get_train_dataset(data_path):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    print(data_path)
    ds = ImageFolder(data_path, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num





class Data_Loader():
    def __init__(self,data_path=None,batch_size=64,p = 0.6,identity_size = 3,img_size=112,crop_size=100,
                rot_angle=10,num_workers=4,pin_memory=True,shuffle=True,**kwargs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.p = p
        self.identity_size = identity_size
        self.img_size = img_size
        self.crop_size = crop_size
        self.rot_angle = rot_angle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.num_classes = None
        self.data_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(rot_angle),
            transforms.RandomCrop(crop_size),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])])
        # self.data_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                          std=[0.5, 0.5, 0.5])
        # ])
        self.build_dataset()
        self.build_loader()
    def reload(self):
        print("Rebuilding loader . . .")
        self.build_loader()
    def __len__(self):
        return self.sample_dataset.__len__()/self.batch_size
    def build_dataset(self):
        self.sample_dataset, self.num_calsses = get_train_dataset(self.data_path)
    def build_loader(self):
        self.sample_loader = torch.utils.data.DataLoader(dataset=self.sample_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=self.shuffle,
                                                    pin_memory=self.pin_memory,
                                                    num_workers=self.num_workers,
                                                    )



#
# if __name__ == '__main__':
#     trainloader = Data_Loader(data_path='/home/njfh/hdd_data/mzh/datasets/cdpqq_new_aug.pkl', batch_size=512)
#     for i, (data, label) in enumerate(trainloader):
#         # imgs, labels = data
#         import pdb;pdb.set_trace()
#         print(data.numpy().shape)
#         print(label.shape)






# class Dataset(data.Dataset):
#
#     def __init__(self, data_path, phase='train'):
#         self.phase = phase
#         with open(os.path.join(data_path), 'r') as fd:
#             imgs = fd.readlines()
#
#         imgs = [img[:-1] for img in imgs]
#         self.imgs = np.random.permutation(imgs)
#
#         normalize = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
#         if self.phase == 'train':
#             self.transforms = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize
#             ])
#         else:
#             self.transforms = transforms.Compose([
#                 transforms.ToTensor(),
#                 normalize
#             ])
#
#     def __getitem__(self, index):
#         sample = self.imgs[index]
#         splits = sample.split()
#         img_path = splits[1]
#         # image = Image.open(img_path)
#         # if len(image.split())!=3:
#         #     #print(img_path,len(image.split()))
#         #     image = image.convert(mode='RGB')
#         img = cv2.imread(img_path)
#         # img = img[..., ::-1]
#         image = Image.fromarray(img)
#         data = self.transforms(image)
#         label = np.int32(splits[2])
#         return data.float(), label
#     def __len__(self):
#         return len(self.imgs)
#
# # def Data_Loader(data_path=None,batch_size=64,img_size=112,crop_size=100,
# #                        rot_angle=10,shuffle=True,num_workers=4,pin_memory=True,phase='train'):
# #     data_transform = transforms.Compose([
# #         transforms.RandomHorizontalFlip(),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.5,0.5,0.5],
# #                              std=[0.5,0.5,0.5])
# #     ])
# #     sample_dataset = Dataset(data_path=data_path)
# #     # data_sampler = data.distributed.Distributed(sample_dataset)
# #     sample_loader = torch.utils.data.DataLoader(dataset=sample_dataset,
# #                                                 batch_size=batch_size,
# #                                                 shuffle=shuffle,
# #                                                 # pin_memory=pin_memory,
# #                                                 num_workers=num_workers,
# #                                                 )
# #     return sample_loader
# if __name__ == '__main__':
#     dataset = Dataset(data_path='/home/mengzihan/insightface_v1/datasets/test/train.lst',
#                       phase='train')
#
#     trainloader = data.DataLoader(dataset, batch_size=512)
#     for i, (data, label) in enumerate(trainloader):
#         # imgs, labels = data
#         import pdb;pdb.set_trace()
#         print(data.numpy().shape)
#         print(label.shape)







# class Dataset(data.Dataset):
#     def __init__(self,data_path=None, transform=None):
#         self.data_path = data_path
#         self.transform = transform
#         if data_path.endswith("pkl"):
#             data = pkl.load(open(data_path,"rb"))
#             self.img_paths = data["img_paths"]
#             self.labels = data["labels"]
#             self.num_classes = data["num_classes"]
#         else:
#             self.img_paths,_ = get_all_images(data_path)
#             self.labels = self.get_label()
#             self.num_classes = len(set(self.labels))
#         self.img_paths,self.labels = concur_shuffle(self.img_paths,self.labels)
#     def __getitem__(self,index):
#         label = self.labels[index]
#         img_path = self.img_paths[index]
#         img = cv2.imread(img_path)
#         img = Image.fromarray(img)
#         if len(img.split()) != 3:
#             img = img.convert("RGB")
#         img = self.transform(img)
#         return img.float(), label
#     def __len__(self):
#         return len(self.img_paths)
#     def get_label(self):
#         dir_names = [x.split("/")[-2] for x in self.img_paths]
#         label_encoder = LabelEncoder()
#         labels = label_encoder.fit_transform(dir_names)
#         return labels
# def Data_Loader(data_path=None,batch_size=64,img_size=224,crop_size=200,
#                 rot_angle=10,shuffle=True,num_workers=4,pin_memory=False):
#     data_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5,0.5,0.5],
#                              std=[0.5,0.5,0.5])
#     ])
#
#     sample_dataset = Dataset(data_path=data_path, transform=data_transform)
#     # data_sampler = data.distributed.Distributed(sample_dataset)
#     sample_loader = torch.utils.data.DataLoader(dataset=sample_dataset,
#                                                 batch_size=batch_size,
#                                                 shuffle=shuffle,
#                                                 # pin_memory=pin_memory,
#                                                 num_workers=num_workers,
#                                                 )
#     return sample_loader