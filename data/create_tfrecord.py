# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> data_loader_eval.py
@Date : Created on 2020/6/29 09:23
@author: chengxioayang
========================"""
from PIL import Image
from tqdm import tqdm
from time import time
from tfrecord.torch.dataset import TFRecordDataset
from tfrecord.torch.dataset import MultiTFRecordDataset
import  numpy as np
from tfrecord.writer import TFRecordWriter
from tfrecord.tools.tfrecord2idx import create_index
import io
import os,glob,sys
import h5py
import cv2
import glob
import math
import torch
import random
import os
from util import find_classes,make_dataset,IMG_EXTENSIONS,make_dataset_fromtxt
from torchvision import transforms as trans
from multiprocessing import Pool,Manager
train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
def decode_image(features,transform=train_transform):
    #####################################################
    imgByteArr=io.BytesIO(features["image"])
    Pil=Image.open(imgByteArr, mode='r')
    ###debud#########################
    if transform is not None:
        image = Pil.convert('RGB')
        sample = transform(image)
    ##################################
    features["image"] = sample
    features["name"]=str(features["name"],encoding="utf-8")
    return features

def decode_image_save(features,transform=train_transform):
    #####################################################
    imgByteArr=io.BytesIO(features["image"])
    Pil=Image.open(imgByteArr, mode='r')
    ###debud#########################
    label_dir=str(features["label"][0])
    target_dir=os.path.join("/ssd_data/tfrecord_47w_ori_img",label_dir)
    if not os.path.exists(target_dir):
       os.makedirs(target_dir)
    #else:
    #    print("target more files",target_dir)
    if transform is not None:
        image = Pil.convert('RGB')
        #tick=time()
        features["name"]=str(features["name"],encoding="utf-8")
        img_name=os.path.split(features["name"])[-1]
        save_name=os.path.join(target_dir, img_name+'.png')
        Pil.save(save_name)
        # print ("Pil RGB",np.array(image))
        sample = transform(image)
    ##################################
    features["image"] = sample
    # features["name"]=str(features["name"],encoding="utf-8")
    return features

def pool_thread_tfrecord(file_part_idx, split_part_num,samples_images_path_classidx,total_class_num, tfrecord_path):
    tfrecord_head, ext = os.path.splitext(tfrecord_path)
    sample_idx_start = file_part_idx * split_part_num
    sample_idx_end = (file_part_idx + 1) * split_part_num
    tfrecord_path_part = tfrecord_head + '_part_' + str(file_part_idx) + ext
    tfrecord_path_part_idx = tfrecord_head + '_part_' + str(file_part_idx)+'.idx'
    if os.path.exists(tfrecord_path_part):
        print(tfrecord_path_part," file already exists,auto remove it now ..")
        os.remove(tfrecord_path_part)
    if os.path.exists(tfrecord_path_part_idx):
        print(tfrecord_path_part_idx, " file already exists,auto remove it now ..")
        os.remove(tfrecord_path_part_idx)
    print("create dataset file part", tfrecord_path_part)
    writer = TFRecordWriter(tfrecord_path_part)
    index = sample_idx_start
    try:
        for sample in tqdm(samples_images_path_classidx[sample_idx_start:sample_idx_end]):
            image_path, label = sample
            image_pil = Image.open(image_path, mode='r')
            image_pil = image_pil.convert("RGB")
            imgByteArr = io.BytesIO()
            image_pil.save(imgByteArr, format='png') #always the same data
            image_bytes = imgByteArr.getvalue()
            writer.write({"image": (image_bytes, "byte"),
                          "label": (label, "int"),
                          "index": (index, "int"),
                          "name": (bytes(image_path, encoding="utf8"), "byte")})
            index += 1
        writer.close()
        print("create idx file :", tfrecord_path_part_idx)
        create_index(tfrecord_path_part, tfrecord_path_part_idx)
        print("finish idx file")
    except Exception as e:
        print("pool process exception",e)
def torch_writer_tfrecord_multiprocess(image_floader,shuffle=True,split_rank_num=8):
    tfrecord_path = image_floader+".tfrecord"
    classes, class_to_idx = find_classes(image_floader)
    total_class_num = len(classes)
    if total_class_num % 8 !=0:
        print("total class num cannot div by GPU nums,check dataset set")
    samples_images_path_classidx = make_dataset(image_floader, class_to_idx, extensions=IMG_EXTENSIONS,
                                                is_valid_file=None)
    #samples_images_path_classidx = make_dataset_fromtxt("./train_test.lst")
    total_sample_num = len(samples_images_path_classidx)
    if shuffle:
        random.shuffle(samples_images_path_classidx)
    split_part_num=math.ceil(total_sample_num/split_rank_num)
    while len(samples_images_path_classidx)<split_rank_num*split_part_num:
        print("To make data part equal,some sample replicate times")
        samples_images_path_classidx.append(samples_images_path_classidx[random.randint(0,total_sample_num)])
    total_sample_num = len(samples_images_path_classidx)
    print("class num",total_class_num,"sample num",total_sample_num)
    worker = split_rank_num
    pool = Pool(worker)
    try:
        for i_cont in range(split_rank_num):
            pool.apply_async(pool_thread_tfrecord,
                             (i_cont, split_part_num, samples_images_path_classidx, total_class_num, tfrecord_path))
        pool.close()
        pool.join()
    except Exception as e:
        print(e)
    finally:
        print ("success finish tfrecord file create!")
 ##############################tfrecord write############
def torch_read_tfrecord(image_path,part_idx=0):
    tfrecord_path = image_path+ '_part_' + str(part_idx)+".tfrecord"
    index_path = image_path+ '_part_' + str(part_idx)+".idx"
    # index_path =None
    description = {"image": "byte", "label": "int","index":"int","name":"byte"}
    batch_size = 1
    num_worker =6
    dataset = TFRecordDataset(tfrecord_path, index_path, description, shuffle_queue_size=batch_size*num_worker,
                              transform=decode_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size , shuffle=False, pin_memory=False,drop_last=False,
                            num_workers=num_worker)
    i=1
    for data in tqdm(loader):
        print("data",i,len(data["label"]),data["label"])
        #print(data["label"],data["image"])
        i+=1
def torch_read_multi_tfrecord():
    tfrecord_pattern = "/home/cxy/face_ms1s/{}.tfrecord"
    index_pattern = "/home/cxy/face_ms1s/{}.idx"
    splits = {
        "datasets_miracle_v2_part_0": 0.125,
        "datasets_miracle_v2_part_1": 0.125,
        "datasets_miracle_v2_part_2": 0.125,
        "datasets_miracle_v2_part_3": 0.125,
        "datasets_miracle_v2_part_4": 0.125,
        "datasets_miracle_v2_part_5": 0.125,
        "datasets_miracle_v2_part_6": 0.125,
        "datasets_miracle_v2_part_7": 0.125
    }
    description = {"image": "byte", "label": "int","index":"int","name":"byte"}
    batch_size = 96
    num_worker =6
    # dataset = TFRecordDataset(tfrecord_path, index_path, description, shuffle_queue_size=batch_size*num_worker,
    #                           transform=decode_image)
    dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description,shuffle_queue_size=batch_size*num_worker,transform=decode_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size , shuffle=False, pin_memory=False,drop_last=False,
                            num_workers=num_worker)
    i=1
    for data in tqdm(loader):
        print("data",i,len(data["label"]))
        i+=1
def torch_read_multi_tfrecord_save_img():
    tfrecord_pattern = "/ssd_data/train_data_all_tfrecord_47w/{}.tfrecord"
    index_pattern = "/ssd_data/train_data_all_tfrecord_47w/{}.idx"
    splits = {
        "trained_all_part_0": 0.125,
        "trained_all_part_1": 0.125,
        "trained_all_part_2": 0.125,
        "trained_all_part_3": 0.125,
        "trained_all_part_4": 0.125,
        "trained_all_part_5": 0.125,
        "trained_all_part_6": 0.125,
        "trained_all_part_7": 0.125
    }
    description = {"image": "byte", "label": "int","index":"int","name":"byte"}
    batch_size = 96
    num_worker =6
    # dataset = TFRecordDataset(tfrecord_path, index_path, description, shuffle_queue_size=batch_size*num_worker,
    #                           transform=decode_image)
    dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description,shuffle_queue_size=batch_size*num_worker,transform=decode_image_save)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size , shuffle=False, pin_memory=False,drop_last=False,
                            num_workers=num_worker)
    i=0
    for data in tqdm(loader):
        print("data",i,len(data["label"]))
        i+=1   
    print("total decode image:",i*batch_size) 
def get_floder_num(targetdir):
    gallery_base_imgs = glob.glob(targetdir + "/*/*.png")
    print("total images :",len(gallery_base_imgs))
if __name__=='__main__':
    # image_path="/sdd_data/dalian177_file_en"
    # image_path = "/sdd_data/jiankong"
    # image_path = "/sdd_data/livedata"
    # image_path="/raid_data/100WID"
    # image_path = "/sdd_data/100WID"
    # image_path = "/sdd_data/100WID"
    # image_path = "/home/sdd/cxy/croped_images"
    image_path='/home/DistributeDir/Data/dalian177_file_en'
    torch_writer_tfrecord_multiprocess(image_path)
    # image_path = "/ssd_data/train_data_all_tfrecord_47w/trained_all"
    # image_path = "/sdd_data/face_ms1s/datasets_miracle_v2"
    # torch_writer_tfrecord(image_path)
    # torch_writer_tfrecord_multiprocess(image_path)
    # torch_read_tfrecord(image_path)
    # torch_read_multi_tfrecord()
    # torch_read_tfrecord(image_path)
    # torch_read_multi_tfrecord_save_img()

