# import cPickle
import _pickle as cPickle
import time
import os
import shutil
import csv
import mxnet as mx
import cv2
import numpy as np
def initialize(save_file_path):
    if (not os.path.exists(save_file_path)):
        os.mkdir(save_file_path)
    else:
        shutil.rmtree(save_file_path)
        os.mkdir(save_file_path)
def initialize_excel(similarity_save_name,flag):
    out = open(similarity_save_name, 'w')
    csv_write = csv.writer(out, dialect='excel')
    if flag == 'cal':
        label = ['Threshold', 'Fppi', 'Recall','topk1','topk10','topk20']
        csv_write.writerow(label)
    elif flag == 'result':
        label = ['QueryName', 'Result']
        csv_write.writerow(label)
    print('file simliraty save to {}'.format(similarity_save_name))
    return csv_write
def add_feature(Galleryname,GalleryFeature,Queryname,QueryFeature):
    """
    将两个特征合并，两个文件名列表合并
    :param Galleryname: 底库名
    :param GalleryFeature: 底库特征
    :param Queryname: 查询集名
    :param QueryFeature: 查询集特征
    :return: 合并后的名字列表，合并后的特征
    """
    Galleryname = Galleryname + Queryname
    GalleryFeature = mx.nd.concat(GalleryFeature,QueryFeature,dim=0)
    return  Galleryname,GalleryFeature
