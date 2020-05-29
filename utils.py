# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> utils.py
@Date : Created on 2020/4/20 9:21
@author: wanghao
========================"""

import argparse
from importlib import import_module
import os
import io
import numpy as np
import pickle
import datetime
import os
import pdb
import shutil
import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse
from  datetime import datetime
from torchvision import transforms as trans
from importlib import import_module
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
def get_func(func_name):
    """An easy call function to get Module by name.
    """
    if func_name is None:
        return None
    parts = func_name.split(".")
    if len(parts) == 1:
        return globals()[parts[0]]
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])

def init_func_and_tag(func_name,kwargs):
    """An easy call function to get and init Module by name
    """
    if func_name is None:
        return None
    parts = func_name.split(".")
    if len(parts) == 1:
        return globals()[parts[0]],parts[0]
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    func = getattr(module, parts[-1])
    return func(**kwargs),parts[-2]


def parse_args():
    """An easy method get config file.
    """
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument(
        '--cfg', help='experiment configure file path', type=str, \
        default="config.yaml")
    parser.add_argument("--local_rank",default=0,type=int,help="")
    return parser.parse_args()

def get_all_images(root_path,tails=["jpg","png","JPG","PNG","jpeg","JPEG"]):
    """
        Get all the images paths under the root_path
    """
    paths = []
    names = []
    for root_dir, dirs,files in os.walk(root_path):
        for file in files:
            if file[-3:] in tails:
                paths.append(os.path.join(root_dir,file))
                names.append(file)
    return np.array(paths),np.array(names)
def concur_shuffle(img_paths,labels):
    """
        Shuffle image paths and labels simultaneously
    """
    index = np.arange(len(img_paths))
    img_paths = np.array(img_paths)
    labels = np.array(labels)
    np.random.shuffle(index)
    img_paths = img_paths[index]
    labels = labels[index]
    return img_paths,labels
def generate_list(data_path):
    """
        Generate list file
    """
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    img_paths,_ = get_all_images(data_path,["jpg","png","JPG","PNG","jpeg","JPEG"])
    dir_names = [x.split("/")[-2] for x in img_paths]
    img_paths = np.asarray(img_paths,dtype=str)
    labels = label_encoder.fit_transform(dir_names)
    num_classes = len(set(dir_names))
    dic = {"num_classes":num_classes,"img_paths":img_paths,"labels":labels}
    pickle.dump(dic,open(data_path+".pkl","wb"))

def separate_fc_params(modules):
    """
        Separate params that are from fc and bn layers out of params from other layers
    """
    if not isinstance(modules, list):
        modules = list(modules.modules())
    paras_only_fc1 = []
    paras_wo_fc1 = []
    for layer in modules:
        # print(str(layer.__class__))
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'linear.Linear' in str(layer.__class__) or 'BatchNorm1d' in str(layer.__class__):
                paras_only_fc1.extend(layer.parameters())
            else:
                paras_wo_fc1.extend(layer.parameters())
    return paras_only_fc1, paras_wo_fc1
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
def get_time():
    """
    :return: time str lick 2020-04-12-12-34
    """
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf
def de_preprocess(tensor):
    return tensor*0.5 + 0.5
hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

########################################################################################################################

