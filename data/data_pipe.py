from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader,BatchSampler,SequentialSampler,IterableDataset,RandomSampler,DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import math
import pickle
import torch
import mxnet as mx
import h5py
from tqdm import tqdm
from collections import defaultdict
import io
from tfrecord.torch.dataset import TFRecordDataset
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
def decode_image(features,transform=train_transform):
    imgByteArr=io.BytesIO(features["image"])
    PIL_img=Image.open(imgByteArr, mode='r')
    ##################################
    if transform is not None:
        image = PIL_img.convert('RGB')
        sample = transform(image)
    features["image"]=sample
    features["name"]=str(features["name"],encoding="utf-8")
    return features
class dataset_h5_concurrency(torch.utils.data.Dataset):
    def __init__(self, in_file,transform=train_transform):
        super(dataset_h5_concurrency, self).__init__()
        # self.file = h5py.File(in_file, 'r',libver='latest',swmr=True)
        self.file = in_file
        self.transform=transform
    def __getitem__(self, index):
        #open file every time to get one data,too cost
        with h5py.File(self.file, 'r') as file:
            image_array=np.uint8(file['dataset_images'][index, :])
            PIL_img=Image.fromarray(image_array)
            image=PIL_img.convert('RGB')
            if self.transform is not None:
                sample = self.transform(image)
            # label = self.file['dataset_labels'][index,:]
            label = file['dataset_labels'][index]
        # print (sample,label)
        return sample,label
    def __len__(self):
        with h5py.File(self.file, 'r') as db:
            self.n_images, self.image_h, self.image_w,image_c = db['dataset_images'].shape
        return self.n_images
class MyBatchSampler(BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last=False,dataset=None,split_part=2,class_num=1000000):
        super(MyBatchSampler,self).__init__(sampler, batch_size, drop_last)
        self.split_part=split_part
        self.class_per_split=math.ceil(class_num/split_part)
        self.dataset=dataset

    def __iter__(self):
        # batch = []
        batch_dict=defaultdict(list)
        for idx in self.sampler:
            # print('dataset idx',idx)
            # print('data label class_index of the target class',self.dataset.__getitem__(idx)[-1])
            class_index=self.dataset.__getitem__(idx)[-1]
            i_dt=math.floor(class_index/self.class_per_split)
            batch_dict[i_dt].append(idx)
            if len( batch_dict[i_dt]) == self.batch_size:
                yield  batch_dict[i_dt]
                batch_dict[i_dt] = []
        for i_dt in range(self.split_part):
            if len(batch_dict[i_dt]) > 0 and not self.drop_last:
                yield batch_dict[i_dt]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_loader(conf,worker_rank,use_hdf5=True,use_tfrecord=False):
    if use_hdf5:
        class_num = 144752
        # class_num = 1013232
        train_sampler = None
        hdf5file = str(conf.ms1m_folder) + '/datasets_miracle_v2_part' + str(worker_rank) + '.hdf5'
        # hdf5file = '/sdd_data/100WID_part' + str(worker_rank) + '.hdf5'
        print("using hdf5 file,", hdf5file, ",DataLoader with multi process")
        # datah5=dataset_h5(hdf5file)
        datah5 = dataset_h5_concurrency(hdf5file)
        # loader =DataLoader(dataset=datah5, batch_size=conf.batch_size,  shuffle=False, pin_memory=conf.pin_memory,
        #                     num_workers=0)
        loader = DataLoader(dataset=datah5, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers)

    elif use_tfrecord:
        # class_num=144752
        # tfrecord_path =str(conf.ms1m_folder) + '/datasets_miracle_v2_part_' + str(worker_rank) + '.tfrecord'
        # index_path = str(conf.ms1m_folder) + '/datasets_miracle_v2_part_' + str(worker_rank) + '.idx'
        # tfrecord_path = str(conf.ms1m_folder) + '/100WID_part_' + str(worker_rank) + '.tfrecord'
        # index_path = str(conf.ms1m_folder) + '/100WID_part_' + str(worker_rank) + '.idx'
        tfrecord_path = str(conf.data_path) + '/100WID_part_' + str(worker_rank) + '.tfrecord'
        index_path = str(conf.data_path) + '/100WID_part_' + str(worker_rank) + '.idx'
        class_num = 1013232
        train_sampler = None
        print("use tfrecord file",tfrecord_path)
        description = {"image": "byte", "label": "int", "index": "int", "name": "byte"}
        dataset = TFRecordDataset(tfrecord_path, index_path, description,shuffle_queue_size=conf.batch_size*conf.num_workers*10, transform=decode_image)
        # dataset = TFRecordDataset(tfrecord_path, index_path, description, transform=decode_image)
        # loader = torch.utils.data.DataLoader(dataset,  batch_size=conf.batch_size ,shuffle=False,   pin_memory=conf.pin_memory,
        #                     num_workers=conf.num_workers,drop_last=False,sampler=train_sampler)
        loader = DataLoaderX(dataset, batch_size=conf.batch_size, shuffle=False,
                                             pin_memory=conf.pin_memory,
                                             num_workers=conf.num_workers, drop_last=False, sampler=train_sampler)
    else:
        print("use data folder")
        # ds, class_num = get_train_dataset(conf.ms1m_folder / 'datasets_miracle_v2')
        ds, class_num = get_train_dataset(conf.data_path / '100WID')

        train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
        loader = DataLoader(ds, batch_size=conf.batch_size,  shuffle=(train_sampler is None), pin_memory=conf.pin_memory,
                            num_workers=0,drop_last=False,sampler=train_sampler)
        # loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=(train_sampler is None), pin_memory=conf.pin_memory,
        #                     num_workers=conf.num_workers, drop_last=False, sampler=train_sampler)
        print('ms1m ImageFlod generated:class ', class_num)
    print("check Data class num:",class_num)
    return loader, class_num ,train_sampler
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame
def get_val_data_own(data_path):
    agedb, agedb_issame = get_val_pair(data_path, 'agedb')
    shujutang, shujutang_issame = get_val_pair(data_path, 'shujutang')
    company, company_issame = get_val_pair(data_path, 'company')
    return agedb, shujutang, company, agedb_issame, shujutang_issame, company_issame
def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)

# class train_dataset(Dataset):
#     def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
#         self.imgs = bcolz.carray(rootdir = imgs_bcolz)
#         self.labels = bcolz.carray(rootdir = label_bcolz)
#         self.h_flip = h_flip
#         self.length = len(self.imgs) - 1
#         if h_flip:
#             self.transform = trans.Compose([
#                 trans.ToPILImage(),
#                 trans.RandomHorizontalFlip(),
#                 trans.ToTensor(),
#                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         self.class_num = self.labels[-1] + 1
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, index):
#         img = torch.tensor(self.imgs[index+1], dtype=torch.float)
#         label = torch.tensor(self.labels[index+1], dtype=torch.long)
#         if self.h_flip:
#             img = de_preprocess(img)
#             img = self.transform(img)
#         return img, label