from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm
import os
from torch.utils import data
def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    print(imgs_folder)
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num
def get_train_mydataset(train_list):
    train_dataset = Dataset(train_list, phase='train')
    return train_dataset

def get_train_loader(conf):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder)
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder/'imgs')
        print('vgg loader generated')
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder+'/imgs')
    elif conf.data_mode == 'cadc':
        ds, class_num = get_train_dataset(conf.cadc_folder)
        # ds = get_train_mydataset(conf.train_list)
        # class_num = conf.num_classes

    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num 
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    print(path)
    bins, issame_list = pickle.load(open(path, 'rb'))
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path+'/'+name, mode='r')
    issame = np.load(path+'/'+'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

    # return cfp_fp, lfw, cfp_fp_issame, lfw_issame
def get_val_data_own(data_path):
    agedb, agedb_issame = get_val_pair(data_path, 'agedb')
    shujutang, shujutang_issame = get_val_pair(data_path, 'shujutang')
    company, company_issame = get_val_pair(data_path, 'company')
    jiankong, jiankong_issame = get_val_pair(data_path, 'jiankong')
    return agedb, shujutang, company, agedb_issame, shujutang_issame, company_issame ,jiankong, jiankong_issame

def load_mx_rec(rec_path):
    save_path = rec_path+'/imgs'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path+"/"+'train.idx'), str(rec_path+'/train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        #print(header.label)
        label = int(header.label[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)
        label_path = save_path+'/'+str(label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        img.save(label_path+'/{}.jpg'.format(idx), quality=95)

def gen_inputs(file_list, batch_size):
    """
    get file batch
    :param file_list: file name list
    :param batch_size:
    :return:
    """
    file_len = len(file_list)
    step_count = file_len // batch_size if file_len % batch_size == 0 else file_len // batch_size + 1
    for i in range(step_count):
        batch_name_list = file_list[i * batch_size: (i + 1) * batch_size]
        # print(len(batch_name_list))
        batch_img_list = []
        # start_time = datetime.datetime.now()
        for img_path in batch_name_list:
            img = cv2.imread(img_path)
            if img.shape[0] == 112 and img.shape[1] == 112:
                batch_img_list.append(img)
            else:
                print(img_path)
        yield batch_img_list, batch_name_list

def get_feature_batch(model, file_list, batch_size):
    """
    get feature batch
    :param file_list: file list name
    :param batch_size:
    :return:name list and feature list
    """
    name_list = []
    feature_list = []
    for batch_img_list, batch_name_list in tqdm(gen_inputs(file_list, batch_size)):
        batch_inputs = model.get_input_batch(batch_img_list)
        batch_features = model.get_feature_batch(batch_inputs)
        feature_list.extend(batch_features)
        b_n_l = []
        for name in batch_name_list:
            b_n_l.append(name)
        name_list.extend(b_n_l)
    return name_list, feature_list










class Dataset(data.Dataset):

    def __init__(self, data_list_file, phase='train'):
        self.phase = phase
        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [img[:-1] for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = trans.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])

        #normalize = T.Normalize([127.5,127.5,127.5], [127.5,127.5,127.5])

        if self.phase == 'train':
            self.transforms = trans.Compose([
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                normalize
            ])
        else:
            self.transforms = trans.Compose([
                trans.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[1]
        # image = Image.open(img_path)
        # if len(image.split())!=3:
        #     #print(img_path,len(image.split()))
        #     image = image.convert(mode='RGB')
        img = cv2.imread(img_path)
        img = img[..., ::-1]
        image = Image.fromarray(img)
        data = self.transforms(image)
        label = np.int32(splits[2])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)

# if __name__=="__main__":
#     load_bin("/home/njfh/hdd_data/mzh/datasets/faces_emore/lfw.bin","./data/testdb",transform)