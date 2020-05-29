import numpy as np
import os,glob,sys
import h5py
import cv2
import math
import torch
import random
from PIL import Image
from tqdm import tqdm
from util import find_classes,make_dataset,IMG_EXTENSIONS
from hdf5file_write_read import HDF5DatasetWriter
from multiprocessing import Pool,Manager
def pool_thread_class(file_part_idx, split_part_num,samples_images_path_classidx,total_class_num, hdf5_path, LOCK):
    try:
        h5file_head, ext = os.path.splitext(hdf5_path)
        sample_idx_start = file_part_idx * split_part_num
        sample_idx_end = (file_part_idx + 1) * split_part_num
        hdf5_path_part = h5file_head + '_part' + str(file_part_idx) + ext
        print("create dataset file part", hdf5_path_part)
        file_Writer = HDF5DatasetWriter(dims=(split_part_num, 112, 112, 3),
                                       total_class_num=total_class_num, outputPath=hdf5_path_part)
        # print(samples_images_path_classidx)
        for sample in tqdm(samples_images_path_classidx[sample_idx_start:sample_idx_end]):
            image_path, classidx = sample
            image_array = cv2.imread(image_path)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            file_Writer.add(image_array,classidx)
        file_Writer.close()
        print("finish create ", hdf5_path_part)
    except Exception as e:
        print("hdf5 file create exception",e)
    finally:
        print ("hdf5 file create process exits!")
def pool_thread(file_part_idx, split_part_num,samples_images_path_classidx,total_class_num, hdf5_path, LOCK):
    h5file_head, ext = os.path.splitext(hdf5_path)
    sample_idx_start = file_part_idx * split_part_num
    sample_idx_end = (file_part_idx + 1) * split_part_num
    hdf5_path_part = h5file_head + '_part_' + str(file_part_idx) + ext
    print("create dataset file part", hdf5_path_part)
    file = h5py.File(hdf5_path_part, 'w')
    images_list = []
    labels_list = []
    for sample in tqdm(samples_images_path_classidx[sample_idx_start:sample_idx_end]):
        image_path, classidx = sample
        image_array = cv2.imread(image_path)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        images_list.append(image_array)
        labels_list.append(classidx)
        ######################################
    images = file.create_dataset("dataset_images", (sample_idx_end - sample_idx_start, 112, 112, 3), dtype=np.uint8)
    labels = file.create_dataset("dataset_labels", (sample_idx_end - sample_idx_start, 1), dtype=np.int64)
    file.create_dataset("dataset_class_total", (1,), data=total_class_num, dtype=np.int64)
    images[:] = np.array(images_list)
    labels[:] = np.array(labels_list)
    file.close()
    with LOCK:
        print("finish create ", hdf5_path_part)
def createh5pyfile_multiprocess(hdf5_path,image_floader,split_rank_num=8,data_order = 'tf',img_shape=(3,112,112),shuffle=True):
    img_channel,img_height,img_width=img_shape
    classes, class_to_idx =  find_classes(image_floader)
    total_class_num = len(classes)
    samples_images_path_classidx = make_dataset(image_floader, class_to_idx, extensions=IMG_EXTENSIONS, is_valid_file=None)
    total_sample_num=len(samples_images_path_classidx)
    if shuffle:
        random.shuffle(samples_images_path_classidx)
    split_part_num=math.ceil(total_sample_num/split_rank_num)
    while len(samples_images_path_classidx)<split_rank_num*split_part_num:
        print("To make data part equal,some sample replicate times")
        samples_images_path_classidx.append(samples_images_path_classidx[random.randint(0,total_sample_num)])
    worker = split_rank_num
    pool = Pool(worker)
    manage = Manager()
    dict = manage.dict()
    LOCK = manage.Lock()
    try:
        for i_cont in range(split_rank_num):
            # pool.apply_async(pool_thread, ( i_cont,split_part_num,samples_images_path_classidx,total_class_num,hdf5_path, LOCK))
            pool.apply_async(pool_thread_class,
                             (i_cont, split_part_num, samples_images_path_classidx, total_class_num, hdf5_path, LOCK))
        pool.close()
        pool.join()
    except Exception as e:
        print(e)
    finally:
        print ("success finish hdf5 file create!")
from torchvision import transforms as trans
import torch
train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file,transform=train_transform):
        super(dataset_h5, self).__init__()
        self.file = h5py.File(in_file, 'r')
        self.transform=transform
        self.n_images, self.image_h, self.image_w,image_c = self.file['dataset_images'].shape

    def __getitem__(self, index):
        image_array=np.uint8(self.file['dataset_images'][index, :])
        PIL_img=Image.fromarray(image_array)
        # print("read ",image_array)
        image=PIL_img.convert('RGB')
        # print("convert ",np.array(image))
        # print (image.size,image.shape)
        if self.transform is not None:
            sample = self.transform(image)
        # label = self.file['dataset_labels'][index,:]
        label = self.file['dataset_labels'][index]
        # print (sample,label)
        # print (sample.size)
        return sample,label
    def __len__(self):
        return self.n_images

def get(hdf5_path):
    datah5 = dataset_h5(hdf5_path)
    train_sampler = None
    loader = torch.utils.data.DataLoader(dataset=datah5, batch_size=32, shuffle=(train_sampler is None),
                                         pin_memory=True,
                                         num_workers=0)
    return loader
if __name__=='__main__':
    image_floader='/sdd_data/face_ms1s/datasets_miracle_v2'
    # image_floader = '/sdd_data/100WID'
    # hdf5_path=image_floader+'.hdf5'
    createh5pyfile_multiprocess(hdf5_path, image_floader)
    # #############################################################
    # hdf5_path=image_floader + '_part7.hdf5'   #new
    # # hdf5_path = image_floader + '_part7.hdf5' #old





