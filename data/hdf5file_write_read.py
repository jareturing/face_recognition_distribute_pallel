import h5py
import os
import numpy as np
class HDF5DatasetWriter:
    def __init__(self, dims,total_class_num, outputPath, dataKey="dataset_images", bufSize=1000):
        # 如果输出文件路径存在，提示异常
        if os.path.exists(outputPath):
            # print("The supplied 'outputPath' already exists and cannot be overwritten. Manually delete the file before continuing", outputPath)
            raise ValueError("The supplied 'outputPath' already exists and cannot be overwritten. Manually delete the file before continuing", outputPath)

        # 构建两种数据，一种用来存储图像特征一种用来存储标签
        # self.db = h5py.File(outputPath, "w")
        self.db = h5py.File(outputPath, "w",libver='latest')
        self.data = self.db.create_dataset(dataKey, dims, dtype=np.uint8)
        self.labels = self.db.create_dataset("dataset_labels", (dims[0],), dtype=np.int64)
        self.dataset_label_count = self.db.create_dataset("dataset_class_total",(1,), data=total_class_num, dtype=np.int64)

        # 设置buffer大小，并初始化buffer
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0   # 用来进行计数

    
    def add(self, rows, labels):
        # self.buffer["data"].extend(rows)
        # self.buffer["labels"].extend(labels)
        self.buffer["data"].append(rows)
        self.buffer["labels"].append(labels)
        
        # 查看是否需要将缓冲区的数据添加到磁盘中
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # 将buffer中的内容写入磁盘之后重置buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # 存储类别标签
        dt = h5py.special_dtype(vlen=str)  # 表明存储的数据类型为字符串类型
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        # 将classLabels赋值给labelSet但二者不指向同一内存地址
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) > 0:  # 查看是否缓冲区中还有数据
            self.flush()

        self.db.close()
class HDF5DatasetGenerator:

    def __init__(self, dbPath, batchSize, preprocessors = None, aug = None, binarize=True, classes=2):
        # 保存参数列表
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        # hdf5数据集
        # self.db = h5py.File(dbPath,'r')
        self.db = h5py.File(dbPath,'r',libver='latest',swmr=True)
        self.numImages = self.db['labels'].shape[0]
    
    def generator(self, passes=np.inf):
        epochs = 0
        # 默认是无限循环遍历，因为np.inf是无穷
        while epochs < passes:
            # 遍历数据
            for i in np.arange(0, self.numImages, self.batchSize):
                # 从hdf5中提取数据集
                images = self.db['images'][i: i + self.batchSize]
                labels = self.db['labels'][i: i + self.batchSize]
                
                # 检查是否标签需要二值化处理
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                # 预处理
                if self.preprocessors is not None:
                    proImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        proImages.append(image)
                    images = np.array(proImages)

                # 查看是否存在数据增强，如果存在，应用数据增强
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images,
                        labels, batch_size = self.batchSize))
                # 返回
                yield (images, labels)
            epochs += 1
    def close(self):
        # 关闭db
        self.db.close()
