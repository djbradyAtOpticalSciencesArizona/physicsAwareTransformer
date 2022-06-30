from PIL import Image, ImageFilter
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np

def fourCond(img_hr_left, img_hr_right, rand1 = None, rand2 =None):
    if rand1 is None:
        rand1 = random.random()
    if rand2 is None:
        rand2 = random.random()
    if rand1 < 0.5:
        if rand2 < 0.5:
            ## condition 1:
            img_lr_left = img_hr_left.convert('L').filter(ImageFilter.BoxBlur(0.5+random.random()))
            img_lr_right = img_hr_right
        else:
            img_lr_left = img_hr_left.convert('L')
            img_lr_right = img_hr_right.filter(ImageFilter.BoxBlur(0.5+random.random()))
    else:
        (width, height) = (img_hr_right.width // 2, img_hr_right.height // 2)
        if rand2 < 0.5:
            img_lr_left = img_hr_left.convert('L')
            img_lr_right = img_hr_right.resize((width, height))
            img_lr_right = img_lr_right.resize((img_hr_right.width, img_hr_right.height))
        else:
            img_lr_left = img_hr_left.convert('L')
            img_lr_right = img_hr_right.filter(ImageFilter.BoxBlur(0.5+random.random())).resize((width, height))
            img_lr_right = img_lr_right.resize((img_hr_right.width, img_hr_right.height))

    img_hr_left = np.array(img_hr_left,dtype=np.float32)
    img_lr_left = np.tile(np.expand_dims(np.array(img_lr_left,dtype=np.float32), axis=2), (1, 1, 3))
    img_lr_right = np.array(img_lr_right,dtype=np.float32)
    img_hr_right = np.array(img_hr_right,dtype=np.float32)
    
    return img_hr_left, img_hr_right, img_lr_left, img_lr_right

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')
        
        img_hr_left, img_hr_right, img_lr_left, img_lr_right = fourCond(img_hr_left, img_hr_right)

        h, w, c = img_lr_left.shape
        num = self.file_list[index].split('_')[-1]
        if os.path.exists(f'{self.dataset_dir}/xxs_{num}.npy'):
            xxs = np.load(f'{self.dataset_dir}/xxs_{num}.npy')
            yys = np.load(f'{self.dataset_dir}/yys_{num}.npy')
        else:
            xxs = np.zeros((h, w, w), dtype=np.int16)
            yys = np.zeros((h, w, w), dtype=np.int16)
            for i in range(h):
                for j in range(w):
                    xxs[i, j] = i*np.ones((w,), dtype=np.int16)
                    yys[i, j] = np.arange(w, dtype=np.int16)
            xxs, yys = xxs.reshape((h*w, w)), yys.reshape((h*w, w))
        Pos = (xxs, yys)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right), Pos
    
    def __len__(self):
        return len(self.file_list)
    
class TrainSetMultiLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetMultiLoader, self).__init__()
        self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png').convert('L')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')

        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)[:, :, np.newaxis]
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_lr_rights = [img_lr_right[...,0:1], img_lr_right[...,1:2], img_lr_right[...,2:3]]

        h, w, c = img_lr_left.shape
        num = self.file_list[index].split('_')[-1]
        Pos = []
        for k in range(len(img_lr_rights)):
            if os.path.exists(f'{self.dataset_dir}/xxs_{num}.npy'):
                xxs = np.load(f'{self.dataset_dir}/xxs_{num}.npy')
                yys = np.load(f'{self.dataset_dir}/yys_{num}.npy')
            else:
                xxs = np.zeros((h, w, w), dtype=np.int16)
                yys = np.zeros((h, w, w), dtype=np.int16)
                for i in range(h):
                    for j in range(w):
                        xxs[i, j] = i*np.ones((w,), dtype=np.int16)
                        yys[i, j] = np.arange(w, dtype=np.int16)
                xxs, yys = xxs.reshape((h*w, w)), yys.reshape((h*w, w))
            Pos.append((xxs, yys))
        img_lr_rights = [toTensor(img_lr_right) for img_lr_right in img_lr_rights]
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), img_lr_rights, Pos
    
    def __len__(self):
        return len(self.file_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(dataset_dir + '/hr')
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/lr_x' + str(2) + '/' + self.file_list[index] + '/lr0.png') #Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/lr_x' + str(2) + '/' + self.file_list[index] + '/lr1.png') #Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
#         img_lr_left  = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr0.png')
#         img_lr_right = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr1.png')
        img_hr_left, img_hr_right, img_lr_left, img_lr_right = fourCond(img_hr_left, img_hr_right, 0.9, 0.1)
        
        h, w, c = img_lr_left.shape
        jn = 80
        xxs = np.zeros((h, w, jn))#, dtype=np.uint16)
        yys = np.zeros((h, w, jn))#, dtype=np.uint16)
        for i in range(h):
            for j in range(w):
                xxs[i, j] = i*np.ones((jn,))#, dtype=np.uint16)
                if j < jn: #j > w - jn: #
                    a, b = 0, jn#w-jn, w #
                else:
                    a, b = j-jn, j #j, j+jn #
                yys[i, j] = np.arange(a, b)#, dtype=np.uint16)
        Pos = (xxs, yys)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right), Pos
    def __len__(self):
        return len(self.file_list)

class TestSetMultiLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetMultiLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(dataset_dir + '/hr')
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr0.png').convert('L')
        img_lr_right = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr1.png')
        
        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)[..., np.newaxis]
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_lr_rights = [img_lr_right[...,0:1], img_lr_right[...,1:2], img_lr_right[...,2:3]]

        h, w, c = img_lr_left.shape
        Pos = []
        for k in range(len(img_lr_rights)):
            jn = 80
            xxs = np.zeros((h, w, jn))#, dtype=np.uint16)
            yys = np.zeros((h, w, jn))#, dtype=np.uint16)
            for i in range(h):
                for j in range(w):
                    xxs[i, j] = i*np.ones((jn,))#, dtype=np.uint16)
                    if j < jn: #j > w - jn: #
                        a, b = 0, jn#w-jn, w #
                    else:
                        a, b = j-jn, j #j, j+jn #
                    yys[i, j] = np.arange(a, b)#, dtype=np.uint16)
            Pos.append((xxs, yys))
        img_lr_rights = [toTensor(img_lr_right) for img_lr_right in img_lr_rights]
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), img_lr_rights, Pos
    def __len__(self):
        return len(self.file_list)
    
class TestDriggersLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestDriggersLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(dataset_dir + '/hr')
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        img_lr_left  =  Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        img_lr_right =  Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        
        img_hr_left  = np.repeat(np.array(img_hr_left,  dtype=np.float32)[..., np.newaxis], 3, axis=2)
        #img_hr_left = (img_hr_left/img_hr_left.max())**(1/2.2)*255.0
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.repeat(np.array(img_lr_left,  dtype=np.float32)[..., np.newaxis], 3, axis=2)
        #img_lr_left = (img_lr_left/img_lr_left.max())**(1/2.2)*255.0
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
                
        h, w, c = img_lr_left.shape
        num = self.file_list[index].split('_')[-1]
        if os.path.exists(f'{self.dataset_dir}/xxs_{num}.npy'):
            xxs = np.load(f'{self.dataset_dir}/xxs_{num}.npy')
            yys = np.load(f'{self.dataset_dir}/yys_{num}.npy')
        else:
            print('POS NOT FOUND.')
            xxs = np.zeros((h, w, w), dtype=np.int16)
            yys = np.zeros((h, w, w), dtype=np.int16)
            for i in range(h):
                for j in range(w):
                    xxs[i, j] = i*np.ones((w,), dtype=np.int16)
                    yys[i, j] = np.arange(w, dtype=np.int16)
            xxs, yys = xxs.reshape((h*w, w)), yys.reshape((h*w, w))
        Pos =(xxs, yys)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right), Pos
    def __len__(self):
        return len(self.file_list)
    
# class TestRealLoader(Dataset):
#     def __init__(self, dataset_dir, scale_factor):
#         super().__init__()
#     def __getitem__(self, index):
#         hr_image_left  = np.load('../data/test/peacock/nano1_patch.npy')[367:734,640:960]#[180:, 160:]
#         hr_image_right = np.load('../data/test/peacock/71_patch.npy')[225:, 400:600]
#         lr_image_left  = np.load('../data/test/peacock/nano1_patch.npy')[367:734,640:960]#[180:, 160:]
#         lr_image_right = np.load('../data/test/peacock/71_patch.npy')[225:, 400:600]
        
#         xxs = np.load('../data/test/peacock/xxs.npy')#[180:, 160:]
#         yys = np.load('../data/test/peacock/yys.npy')#[180:, 160:]
        
#         print(hr_image_left.shape, hr_image_right.shape)
#         Pos = (xxs, yys)
        
#         hr_image_left  = torch.from_numpy(hr_image_left.transpose((2, 0, 1))).float()
#         hr_image_right = torch.from_numpy(hr_image_right.transpose((2, 0, 1))).float()
#         lr_image_left  = torch.from_numpy(lr_image_left.transpose((2, 0, 1))).float()
#         lr_image_right = torch.from_numpy(lr_image_right.transpose((2, 0, 1))).float()
#         return hr_image_left, hr_image_right, lr_image_left, lr_image_right, Pos
#     def __len__(self):
#         return 1

class TestRealLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super().__init__()
    def __getitem__(self, index):
        hr_image_left  = Image.open('../data/test/PittsStereo-RGBNIR/211357_001468_NIRResize.png')
        hr_image_right = Image.open('../data/test/PittsStereo-RGBNIR/211357_001468_RGBResize.png')
        lr_image_left  = Image.open('../data/test/PittsStereo-RGBNIR/211357_001468_NIRResize.png')
        lr_image_right = Image.open('../data/test/PittsStereo-RGBNIR/211357_001468_RGBResize.png')
        
        hr_image_left = np.tile(np.expand_dims(np.array(hr_image_left,dtype=np.float32), axis=2), (1, 1, 3))#[:300,:300]
        hr_image_right = np.array(hr_image_right,dtype=np.float32)#[:300,:300]
        lr_image_left = np.tile(np.expand_dims(np.array(lr_image_left,dtype=np.float32), axis=2), (1, 1, 3))#[:300,:300]
        lr_image_right = np.array(lr_image_right,dtype=np.float32)#[:300,:300]
        
        print(hr_image_left.shape, hr_image_right.shape)
        
        
        h, w, c = lr_image_left.shape
        xxs = np.zeros((h, w, 40))#, dtype=np.uint16)
        yys = np.zeros((h, w, 40))#, dtype=np.uint16)
        for i in range(h):
            for j in range(w):
                xxs[i, j] = i*np.ones((40,))#, dtype=np.uint16)
                if j < 20:
                    a, b = 0, 40
                elif j > w-20:
                    a, b = w-40, w
                else:
                    a, b = j-20, j+20
                yys[i, j] = np.arange(a, b)#, dtype=np.uint16)
        Pos = (xxs, yys)
        
        hr_image_left  = toTensor(hr_image_left)
        hr_image_right = toTensor(hr_image_right)
        lr_image_left  = toTensor(lr_image_left)
        lr_image_right = toTensor(lr_image_right)
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right, Pos

    def __len__(self):
        return 1
    
def augumentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        if random.random()<0.5:     # flip horizonly
            lr_image_left = lr_image_left[:, ::-1, :]
            lr_image_right = lr_image_right[:, ::-1, :]
            hr_image_left = hr_image_left[:, ::-1, :]
            hr_image_right = hr_image_right[:, ::-1, :]
        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]
        """"no rotation
        if random.random()<0.5:
            lr_image_left = lr_image_left.transpose(1, 0, 2)
            lr_image_right = lr_image_right.transpose(1, 0, 2)
            hr_image_left = hr_image_left.transpose(1, 0, 2)
            hr_image_right = hr_image_right.transpose(1, 0, 2)
        """
        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)
