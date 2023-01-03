import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
import cv2 as cv

from tqdm.auto import tqdm
from PIL import Image

class SegmentationInferenceDataset(Dataset):
    def __init__(self, csv_dir, img_dir, target_value,
                 img_transform=None, 
                 image_size=300, 
                 stride=150,
                 file_name=None,
                 threshold=230,
                 binary_threshold=False):
        
        self.target_value = target_value
        self.csv = pd.read_csv(csv_dir)
        
        self.file_name = file_name
        self.img_transform = img_transform
        self.patches = []
        self.image_size = image_size
        
        total = 0
        cnt = 0
        self.image = Image.open(os.path.join(img_dir, file_name)).convert('RGB')
        gray_img = np.array(Image.open(os.path.join(img_dir, file_name)).convert('L'))
        
        if binary_threshold=='ostu':
            _, otsu_threshold = cv.threshold(255-gray_img, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            otsu_threshold = 255 - gray_img > _
        elif binary_threshold:
            otsu_threshold = gray_img < binary_threshold
        else:
            otsu_threshold = 1
        
        if self.img_transform:
            self.image = self.img_transform(self.image)
            
        for i in range(3):
            self.image[i] = self.image[i] * otsu_threshold
        
        for row in range(0, self.image.shape[1]-image_size, stride):
            for col in range(0, self.image.shape[2]-image_size, stride):
                self.patches.append((row, col))
                total += 1
        print(f'{total} patches created.')
        
    def __getitem__(self, idx):
        r, c = self.patches[idx][0], self.patches[idx][1]
        patch = self.image[:, r:r+self.image_size, c:c+self.image_size]
        return patch
    
    def __len__(self):
        return len(self.patches)

class SegmentationDataset(Dataset):
    def __init__(self, csv_dir, img_dir, target_value,
                 img_transform=None, 
                 mask_transform=None, 
                 image_size=300, 
                 stride=150,
                 mask_only=False,
                 file_names=None,
                 threshold=230,
                 cnt_threshold=10,
                 test=None,
                 binary_threshold=False):
        
        self.target_value = target_value
        self.csv = pd.read_csv(csv_dir)
        
        if file_names:
            self.img_file_names = file_names[0]
            self.mask_file_names = file_names[1]
        else:        
            if mask_only:
                self.img_file_names = []
                self.mask_file_names = []
                for i in range(len(self.csv)):
                    if self.csv['mask_path'].iloc[i] != '-':
                        self.img_file_names.append(self.csv['img_path'].iloc[i])
                        self.mask_file_names.append(self.csv['mask_path'].iloc[i])
            else:
                self.img_file_names = list(self.csv['img_path'])
                self.mask_file_names = list(self.csv['mask_path'])
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        self.patches = []
        self.labels = []
        self.label_patches = []
     
        if test:
            self.img_file_names = self.img_file_names[:test]
            self.mask_file_names = self.mask_file_names[:test]
            
        total = 0
        cnt = 0
        for idx, (img, msk) in tqdm(enumerate(zip(self.img_file_names, self.mask_file_names)), total=len(self.img_file_names), desc='extracting patch...'):
            self.image = Image.open(os.path.join(img_dir, img)).convert('RGB')
            gray_img = np.array(Image.open(os.path.join(img_dir, img)).convert('L'))
            
            if binary_threshold=='ostu':
                _, otsu_threshold = cv.threshold(255-gray_img, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                otsu_threshold = 255 - gray_img > _
            elif binary_threshold:
                otsu_threshold = gray_img < binary_threshold
            else:
                otsu_threshold = 1
            
            if self.img_transform:
                self.image = self.img_transform(self.image)
                gray_img = self.img_transform(gray_img)
                
            self.mask = Image.open(os.path.join(img_dir, msk)).convert('L')
            if self.mask_transform:
                self.mask = self.mask_transform(self.mask)
                
            for i in range(3):
                self.image[i] = self.image[i] * otsu_threshold
            
            gray_img[0] = gray_img[0] * otsu_threshold
                
            for row in range(0, self.image.shape[1]-image_size, stride):
                for col in range(0, self.image.shape[2]-image_size, stride):
                    patch = self.image[:, row:row+image_size, col:col+image_size]
                    gray_scale = gray_img[:, row:row+image_size, col:col+image_size]
                    
                    total += 1
                    if binary_threshold and (gray_scale>threshold/255).sum()<=cnt_threshold:
                        cnt += 1
                        continue
                    elif not binary_threshold and (gray_scale<threshold/255).sum()<=cnt_threshold:
                        cnt += 1
                        continue
                    
                    self.patches.append(patch)
                    label_patch = self.mask[:, row:row+image_size, col:col+image_size]==0
                    label_patch = label_patch.float()
                    self.label_patches.append(label_patch)
                    
                    if (self.mask[:, row:row+image_size, col:col+image_size]==0).sum().item()>0:
                        self.labels.append(1)
                    else:
                        self.labels.append(0)
        print(f'{total} patches created, {cnt} patches were passed. (under {threshold})')
        
    def data_augmentation(self, methods, label_tranform=False, label_only=False):
        self.augmented_patches = []
        self.augemented_labels = []
        self.augmented_label_patches = []
        for method, factor in methods:
            for idx, patch in tqdm(enumerate(self.patches), total=len(self.patches)):
                if label_only:
                    if self.labels[idx] != label_only:
                        continue
                if factor:
                    self.augmented_patches.append(method(patch, factor))
                else:
                    self.augmented_patches.append(method(patch))
                if label_tranform:
                    self.augmented_label_patches.append(method(self.label_patches[idx]))
                else:
                    self.augmented_label_patches.append(self.label_patches[idx])
                self.augemented_labels.append(self.labels[idx])
        self.patches.extend(self.augmented_patches)
        self.labels.extend(self.augemented_labels)
        self.label_patches.extend(self.augmented_label_patches)
        
        print(f'{len(self.augmented_patches)} is added. total dataset size is {len(self.patches)}.')
        
    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx], self.label_patches[idx]
    
    def __len__(self):
        return len(self.patches)

class MILDataset(Dataset):
    def __init__(self, csv_dir, img_dir, mask_dir, target_value,
                 img_transform=None, 
                 mask_transform=None,
                 image_size=300, 
                 stride=150,
                 threshold=230,
                 cnt_threshold=10,
                 tumor_threshold=100,
                 test=None,
                 binary_threshold=False,
                 max_width=3000,
                 n_patches=100,
                 test_dataset=False):
        
        self.target_value = target_value
        self.csv = pd.read_csv(csv_dir)
        
        if test_dataset:
            self.img_file_names = [i[12:] for i in self.csv['img_path']]
        else:
            self.img_file_names = [i[13:] for i in self.csv['img_path']]
        
        self.test_dataset = test_dataset
        if test_dataset==False:
            self.labels = [i for i in self.csv['N_category']]
        else:
            self.labels = None
        
        self.image_size = image_size

        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        self.patches = []
             
        if test:
            self.img_file_names = self.img_file_names[:test]
            
        total = 0
        cnt = 0
        for idx, img in tqdm(enumerate(self.img_file_names), total=len(self.img_file_names), desc='extracting patch...'):
            self.image = Image.open(os.path.join(img_dir, img)).convert('RGB')
            gray_img = np.array(Image.open(os.path.join(img_dir, img)).convert('L'))
            mask = Image.open(os.path.join(mask_dir, img)).convert('L')
            
            if binary_threshold=='ostu':
                _, otsu_threshold = cv.threshold(255-gray_img, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                otsu_threshold = 255 - gray_img > _
            elif binary_threshold:
                otsu_threshold = gray_img < binary_threshold
            else:
                otsu_threshold = 1
            
            if self.img_transform:
                self.image = self.img_transform(self.image)
                gray_img = self.mask_transform(gray_img)
                
            if self.mask_transform:
                self.mask = self.mask_transform(mask)
                
            for i in range(3):
                self.image[i] = self.image[i] * otsu_threshold
            gray_img[0] = gray_img[0] * otsu_threshold
            t_pat = []
            nt_pat = []
            for row in range(0, self.image.shape[1]-image_size, stride):
                for col in range(0, min(self.image.shape[2]-image_size, max_width), stride):                    
                    if (self.mask[:, row:row+image_size, col:col+image_size]).sum()<tumor_threshold:
                        t_pat.append((self.image[:, row:row+image_size, col:col+image_size].unsqueeze(0), (gray_img[:, row:row+image_size, col:col+image_size]>threshold/255).sum(), (self.mask[:, row:row+image_size, col:col+image_size]).sum()))
                    else:
                        nt_pat.append((self.image[:, row:row+image_size, col:col+image_size].unsqueeze(0), (gray_img[:, row:row+image_size, col:col+image_size]>threshold/255).sum(), (self.mask[:, row:row+image_size, col:col+image_size]).sum()))
            
            t_pat.sort(key=lambda x: x[1])
            nt_pat.sort(key=lambda x: x[1])
            t_pat = [i[0] for i in t_pat]
            nt_pat = [i[0] for i in nt_pat]
                        
            if (len(t_pat)+len(nt_pat))<n_patches:
                patch = t_pat + nt_pat
                indices = [i for i in range(len(patch))]
                sampled_indices = np.random.choice(indices, n_patches-len(patch), replace=True)
                for p in sampled_indices:
                    patch.append(patch[p])
            elif len(t_pat)>n_patches:
                # indices = [i for i in range(len(t_pat))]
                # sampled_indices = np.random.choice(indices, n_patches, replace=False)
                patch = t_pat[:n_patches]
            else:
                # indices = [i for i in range(len(nt_pat))]
                # sampled_indices = np.random.choice(indices, n_patches-len(t_pat), replace=False)
                patch = t_pat + nt_pat[:n_patches-len(t_pat)]
            
            self.patches.append(torch.cat(patch))  

        print(f'{total} patches created, {cnt} patches were passed. (under {threshold})')
        
    def __getitem__(self, idx):
        if self.test_dataset:
            return self.patches[idx]
        else:
            return self.patches[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.patches)   
    