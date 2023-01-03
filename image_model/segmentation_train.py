import torch
import torch.nn as nn

from torch import optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import hflip, vflip, adjust_hue, adjust_brightness, adjust_contrast, adjust_saturation, rotate

import os
import numpy as np
import pandas as pd
import custom_dataset
import utils
import models
import ssl

from tqdm.auto import tqdm

# 2가지 모델로 pipeline을 구성했음.
# 1) timm-efficientnet-b3 encoder based Unet, 2) inceptionresnetv2 encoder based Unet
# 각 모델로 학습을 시키기 위해서 args에 'encoder'를 바꾸면 됨.

# 1) timm-efficientnet-b3
# encoder -> timm-efficientnet-b3
# 2) inceptionresnetv2
# encoder -> inceptionresnetv2



# 모델 다운로드 과정에서 발생하는 오류 해결
ssl._create_default_https_context = ssl._create_unverified_context

# hyperparameters
args = {
    'batch_size' : 16,
    'learning_rate' : 1e-3,
    'target_value' : 0,
    'cnt_threshold' : 10000,
    'binary_threshold' : 240,
    'patch_size' : 320,
    'stride' : 160,
    'random_seed' : 10,
    'test_size' : 0.15,
    'out_dim' : 2,
    'n_epochs' : 10,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'test' : None,
    'encoder' : 'timm-efficientnet-b3',
    'save' : True
}

# random seed 세팅
utils.set_seeds(seed=args['random_seed'])


# 경로 설정
root_dir = "./submission_codes"
csv_dir = os.path.join(root_dir, "data", "clinical_data")
img_dir = os.path.join(root_dir, "data", "image_data")
model_dir = os.path.join(root_dir, "data", "saved_models")

file_names = os.listdir(os.path.join(img_dir, "train_cropped"))
file_names.sort()

# train/validation dataset에 들어갈 이미지 random choice
split_indices = np.random.choice(len(file_names), 
                                 int(len(file_names)*args['test_size']), 
                                 replace=False)
split_indices.sort()
train_file_names = []
val_file_names = []

for i in range(len(file_names)):
    if i in split_indices:
        val_file_names.append(file_names[i])
    else:
        train_file_names.append(file_names[i])


# dataset 로딩
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
mask_transform = transforms.ToTensor()

train_dataset = custom_dataset.SegmentationDataset(
    csv_dir=os.path.join(csv_dir, "train.csv"),
    img_dir=img_dir,
    target_value=args['target_value'],
    img_transform=img_transform,
    mask_transform=mask_transform,
    mask_only=True,
    file_names=[[os.path.join("train_cropped", i) for i in train_file_names], 
                [os.path.join("mask_cropped", i) for i in train_file_names]],
    threshold=1,
    cnt_threshold=args['cnt_threshold'],
    test=args['test'],
    binary_threshold=args['binary_threshold'],
    image_size=args['patch_size'],
    stride=args['stride']
)

val_dataset = custom_dataset.SegmentationDataset(
    csv_dir=os.path.join(csv_dir, "train.csv"),
    img_dir=img_dir,
    target_value=args['target_value'],
    img_transform=img_transform,
    mask_transform=mask_transform,
    mask_only=True,
    file_names=[[os.path.join("train_cropped", i) for i in val_file_names], 
                [os.path.join("mask_cropped", i) for i in val_file_names]],
    threshold=1,
    cnt_threshold=args['cnt_threshold'],
    test=args['test'],
    binary_threshold=args['binary_threshold'],
    image_size=args['patch_size'],
    stride=args['stride']
)


# data augmentation
DA_methods = [(adjust_hue, 0.1), (adjust_hue, -0.1), (adjust_hue, 0.2), (adjust_hue, -0.2)]
train_dataset.data_augmentation(DA_methods)

DA_methods = [(hflip, None)]
train_dataset.data_augmentation(DA_methods, label_tranform=True, label_only=1)

# dataloader 생성
train_dataloader = DataLoader(train_dataset,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              drop_last=True)

val_dataloader = DataLoader(val_dataset,
                            batch_size=args['batch_size'],
                            shuffle=False,
                            drop_last=False)

# 모델 및 optimizer, loss function 정의
model = models.SegmentationUnetModel(args['encoder']).to(args['device'])
optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
criterion = utils.dice_loss

for epoch in range(args['n_epochs']):
    model.train()
    total_loss = 0
    n_iter = 0

    for batch_id, (x, y, y_patch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'{epoch+1} training...'):
        model.train()
        x = x.to(args['device'])
        y = y.to(args['device'])
        y_patch = y_patch.to(args['device'])
        preds = model(x)

        loss = criterion(preds, y_patch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss
        n_iter += 1

        if (batch_id+1)%(len(train_dataloader)//3)==0:
            with torch.no_grad():
                model.eval()
                union, intersection, wrong, miss = 0, 0, 0, 0
                for x, y, y_patch in val_dataloader:
                    x = x.to(args['device'])
                    y = y.to(args['device'])
                    y_patch = y_patch.to(args['device'])
                    pred = model(x) > 0.5
                    u, i, w, m = utils.segmentation_metric(pred, y_patch)
                    union += u
                    intersection += i
                    wrong += w
                    miss += m
            print(f'correct : {intersection/union}, wrong prediction : {wrong/union}, missed : {miss/union}')
            save_dir = os.path.join(model_dir, args['encoder'])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if args['save']:
                torch.save(model, os.path.join(save_dir, f"model_{epoch+1}_{(batch_id+1)//(len(train_dataloader)//3)}.pt"))
            print(f'train loss : {total_loss/n_iter}')
    