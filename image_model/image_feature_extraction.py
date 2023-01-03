import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim as optim
from torchvision.transforms import transforms

import os
import numpy as np
import pandas as pd
import custom_dataset
import utils
import models
from tqdm.auto import tqdm

# n_patch는 (96, 120)을 모두 활용했음. 아래 n_patches variable을 수정하면 됨.
# inference dataset을 train, test로 세팅하여 코드 실행.

inference_dataset = "test"
n_patches = 96

if inference_dataset == 'train':
    test_dataset = False
    csv_name = "train.csv"
    img_name = "train_imgs"
    mask_name = "train_generated_mask"
elif inference_dataset == 'test':
    test_dataset = True
    csv_name = "test.csv"
    img_name = "test_imgs"
    mask_name = "test_generated_mask"
else:
    raise ValueError('Wrong dataset variable!')


utils.set_seeds(seed=42)

args = {
    'target_value' : 0,
    'binary_threshold' : 240,
    'tumor_threshold' : 3000,
    'patch_size' : 300,
    'stride' : 150,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'test' : 10,
    'save' : True,
    'max_width' : 3500,
    'n_patches' : n_patches,
}

# 경로 설정
root_dir = "./submission_codes"
data_dir = os.path.join(root_dir, "data")
csv_dir = os.path.join(data_dir, "clinical_data")
img_dir = os.path.join(data_dir, "image_data")

dataset = custom_dataset.MILDataset(
    csv_dir = os.path.join(csv_dir, csv_name),
    img_dir = os.path.join(img_dir, img_name),
    mask_dir = os.path.join(img_dir, mask_name),
    target_value=args['target_value'],
    img_transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)]),
    mask_transform=transforms.ToTensor(),
    image_size=args['patch_size'],
    stride=args['stride'],
    threshold=1,
    tumor_threshold=args['tumor_threshold'],
    test=args['test'],
    binary_threshold=args['binary_threshold'],
    max_width=args['max_width'],
    n_patches=args['n_patches'],
    test_dataset=test_dataset
)

                 
def extract_features(resnet, dataset, args, image_features):  
    model = models.EfficientNetB3(resnet=resnet).to(args['device'])

    hidden_features_sum = []
        
    if inference_dataset == 'train':
        train_labels = dataset.labels
        with torch.no_grad():
            for x, y in tqdm(dataset, total=len(dataset)):
                x = x.to(args['device'])
                s = model(x)
                hidden_features_sum.append(s.cpu())
    else:
        with torch.no_grad():
            for x in tqdm(dataset, total=len(dataset)):
                x = x.to(args['device'])
                s = model(x)
                hidden_features_sum.append(s.cpu())
            
            
    hidden_features_sum = np.array(torch.cat(hidden_features_sum))
    print(hidden_features_sum.shape, resnet)

    if args['save']:
        df_sum = pd.DataFrame(hidden_features_sum)
        save_dir = os.path.join(img_dir, image_features)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df_sum.to_csv(os.path.join(save_dir, f"{inference_dataset}_df_sum.csv"), index=False)
    
        if inference_dataset == 'train':
            label_df = pd.DataFrame(train_labels)
            label_df.to_csv(os.path.join(save_dir, f"{inference_dataset}_label.csv"), index=False)


print(args['n_patches'])
resnet = True
if resnet:
    save_name = f"{args['n_patches']}_inceptionresnetv2"
else:
    save_name = f"{args['n_patches']}_efficientnetb3"
    
extract_features(resnet, dataset, args, save_name)

resnet = False
if resnet:
    save_name = f"{args['n_patches']}_inceptionresnetv2"
else:
    save_name = f"{args['n_patches']}_efficientnetb3"
    
extract_features(resnet, dataset, args, save_name)