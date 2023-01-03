import torch
import torch.nn as nn

from torch import optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import os
import numpy as np
import pandas as pd
import custom_dataset
import utils
import models

from tqdm.auto import tqdm
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# train dataset과 test dataset 모두에서 annotation mask를 만들어야 함.
# train dataset일 경우, inference_dataset variable을 "train", 
# test dataset일 경우, inference_dataset variable을 "test"로 세팅하면 됨.
# 실험에 활용된 annotation mask는 computer resource 이슈로 inference는 100~300 정도로 끊어 생성했음.

inference_dataset = "test"

if inference_dataset == "train":
    img_dir_name = "train_imgs"
    slicing = 13
    save_dir_name = "train_generated_mask"
    csv_name = "train.csv"
elif inference_dataset == 'test':
    img_dir_name = "test_imgs"
    slicing = 12
    save_dir_name = "test_generated_mask"
    csv_name = "test.csv"
else:
    raise ValueError('Wrong dataset variable!')

args = {
    'batch_size' : 64,
    'learning_rate' : 1e-3,
    'target_value' : 0,
    'cnt_threshold' : -1,
    'binary_threshold' : 240,
    'patch_size' : 320,
    'stride' : 50,
    'random_seed' : 10,
    'test_size' : 0.15,
    'n_epochs' : 10,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'encoder' : 'timm-efficientnet-b3',
}

# random seed 세팅
utils.set_seeds(seed=args['random_seed'])

# 경로 설정
root_dir = "./submission_codes"
csv_dir = os.path.join(root_dir, "data", "clinical_data")
# train set과 test set에 대하여 모두 inference를 진행해야 함.
img_dir = os.path.join(root_dir, "data", "image_data", img_dir_name)
save_dir = os.path.join(root_dir, "data", "image_data", save_dir_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train = pd.read_csv(os.path.join(csv_dir, csv_name))
img_path = [i[slicing:] for i in train['img_path']]

# model 로딩
model1 = torch.load(os.path.join(root_dir, f"/data/saved_models/timm-efficientnet-b3/model_6_3.pt").to(args['device']))
model2 = torch.load(os.path.join(root_dir, f"/data/saved_models/inceptionresnetv2/model_6_2.pt").to(args['device']))
model1.eval()
model2.eval()


img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
mask_transform = transforms.ToTensor()

for img_idx, file_name in enumerate(img_path):
    dataset = custom_dataset.SegmentationInferenceDataset(
        csv_dir=os.path.join(csv_dir, csv_name),
        img_dir=img_dir,
        target_value=args['target_value'],
        img_transform=img_transform,
        file_name=file_name,
        binary_threshold=args['binary_threshold'],
        image_size=args['patch_size'],
        stride=args['stride']
        )
    dataloader = DataLoader(dataset,
                            batch_size=args['batch_size'],
                            shuffle=False,
                            drop_last=False)

    preds = []
    with torch.no_grad():
        for x in tqdm(dataloader, desc=f'{img_idx+1} inference...', total=len(dataloader)):
            x = x.to(args['device'])    
            pred1 = nn.Sigmoid()(model1(x))
            pred2 = nn.Sigmoid()(model2(x))
            
            pred = pred1 / 2 + pred2 / 2
            preds.append(pred.cpu())
    preds = torch.cat(preds)
    col = (dataset.image.shape[1]-args['patch_size'])//args['stride'] + 1
    row = (dataset.image.shape[2]-args['patch_size'])//args['stride'] + 1
    new_img = torch.zeros(dataset.image.shape[1:])
    cnt = torch.zeros(dataset.image.shape[1:])
    cnt += 1e-4
    idx = 0
    for r in range(0, dataset.image.shape[1]-args['patch_size'], args['stride']):
        for c in range(0, dataset.image.shape[2]-args['patch_size'], args['stride']):
            new_img[r:r+args['patch_size'], c:c+args['patch_size']] += preds[idx].squeeze(0)
            cnt[r:r+args['patch_size'], c:c+args['patch_size']] += 1
            idx += 1
    new_img /= cnt
    new_img = (new_img > 0.5)
    new_img = np.array(new_img)
    image = Image.fromarray(new_img)
    image.save(os.path.join(save_dir, file_name))
    print(os.path.join(save_dir, file_name))
