import os
import numpy as np
import pandas as pd

from PIL import Image
from tqdm.auto import tqdm

# 주어진 이미지가 비슷한 조직이 1~4번 반복된 형태로 존재하기 때문에
# 2등분선, 4등분선을 그어 background에 해당하는 pixel의 개수를 통해 이를 구분함.

# 2등분선, 4등분선을 그었을 때 background에 해당하지 않는 pixel의 개수를 return해주는 함수.
def count_overlapped_image(image, threshold):
    img_size = image.size
    image = np.array(image)
    
    mid = image[:, img_size[0]//2]
    mid_cnt = sum(mid<threshold)
    
    quad = image[:, img_size[0]//4]
    quad_cnt = sum(quad<threshold)
    
    return mid_cnt, quad_cnt


# 데이터 로딩
root_dir = "./submission_codes"
csv_dir = os.path.join(root_dir, "data", "clinical_data")
img_dir = os.path.join(root_dir, "data", "image_data")
train = pd.read_csv(os.path.join(csv_dir, 'train.csv'))

# 이미지 파일들의 이름을 가져옴.
train_image_names = []
for i in range(len(train)):
    if train['mask_path'].iloc[i] != '-':
        train_image_names.append(train['mask_path'].iloc[i][-14:])

# 각 이미지들이 조직이 몇번 반복된 형태인지 구분.
# 1) 2등분선 기준 30 pixel 미만 존재할 경우,  
# 1-1) 4등분선 기준 30 pixel 이하 존재하면 4번 반복으로 간주
# 1-2) 4등분선 기준 30 pixel 초과 존재하면 2번 반복으로 간주
# 2) 2등분선 기준 30 pixel 이상 존재할 경우,
# 2-1) 가로 길이와 세로 길이의 비율이 1.3을 넘어갈 경우, 3번 반복으로 간주
# 2-2) 가로 길이와 세로 길이의 비율이 1.3 이하일 경우, 1번 반복으로 간주
cnt_label = []
for file_name in tqdm(train_image_names):
    image = Image.open(os.path.join(img_dir, "train_imgs", file_name)).convert('L')
    m, q = count_overlapped_image(image, 200)
    
    if m<30:
        if q>30:
            label = 2
        else:
            label = 4
    else:
        if image.size[0]>image.size[1]*1.3:
            label = 3
        else:
            label = 1
    
    cnt_label.append([file_name, label])
    
# annotation mask는 반복된 조직들 중 1개의 조직에만 존재하므로,
# annotation mask가 존재하는 조직을 골라 crop.
cropped_imgs = []
cropped_masks = []

for i in tqdm(range(len(cnt_label)), total=len(cnt_label)):
    image = Image.open(os.path.join(img_dir, "train_imgs", train_image_names[i])).convert('RGB')
    mask = Image.open(os.path.join(img_dir, "train_masks", train_image_names[i])).convert('L')
    
    img_ary = np.array(image)
    mask_ary = np.array(mask)
    img_arys = []
    mask_arys = []
    size = image.size
    for j in range(cnt_label[i][1]):
        img_arys.append(img_ary[:, j*size[0]//cnt_label[i][1]:(j+1)*size[0]//cnt_label[i][1], :])
        mask_arys.append(mask_ary[:, j*size[0]//cnt_label[i][1]:(j+1)*size[0]//cnt_label[i][1]])
    
    target_cnt = []
    for ary in mask_arys:
        target_cnt.append(sum(sum(ary==0)))
    
    idx = target_cnt.index(max(target_cnt))
    
    cropped_img = Image.fromarray(img_arys[idx])
    cropped_mask = Image.fromarray(mask_arys[idx])
    cropped_imgs.append(cropped_img)     
    cropped_masks.append(cropped_mask)
    
# cropped 이미지 저장
save_dir = os.path.join(img_dir, "train_cropped")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, img in enumerate(cropped_imgs):
    img.save(os.path.join(save_dir, train_image_names[i]))
    print(f'{os.path.join(save_dir, train_image_names[i])} saved.')
print(f'{len(cropped_imgs)} images saved.')
    
save_dir = os.path.join(img_dir, "mask_cropped")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, img in enumerate(cropped_masks):
    img.save(os.path.join(save_dir, train_image_names[i]))
    print(f'{os.path.join(save_dir, train_image_names[i])} saved.')
print(f'{len(cropped_masks)} images saved.')
    