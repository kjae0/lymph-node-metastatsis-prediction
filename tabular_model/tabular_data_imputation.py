import os
import pandas as pd
import numpy as np
import utils
import impute

utils.set_seeds(seed=42)

# data directory
root_dir = "C://Users/rlawo/Desktop/Dacon/lymphatic/submission_codes/"
data_dir = os.path.join(root_dir, "data")
save_dir = os.path.join(data_dir, "clinical_data")

# load data
train = pd.read_csv(os.path.join(data_dir, "clinical_data", "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "clinical_data", "test.csv"))

print(f'train.csv has {sum(train.isnull().sum())} Nan values.')
print(f'test.csv has {sum(test.isnull().sum())} Nan values.')

# impute methods > KNN impute

train_impute = train.copy()
test_impute = test.copy()
    
train_imputed, test_imputed, imputer = impute.knn_impute(train_impute, test_impute)

impute_save_dir = os.path.join(save_dir, "knn_impute")
if not os.path.exists(impute_save_dir):
    os.makedirs(impute_save_dir)

train_imputed.to_csv(os.path.join(impute_save_dir, "train_imputed.csv"), index=False)
test_imputed.to_csv(os.path.join(impute_save_dir, "test_imputed.csv"), index=False)
