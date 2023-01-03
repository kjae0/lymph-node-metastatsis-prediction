import os
import pandas as pd
import numpy as np
import torch
import random

from tqdm.auto import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from autogluon.tabular import TabularPredictor
import utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# hyperparameters
random_seed = 42
root_dir = "./submission_codes/"
data_dir = os.path.join(root_dir, "data", "clinical_data")

utils.set_seeds(seed=random_seed)

train_image_pred = pd.read_csv(os.path.join(data_dir, "train_image.csv"))
test_image_pred = pd.read_csv(os.path.join(data_dir, "test_image.csv"))

preds = []
probas = []

train = pd.read_csv(os.path.join(data_dir, 'knn_impute', 'train_mm_scaled.csv'))
test = pd.read_csv(os.path.join(data_dir, 'knn_impute', 'test_mm_scaled.csv'))
train['image'] = train_image_pred['1']
test['image'] = test_image_pred['1']

label = train['N_category']    

gluon_clf = TabularPredictor(label='N_category', 
                                path=f'classification_model', 
                                ).fit(train)
predictions = gluon_clf.predict(test)
    
# load & fill submission file
submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
submission['N_category'] = pd.Series(predictions)


# check validity of submission file
nan_cnt = sum(submission.isnull().sum())
if nan_cnt:
    assert ValueError(f'Nan value detected! : {nan_cnt}')
    
# save submission file
submission.to_csv(os.path.join(root_dir, "submission.csv"), index=False)
