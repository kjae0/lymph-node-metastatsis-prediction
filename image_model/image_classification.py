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


def main(train, test, label, n_splits, random_seed):
    # dataset preprocessing
    label = label['0']
    train['label'] = label


    # train & inference
    train_predictions = pd.DataFrame(np.zeros((train.shape[0], 2)), columns=[0, 1])
    test_predictions = pd.DataFrame(np.zeros((test.shape[0], 2)), columns=[0, 1])

    kfold = StratifiedKFold(n_splits=n_splits, 
                            random_state=random_seed, 
                            shuffle=True)
    kfold.get_n_splits(train, label)
    train_indices = []
    val_indices = []
    for train_ids, val_ids in kfold.split(train, label):
        train_indices.append(train_ids)
        val_indices.append(val_ids)

    for i in range(n_splits):
        train_ids = train_indices[i]
        val_ids = val_indices[i]
        x_train = train.iloc[train_ids]
        x_val = train.iloc[val_ids]
        gluon_clf = TabularPredictor(label='label', 
                                    path=f'image_train_{i}_fold',
                                    verbosity=0 
                                    ).fit(x_train, presets=['best_quality'])
        train_proba = gluon_clf.predict_proba(x_val)
        train_predictions.iloc[val_ids] += train_proba
        
    gluon_clf = TabularPredictor(label='label', 
                                path=f'image_train_full',
                                verbosity=0
                                ).fit(train, presets=['best_quality'])
    test_proba = gluon_clf.predict_proba(test)
    test_predictions += test_proba
    return train_predictions, test_predictions


# hyperparameters
random_seed = 42
n_splits = 5
root_dir = "./submission_codes"
data_dir = os.path.join(root_dir, "data", "image_data")
save_dir = os.path.join(root_dir, "data", "clinical_data")
ensemble_list = ['96_efficientnetb3',
                 '96_inceptionresnetv2',
                 '120_efficientnetb3',
                 '120_inceptionresnetv2']

utils.set_seeds(seed=random_seed)

train_ensemble_result = pd.DataFrame(np.zeros((1000, 2)), columns=[0, 1])
test_ensemble_result = pd.DataFrame(np.zeros((1000, 2)), columns=[0, 1])

for k in tqdm(ensemble_list, desc="training...", total=len(ensemble_list)):
    train = pd.read_csv(os.path.join(data_dir, k, 'train_df_sum.csv'))
    test = pd.read_csv(os.path.join(data_dir, k, 'test_df_sum.csv'))
    label = pd.read_csv(os.path.join(data_dir, k, 'train_label.csv'))
    train_prediction, test_prediction = main(train=train,
                                             test=test,
                                             label=label,
                                             n_splits=n_splits,
                                             random_seed=random_seed)
    train_ensemble_result += train_prediction
    test_prediction += test_prediction
train_ensemble_result /= len(ensemble_list)
test_ensemble_result /= len(ensemble_list)

# save result
train_ensemble_result.to_csv(os.path.join(save_dir, "train_image.csv"), index=False)
test_ensemble_result.to_csv(os.path.join(save_dir, "test_image.csv"), index=False)
    