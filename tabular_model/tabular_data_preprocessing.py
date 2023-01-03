import os
import pandas as pd
import numpy as np
import utils

from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler


def date_encoding(csv):
    year = []
    year_month = []
    month = []
    for i in range(len(csv)):
        date = csv['수술연월일'].iloc[i]
        year.append(int(date[2:4]))
        year_month.append(int(date[2:4])*12 + int(date[5:7]))
        month.append(int(date[5:7]))
    
    csv['year'] = pd.Series(year)
    csv['year&month'] = pd.Series(year_month)
    csv['month'] = pd.Series(month)
    csv = csv.drop('수술연월일', axis=1)
    return csv

def scaling(csv, scaler, test=False):
    scaling_columns = ['나이', 'year', 'year&month','month', '진단명', 
                    '암의 위치', '암의 개수 cat', 'NG', 'HG', 
                    'HG_score_1', 'HG_score_2', 'HG_score_3', 'DCIS_or_LCIS_여부', 
                    'T_category', 'ER', 'PR', 'HER2', 'HER2_IHC', 'HER2_SISH', 
                    '암의 장경 cat', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent']  

    csv = date_encoding(csv)
    
    csv.drop('DCIS_or_LCIS_type', axis=1, inplace=True)
    csv.drop('HER2_SISH_ratio', axis=1, inplace=True)
    csv.drop('BRCA_mutation', axis=1, inplace=True)
    
    cnt_cat = csv['암의 개수'] == 2
    csv['암의 개수 cat'] = pd.Series([int(i) for i in cnt_cat])
    length_cat = csv['암의 장경'] >= 15
    csv['암의 장경 cat'] = pd.Series([int(i) for i in length_cat])

    csv.drop('암의 개수', axis=1, inplace=True)
    csv.drop('암의 장경', axis=1, inplace=True)

    if test==False:
        scaler.fit(csv[scaling_columns])
    scaled = pd.DataFrame(scaler.transform(csv[scaling_columns]), columns=scaling_columns)
    
    if test==False:
        scaled['N_category'] = csv['N_category']
        return scaled, scaler
    else:
        return scaled

def preprocessing(file_dir):
    # load data
    train = pd.read_csv(os.path.join(file_dir, "train_imputed.csv"))
    test = pd.read_csv(os.path.join(file_dir, "test_imputed.csv"))
    
    # preprocessing & scaling
    # scaling methods :  min-max scaling

    # min-max scaling
    mm_scaler = MinMaxScaler()
    train_mm_scaled, train_mm_scaler = scaling(train, mm_scaler, test=False)
    test_mm_scaled = scaling(test, train_mm_scaler, test=True)

    train_mm_scaled.to_csv(os.path.join(file_dir, "train_mm_scaled.csv"), index=False)
    test_mm_scaled.to_csv(os.path.join(file_dir, "test_mm_scaled.csv"), index=False)
       
utils.set_seeds(seed=42)

# data directory
root_dir = "./submission_codes"
data_dir = os.path.join(root_dir, "data", "clinical_data")

preprocessing(os.path.join(data_dir, 'knn_impute'))    
