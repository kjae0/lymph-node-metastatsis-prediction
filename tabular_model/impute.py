import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer

def check_validity(df, origin):
    nan_cnt = sum(df.isnull().sum())
    if nan_cnt:
        raise ValueError('Nan values still exist!')
    if set(df.columns) != set(origin.columns):
        raise ValueError('Some columns are shifted!')

def knn_impute(train, test):
    float_columns = ['나이', '진단명', '암의 위치', '암의 개수',
       '암의 장경', 'NG', 'HG', 'HG_score_1', 'HG_score_2', 'HG_score_3',
       'DCIS_or_LCIS_여부', 'DCIS_or_LCIS_type', 'T_category', 'ER',
       'ER_Allred_score', 'PR', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2',
       'HER2_IHC', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation']
    str_columns = ['img_path', 'mask_path', '수술연월일', 'ID']

    knn_imputer = KNNImputer()
    knn_imputer.fit(train[float_columns])

    train_knn_imputed = pd.DataFrame(knn_imputer.transform(train[float_columns]), columns=float_columns)
    test_knn_imputed = pd.DataFrame(knn_imputer.transform(test[float_columns]), columns=float_columns)

    for col in str_columns:
        train_knn_imputed[col] = train[col]
    train_knn_imputed['N_category'] = train['N_category']

    for col in str_columns:
        if col == 'mask_path':
            continue
        test_knn_imputed[col] = test[col]
    
    check_validity(train_knn_imputed, train)
    check_validity(test_knn_imputed, test)
    
    return train_knn_imputed, test_knn_imputed, knn_imputer
