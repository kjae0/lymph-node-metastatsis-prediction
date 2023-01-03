# Solution Overview 
전반적인 개요는 Overview.png에 설명되어 있습니다.

## Image data
Tumor segmentation을 진행한 후, tumor region을 포함하도록 image patch를 sampling

ImageNet pretrained CNN architecture로 별도의 training 없이 WSI로부터 feature들을 추출

추출된 feature들을 합하여 tabular predictor로 N category probability를 얻고, tabular data에 concatenate함.


## Tabular data
KNN imputation 진행 후, 약간의 preprocessing과 Min-Max scaling으로 전처리함.

image feature와 같이 tabular predictor를 이용하여 결과값을 냄.

Private score : 0.8895629982
