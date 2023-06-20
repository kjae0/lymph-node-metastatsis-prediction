연세대학교 의과대학/JLK/MTS 주최 유방암의 임파선 전이 예측 AI 경진대회 우승 코드입니다.

# Solution Overview 
<img width="2072" alt="overview" src="https://user-images.githubusercontent.com/96808351/210345227-a14bf45c-0a61-444c-a996-f65796688be9.png">

## Image data
Tumor segmentation을 진행한 후, tumor region을 포함하도록 image patch를 sampling

ImageNet pretrained CNN architecture로 별도의 training 없이 WSI로부터 feature들을 추출

추출된 feature들을 합하여 tabular predictor로 N category probability를 얻고, tabular data에 concatenate함.


## Tabular data
KNN imputation 진행 후, 약간의 preprocessing과 Min-Max scaling으로 전처리함.

image feature와 같이 tabular predictor를 이용하여 결과값을 냄.

Private score : 0.8895629982
