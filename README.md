# CassavaLeafClassification2020

Kaggle의 Cassava Leaf Disease Classification Competition의 코드
이미지가 주어졌을 때, Cassava식물의 상태를 5가지로 분류하는 문제

0. "Cassava Bacterial Blight (CBB)"
1. "Cassava Brown Streak Disease (CBSD)"
2. "Cassava Green Mottle (CGM)"
3. "Cassava Mosaic Disease (CMD)"
4. "Healthy"

# 모델
모델은 timm라이브러리에서 총 6가지를 학습

학습 시 5 Fold Cross Validation을 이용
1. resnet50d (ResNet)
2. resnext 50d 32x4d (ResNext)
3. ig_resnext 101_32x8d (ResNext)
4. tf_efficientnet_b3_ns (EfficientNet)
5. tf_efficientnet_b4_ns (EfficientNet)
6. vit_base_patch_16_384 (Vision Transformer)

# 앙상블 기법
6가지 모델들에 대해 앙상블을 적용해 보고 제출을 시도
public score는 EfficientNet b4 + ResNext101 + ViT의 조합이 가장 높았으나
private score는 EfficientNet b3 + ViT의 조합이 가장 높은 것을 확인 할 수 있었음

EfficientNet b4 + ResNext101 + ViT
public LB: 0.9041
private LB: 0.8986

EfficientNet b3 + ResNext50 + ViT
public LB: 0.9025
private LB: 0.9011

# Test Time Augmentation
성능 향상을 위해 앙상블 기법과 더불어 Test Time Augmentation(TTA)적용

최종 제출은 public LB가 가장 높았던 EfficientNet b4 + ResNext101 + ViT로 해서 bronze medal 달성

# Data
코드를 사용하기 위해서는 https://www.kaggle.com/c/cassava-leaf-disease-classification/data로부터 데이터를 받아 train images폴더와 test images 폴더를 CassavaLeafClassification2020/data 폴더안에 넣어주고, train.csv, sample_submission.csv파일을 CassavaLeafClassification2020폴더에 넣어주어야 함

# run_train.py Arguments
"--model", type=str, default="tf_efficientnet_b3_ns"  
"--train_bs", type=int, default=16  
"--valid_bs", type=int, default=32  
"--test_bs", type=int, default=32  
"--epoch", type=int, default=10  
"--fold_num", type=int, default=5  
"--lr", type=float, default=1e-4  
"--weight_decay", type=float, default=1e-6  
"--num_workers", type=int, default=4  
"--accum_iter", type=int, default=2  
"--verbose_step", type=int, default=1  
"--img_size", type=int, default=512  
"--seed", type=int, default=209  
"--T_0", type=int, default=10  
"--min_lr", type=float, default=1e-6  
"--vit_img", type=int, default=384  

# run_inference.py Arguments
"--efficientnet_b3", type=bool, default=True  
"--efficientnet_b4", type=bool, default=False  
"--resnet_50", type=bool, default=False  
"--resnext_50", type=bool, default=True  
"--resnext_101", type=bool, default=False  
"--ViT", type=bool, default=True  
"--test_bs", type=int, default=32  
"--tta", type=int, default=3  
"--num_workers", type=int, default=4  
"--img_size", type=int, default=512  
"--vit_img", type=int, default=384  
