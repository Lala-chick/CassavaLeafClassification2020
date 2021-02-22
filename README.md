# CassavaLeafClassification2020

Kaggle의 Cassava Leaf Disease Classification Competition의 코드
이미지가 주어졌을 때, Cassava식물의 상태를 5가지로 분류하는 문제

1. "Cassava Bacterial Blight (CBB)"
2. "Cassava Brown Streak Disease (CBSD)"
3. "Cassava Green Mottle (CGM)"
4. "Cassava Mosaic Disease (CMD)"
5. "Healthy"

모델은 timm라이브러리에서 총 6가지를 학습하여 앙상블 기법을 사용
1. resnet50d
2. resnext 50d 32x4d
3. ig_resnext 101_32x8d
4. tf_efficientnet_b3_ns
5. tf_efficientnet_b4_ns
6. vit_base_patch_16_384
