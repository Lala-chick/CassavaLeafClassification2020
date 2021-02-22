# CassavaLeafClassification2020

Kaggle의 Cassava Leaf Disease Classification Competition의 코드
이미지가 주어졌을 때, Cassava식물의 상태를 5가지로 분류하는 문제

1. "Cassava Bacterial Blight (CBB)"
2. "Cassava Brown Streak Disease (CBSD)"
3. "Cassava Green Mottle (CGM)"
4. "Cassava Mosaic Disease (CMD)"
5. "Healthy"

모델은 timm라이브러리에서 총 6가지를 학습하여 앙상블 기법을 사용함
학습 시 5 Fold Cross Validation을 이용
1. resnet50d (ResNet)
2. resnext 50d 32x4d (ResNext)
3. ig_resnext 101_32x8d (ResNext)
4. tf_efficientnet_b3_ns (EfficientNet)
5. tf_efficientnet_b4_ns (EfficientNet)
6. vit_base_patch_16_384 (Vision Transformer)

public score는 EfficientNet b4 + ResNext101 + ViT의 조합이 가장 높았으나
private score는 EfficientNet b3 + ViT의 조합이 가장 높은 것을 확인 할 수 있었음


성능 향상을 위해 앙상블 기법과 더불어 Test Time Augmentation(TTA)적용
