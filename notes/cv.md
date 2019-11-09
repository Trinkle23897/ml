# CV field

[TOC]

## Detection & Localization

classification problem

两个loss：一个分类，一个预测框的位置

### R-CNN

RoI

Problems

- Ad hoc training objectives
  - Fine-tune network with softmax classifier (log loss)
  - Train post-hoc linear SVMs (hinge loss)
  - Train post-hoc bounding-box regressions (least squares)
- Training is slow (84h), takes a lot of disk space
- Inference (detection) is slow
  - 47s / image with VGG16 [Simonyan & Zisserman. ICLR15]
  - Fixed by SPP-net [He et al. ECCV14]

### Fast R-CNN

RoI Pooling

### Faster R-CNN

Insert Region Proposal Network (RPN) to predict proposals from features

Jointly train with 4 losses:

1. RPN classify object / not object
2. RPN regress box coordinates
3. Final classification score (object classes)
4. Final box coordinates

### YOLO / SSD

Detection without Proposals

Within each grid cell:

- Regress from each of the B base boxes to a final box with 5 numbers: (dx, dy, dh, dw, confidence)
- Predict scores for each of C classes (including background as a class)



## Segmentation

### Semantic segmentation

分类pixel

对称网络结构，downsampleing和upsampleing合在一起：中间能够获取high-level的特征信息

### Instance Segmentation

多类别pixel分类

#### Mask R-CNN



## Tracking

