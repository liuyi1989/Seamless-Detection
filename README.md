# Seamless-Detection
![illustration](assert/illustration.jpg)
We make the study to unify SOD and COD in a task-agnostic framework via a contrastive distillation paradigm, inspired by their agreeable nature of binary segmentation.

# 💥 News 💥
- [13.2.2025] [Seamless-Detection](https://www.sciencedirect.com/science/article/abs/pii/S0957417425005342) has been accepted to ESWA 2025 !
- [16.12.2024] Paper is now available on arXiv !
- [04.9.2024] We have released the code and model checkpoints for Seamless-Detection !
# Quick Start

- Pretrained backbone:[MoCo-v2](https://github.com/facebookresearch/moco).

- Training
```python
python train_compare.py \
cornet_compare \
--gpus=0 \
--save \
--found \
--vals=ECSSD
```
- Test
```
python test.py \
cornet_compare \
--weight="./weight/cornet_compare/resnet/base/cornet_compare_base_24.pth" \
--gpus=0 \
--save \
--vals=ECSSD,DUTS-TE,DUT-OMRON,PASCAL-S
```
# Requirements
python 3.9

pytorch 1.11.0

tensorboardX 2.5

# Weights and Results
Baidu | 提取码:sldt

# Citation
If you find Seamless-Detection to be useful for your work, please consider citing our paper:
```
 @article{liu2025seamless,
   title={Seamless Detection: Unifying Salient Object Detection and Camouflaged Object Detection},
   author={Liu, Yi and Li, Chengxin and Dong, Xiaohui and Li, Lei and Zhang, Dingwen and Xu, Shoukun and Han, Jungong},
   journal={Expert Systems with Applications},
   pages={126912},
   year={2025},
   publisher={Elsevier}
 }
```
