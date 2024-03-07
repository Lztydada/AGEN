# AGEN
Code for AGEN.
# Dataset
The Aircraft100 dataset can download from [here](https://drive.google.com/file/d/12L3N-gJMp96ltGgmt92-O8htFqcLS57s/view?usp=sharing). For CUB200, *mini*ImageNet and CIFAR100, please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them, and the text data for the three datasets has been placed in the 'data' folder.
# Training Scripts
CUB200
'''
$python train.py -project agen -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.005 -alpha 0.8  -beta 0.2 -lr_new 0.000005 -decay 0.0005 -epochs_base 50 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 224 96 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping
'''
Aircraft100
'''
$python train.py -project agen -dataset aircraft100 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.005 -alpha 0.8  -beta 0.2 -lr_new 0.000005 -decay 0.0005 -epochs_base 90 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 224 96 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping
'''
*mini*ImageNet
'''
$python train.py -project agen -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -alpha 0.8  -beta 0.2 -lr_new 0.001 -decay 0.0005 -epochs_base 50 -schedule Cosine -gpu 0 -temperature 16 -moco_dim 32 -mlp -moco_t 0.07 -size_crops 84 50 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -num_crops 2 0 -constrained_cropping
'''
CIFAR100
'''
$python train.py -project agen -dataset cifar100 -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -alpha 0.8  -beta 0.2 -lr_new 0.001 -decay 0.0005 -epochs_base 50 -schedule Cosine -gpu 0 -temperature 16 -moco_dim 32 -mlp -moco_t 0.07 -size_crops 32 18 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -num_crops 2 0 -constrained_cropping
'''
