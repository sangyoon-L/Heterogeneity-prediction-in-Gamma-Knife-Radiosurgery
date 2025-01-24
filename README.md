# Predicting Dose Heterogeneity in Gamma Knife Radiosurgery

This is a repository for the project "Predicting Dose Heterogeneity in Gamma Knife Radiosurgery Using a Diffusion Model With 2.5D Attention"

![Heterogeneity-prediction-in-Gamma-Knife-Radiosurgery](/Figures/github_fig1.PNG)
![Heterogeneity-prediction-in-Gamma-Knife-Radiosurgery](/Figures/github_fig2.PNG)

## Requirements
* python 3.6
* pytorch 1.10
* pydicom
* albumentations
* tensorboardX
* SimpleITK

## Convert 3D into 2D for training
2D NPY format is required for training and evaluation. If Nifty, please convert the dataset through the available preprocessing:
python NII_to_npy.py

### Model training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --master_port=9999 train.py --gpu 0,1,2,3,4,5,6,7 --bs 16 --T 1000 --epoch 1600 --experiment HeteroDiff --date=20240831
```
### Model evaluation
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=29500 pred.py --gpu 0 --bs 1 --model_path ./HeteroDiff/trained_models/HeteroDiff_20240831/model_epochXXXX.pth --save_path ./HeteroDiff/result/HeteroDiff_20240831/ --ddim XXX --epoch XXX
```
