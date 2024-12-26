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
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=9999 train.py --gpu 0,1 --bs 16 --T 1000 --epoch 1600
```
### Model evaluation
```
CUDA_VISIBLE_DEVICES=0 python pred.py --gpu 0 --bs 64 --model_path trained_models/T1000_bs32_epoch1600/model_best_mae.pth --TTA 1 --T 1000 --ddim 8
```
