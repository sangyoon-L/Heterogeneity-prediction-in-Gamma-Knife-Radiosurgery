from torch.utils.data import Dataset
import os
import torch
import numpy as np
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip, Resize)
import cv2
from PIL import Image

import re

def list_sort_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    
    def alphanum_key(s):
        return [ tryint(c) for c in re.split('([0-9]+)', s)]
    l.sort(key=alphanum_key)
    return l

class Dataset_PSDM_train(Dataset):
    def __init__(self, data_root='data'):
        self.file_dir_list = []
        self.file_name_list = []
        self.file_dir_list = self.file_dir_list + [data_root] * len(os.listdir(os.path.join(data_root, 'ct_dose')))
        self.file_name_list.extend(os.listdir(os.path.join(data_root, 'ct_dose')))
        
        
        self.file_dir_list = list_sort_nicely(self.file_dir_list)
        self.file_name_list = list_sort_nicely(self.file_name_list)
        
        self.transforms = Compose([
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.3, value=None,
                             mask_value=None, border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_NEAREST),
            HorizontalFlip(p=0.3), VerticalFlip(p=0.3)], p=0.8)
        self.transform_size= Resize(height=128,width=128)
        
        
        
        self.len = len(self.file_name_list)
        

    def __getitem__(self, idx):
        file_dir = self.file_dir_list[idx]
        file_name = self.file_name_list[idx]
        #print(file_name)
        ct_dose = np.load(os.path.join(file_dir, 'ct_dose', file_name))[:, :, np.newaxis]
        mr_dose = np.load(os.path.join(file_dir, 'mr_dose', file_name))[:, :, np.newaxis]
        mask = np.load(os.path.join(file_dir, 'mr_2.5D_20', file_name))[:,:]
        
        
        mask0 = mask[0, :, :, np.newaxis]
        mask1 = mask[1, :, :, np.newaxis]
        mask2 = mask[2, :, :, np.newaxis]
        mask3 = mask[3, :, :, np.newaxis]
        mask4 = mask[4, :, :, np.newaxis]
        mask5 = mask[5, :, :, np.newaxis]
        mask6 = mask[6, :, :, np.newaxis]
        mask7 = mask[7, :, :, np.newaxis]
        mask8 = mask[8, :, :, np.newaxis]
        mask9 = mask[9, :, :, np.newaxis]
        mask10 = mask[10, :, :, np.newaxis]
        mask11 = mask[11, :, :, np.newaxis]
        mask12 = mask[12, :, :, np.newaxis]
        
        
        data_mask = np.concatenate([mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10, mask11, mask12], axis=-1)
        data_mask = self.transform_size(image=data_mask)['image']
        
        ct_dose = self.transform_size(image=ct_dose)['image']
        mr_dose = self.transform_size(image=mr_dose)['image']
        data_all = np.concatenate([ct_dose, mr_dose], axis=-1)
        
        ct_dose = torch.from_numpy(data_all[:, :, 0:1]).permute(2, 0, 1)
        mr_dose = torch.from_numpy(data_all[:, :, 1:2]).permute(2, 0, 1)
        mask = torch.from_numpy(data_mask).permute(2, 0, 1)
        
        ct_dose = (ct_dose+1)*0.5*70
        mr_dose = (mr_dose+1)*0.5*70
        #0-70 range
        
        return ct_dose, mr_dose, mask

    def __len__(self):
        return self.len


class Dataset_PSDM_val(Dataset):
    def __init__(self, data_root='data'):
        self.file_dir_list = []
        self.file_name_list = []
        self.file_dir_list = self.file_dir_list + [data_root] * len(os.listdir(os.path.join(data_root, 'ct_dose')))
        self.file_name_list.extend(os.listdir(os.path.join(data_root, 'ct_dose')))
        
        self.file_dir_list = list_sort_nicely(self.file_dir_list)
        self.file_name_list = list_sort_nicely(self.file_name_list)
        
        
        self.transform_size= Resize(height=128,width=128)
        self.len = len(self.file_name_list)
        
    def __getitem__(self, idx):

        
        file_dir = self.file_dir_list[idx]
        file_name = self.file_name_list[idx]
        ct_dose = np.load(os.path.join(file_dir, 'ct_dose', file_name))[:, :, np.newaxis]
        mr_dose = np.load(os.path.join(file_dir, 'mr_dose', file_name))[:, :, np.newaxis]
        mask = np.load(os.path.join(file_dir, 'mr_2.5D_20', file_name))[:,:]

        mask0 = mask[0, :, :, np.newaxis]
        mask1 = mask[1, :, :, np.newaxis]
        mask2 = mask[2, :, :, np.newaxis]
        mask3 = mask[3, :, :, np.newaxis]
        mask4 = mask[4, :, :, np.newaxis]
        mask5 = mask[5, :, :, np.newaxis]
        mask6 = mask[6, :, :, np.newaxis]
        mask7 = mask[7, :, :, np.newaxis]
        mask8 = mask[8, :, :, np.newaxis]
        mask9 = mask[9, :, :, np.newaxis]
        mask10 = mask[10, :, :, np.newaxis]
        mask11 = mask[11, :, :, np.newaxis]
        mask12 = mask[12, :, :, np.newaxis]
        
        data_mask = np.concatenate([mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10, mask11, mask12], axis=-1)
        data_mask = self.transform_size(image=data_mask)['image']
       
        ct_dose = self.transform_size(image=ct_dose)['image']
        mr_dose = self.transform_size(image=mr_dose)['image']
        data_all = np.concatenate([ct_dose, mr_dose], axis=-1)
    

        ct_dose = torch.from_numpy(data_all[:, :, 0:1]).permute(2, 0, 1)
        mr_dose = torch.from_numpy(data_all[:, :, 1:2]).permute(2, 0, 1)
        mask = torch.from_numpy(data_mask).permute(2, 0, 1)
        
        ct_dose = (ct_dose+1)*0.5*70
        mr_dose = (mr_dose+1)*0.5*70
        #0-70 range
        
        return ct_dose, mr_dose, mask

    def __len__(self):
        return self.len