import os
from Nii_utils import NiiDataRead
import numpy as np

original_dir = r'preprocessed_data'
save_dir = r'preprocessed_data/NPY'

name_list = ['mr_dose', 'ct_dose','ct','tgt']

for phase in ['train','validation', 'test']:
    for ID in os.listdir(os.path.join(original_dir, '{}-pats_preprocess'.format(phase))):
        print(ID)
    
        for name in name_list:
            print(name)
            img, _, _, _ = NiiDataRead(os.path.join(original_dir, '{}-pats_preprocess'.format(phase), ID, '{}.nii.gz'.format(name)))
            if name == 'mr_dose':
                img = np.clip(img, 0, 70)
                img = img / 70
            elif name == 'ct_dose':
                img = np.clip(img, 0, 70)
                img = img / 70

            for i in range(img.shape[0]):

                img_one = img[i,:, :]
                np.save(os.path.join(save_dir, phase, name, '{}_{}.npy'.format(ID, i)), img_one)