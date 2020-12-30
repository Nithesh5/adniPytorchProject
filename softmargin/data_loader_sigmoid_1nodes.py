from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
import torch
import os
from torchvision import transforms

compose = transforms.Compose([
  transforms.ToTensor(),
])


class ADNIDataloaderAllData(Dataset):

    def __init__(self, df, root_dir, transform):
        self.df = df
        self.root_dir = root_dir
        #self.transform = transform
        self.transform = None


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0]) #1
        image = nib.load(img_path)

        img_shape = image.shape
        target_img_shape = (256, 256, 166)
        if img_shape != target_img_shape:
            resampled_nii = resample_img(image, target_affine=np.eye(4) * 2, target_shape=target_img_shape,
                                         interpolation='nearest')
            resampled_img_data = resampled_nii.get_fdata()
        else:
            resampled_img_data = image.get_fdata()

        resampled_data_arr = np.asarray(resampled_img_data)

        #min_max_normalization
        resampled_data_arr -= np.min(resampled_data_arr)
        resampled_data_arr /= np.max(resampled_data_arr)

        if self.transform:
            resampled_data_arr = self.transform(resampled_data_arr)

        resampled_data_arr = np.reshape(resampled_data_arr, (1, 256, 256, 166)) # ignored bz 1 is added in transform

        y_label = -1.0 if self.df.iloc[index, 1] == 'AD' else 1.0 #bz using cross entropy #1

        y_label = torch.tensor(y_label, dtype=torch.float)

        return resampled_data_arr, y_label