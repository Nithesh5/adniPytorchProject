from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
import torch
import os
from torchvision import transforms
import torchio as tio

compose = transforms.Compose([
    transforms.ToTensor(),
])

#https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomElasticDeformation

"""
transform = tio.RandomAffine(
    scales=(0.9, 1.2),
    degrees=10,
    isotropic=True,
    image_interpolation='nearest',
)

transform = tio.RandomElasticDeformation(
    num_control_points=(7, 7, 7),  # or just 7
    locked_borders=2,
)
"""

transforms_dict = {
    tio.RandomAffine(): 0.75,
    tio.RandomElasticDeformation(): 0.25,
}  # Using 3 and 1 as probabilities would have the same effect

transform_flip = tio.OneOf(transforms_dict)

class ADNIDataloaderAllData(Dataset):

    def __init__(self, df, root_dir, transform):
        self.df = df
        self.root_dir = root_dir
        # self.transform = transform
        self.transform = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0])  # 1
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

        # min_max_normalization
        resampled_data_arr -= np.min(resampled_data_arr)
        resampled_data_arr /= np.max(resampled_data_arr)

        resampled_data_arr = np.reshape(resampled_data_arr, (1, 256, 256, 166))  # ignored bz 1 is added in transform

        if self.transform:
            resampled_data_arr = transform_flip(resampled_data_arr)

        """
        import matplotlib.pyplot as plt
        #testing
        print("after")
        img = resampled_data_arr
        img = np.reshape(img, (256, 256, 166))
        # img=img.get_fdata()
        slice1 = (img[60, :, :])
        slice2 = (img[:, 80, :])
        slice3 = (img[:, :, 60])
        print(img.shape)
        plt.imshow(slice3, cmap='gray')
        plt.show()
        """

        y_label = 0.0 if self.df.iloc[index, 1] == 'AD' else 1.0  # bz using cross entropy #1
        y_label = torch.tensor(y_label, dtype=torch.float)

        return resampled_data_arr, y_label
