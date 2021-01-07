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
        target_img_shape = (1, 256, 256, 166)
        target_img_affine = "[[3.22155614e-08  2.46488298e-04 - 1.20344027e+00  9.32926025e+01], \
         [-2.46255622e-04 - 9.42077751e-01 - 3.14872030e-04  1.56945999e+02], \
         [-9.41189972e-01  2.46487912e-04  4.11920069e-08  1.12556000e+02], \
         [0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]"

        resampled_img_data = image.get_fdata()

        resampled_data_arr = np.asarray(resampled_img_data)

        #print("New method")
        #print(resampled_data_arr.shape)
        req = np.expand_dims(resampled_data_arr, axis=0)

        #print(req.shape)

        req_shape = req.shape
        resampled_data_arr = np.reshape(resampled_data_arr, req_shape)
        #print("Niiiiiiiiiiiiiiiiiiiiiiiiiiii")
        #print(resampled_data_arr.shape)
#        if img_shape != target_img_shape:
        if resampled_data_arr != target_img_shape:
            print("inside")
            """
            resampled_nii = resample_img(image, target_affine=np.eye(4) * 2, target_shape=target_img_shape,
                                         interpolation='nearest')
            resampled_img_data = resampled_nii.get_fdata()
            """
            print(resampled_img_data.shape)
            transform = tio.Resample(4)


            resampled_img_data = transform(resampled_data_arr)  # images in fpg are now in MNI space
            #resampled_img_data = resampled_img_data.get_fdata()
            print("shape")
            print(resampled_img_data.shape)
            print("going out")
        else:
            print("img_path")
            print(img_path)
            resampled_img_data = image.get_fdata()

        resampled_data_arr = np.asarray(resampled_img_data)

        # min_max_normalization
        resampled_data_arr -= np.min(resampled_data_arr)
        resampled_data_arr /= np.max(resampled_data_arr)

        #if self.transform:
        #    resampled_data_arr = self.transform(resampled_data_arr)



        if self.transform:
            #flip = tio.RandomFlip(axes=('P',), flip_probability=1) #Please use one of: L, R, P, A, I, S, T, B
            #resampled_data_arr = flip(resampled_data_arr)
            resampled_data_arr = transform_flip(resampled_data_arr)

        y_label = 0.0 if self.df.iloc[index, 1] == 'AD' else 1.0  # bz using cross entropy #1

        # y_label = [1.0, 0.0] if (self.df.iloc[index, 1] == 'AD') else [0.0, 1.0]  # for other cross entropy #2

        y_label = torch.tensor(y_label, dtype=torch.float)

        return resampled_data_arr, y_label
