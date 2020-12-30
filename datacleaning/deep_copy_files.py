import csv
import os
import shutil

# This code is used to take all the .nii files from subfolders recursively and copy them in new destination folder
# and also make entry in csv file for further uses

# file path of our source folder
source_folder = 'C:\\Users\\nithe\\ADNI_DATASET\\ADNI1_Complete_1Yr_1.5T\\Test1'
# file path of our destination folder
destination_folder = 'C:\\Users\\nithe\\ADNI_DATASET\\ADNI1_Complete_1Yr_1.5T\\Test2'

# To store file name and label in csv file for further process
label = "AD"

with open('All_Data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "label"])

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            print(file)
            writer.writerow([file, label])
            path_file = os.path.join(root, file)
            shutil.copy2(path_file, destination_folder)
