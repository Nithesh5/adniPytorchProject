import pandas as pd
import os, shutil

# This code is used to split the subject group folders into AD/CN/MCI folders

whole_adni_df = pd.read_csv("ADNI1_Complete_1Yr_1.5T_9_22_2020.csv")  # ad=473, cn=702, mci=1109

groupby_subject = whole_adni_df.groupby('Subject').head(1)
all_ad = groupby_subject[groupby_subject['Group'] == 'AD']

allSubject = all_ad['Subject']

for subject in allSubject:
    os.chdir('C:\\Users\\nithe\\ADNI_DATASET\\ADNI1_Complete_1Yr_1.5T\\ADNI')
    shutil.copytree(subject, 'C:\\Users\\nithe\\ADNI_DATASET\\ADNI1_Complete_1Yr_1.5T\\Classified\\MCI\\' + subject)
