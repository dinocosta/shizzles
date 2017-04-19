"""
This script is responsible for analysing the mask images and creating a .txt file which
classifies if the given image has a mask (1) or not (0).

This .txt files will be compared against the MRI images to train a neuronal network in 
classifying if an image contains brain tissue or not.

NOTE:
    If you wish to open the created .txt file as an array you can use numpy:
    np.loadtxt(<filename>.txt, dtype=int)
"""

import numpy as np
import nibabel as nib
import os

# List all files in the dataset folder and sort them in order to have the masks after the 
# mri's .nii.gz.
files_list = sorted(os.listdir("/mnt/disk3/datasets_rm/data_set_skull"))
for i in range(len(files_list)):
    filepath = "/mnt/disk3/datasets_rm/data_set_skull/" + files_list[i]
    # As the files are sorted we know that if i is odd then we are refering to a 
    # masks file.
    if (i % 2 != 0):
        nib_file = nib.load(filepath)
        masks    = np.rollaxis(nib_file.get_data(), 2)
        results  = np.array([])
        aux      = filepath.split('/')
        r_file   = 'mask_classes/class_' + aux[len(aux) - 1].replace('_bet_mask', '') + '.txt'

        for j in range(len(masks)):
            if (np.max(masks[j]) != 0):
                results = np.append(results, 1)
            else:
               results = np.append(results, 0)
        np.savetxt(r_file, results, fmt='%d')

# Save array to file using:
# np.savetxt(<filename>, <array>, fmt='%d')

# Read array from file using:
# array = np.loadtxt(<filename>, dtype=int)
