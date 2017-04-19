import numpy as np
import nibabel as nib
import os

# List all files in the dataset folder and sort them in order to have the masks after the 
# mri's .nii.gz.
files_list = sorted(os.listdir("/mnt/disk3/datasets_rm/data_set_skull"))
for i in range(4):
	filepath = "/mnt/disk3/datasets_rm/data_set_skull/" + files_list[i]
	# As the files are sorted we know that if i is odd then we are refering to a 
	# masks file.
	if (i % 2 != 0):
		nib_file = nib.load(filepath)
		masks    = np.rollaxis(nib_file.get_data(), 2)
		results = np.array([])
		for j in range(len(masks)):
			if (np.max(masks[j]) != 0):
				results = np.append(results, 1)
			else:
				results = np.append(results, 0)
		np.savetxt('masks_results/' + str(i) + '.txt', results, fmt='%d')

# Save array to file using:
# np.savetxt(<filename>, <array>, fmt='%d')

# Read array from file using:
# array = np.loadtxt(<filename>, dtype=int)
