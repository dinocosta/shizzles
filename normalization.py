import numpy as np
import os
import nibabel as nib
from nibabel.testing import data_path


def load_files():
    # List files
    files_list = sorted(os.listdir("/mnt/disk3/datasets_rm/data_set_skull"))
    for i in range(len(files_list)):
        files_list[i] = "/mnt/disk3/datasets_rm/data_set_skull/" + files_list[i]
    
    # Load
    loaded_files = list(map(nib.load, files_list))
    
    shapes_width = []
    shapes_height = []
    for lf in loaded_files:
        shapes_width.append(lf.shape[0])
        shapes_height.append(lf.shape[1])
        
    max_width = max(shapes_width)
    max_height = max(shapes_height)

    loaded_files = list(map(lambda x: x.get_data(), loaded_files))

    return (loaded_files, files_list, max_width, max_height)

def normalize_brightness(image):
    """
    Normalize the brightness values of an image.

    Params:
        image - The image to be normalized.

    Returns:
        A copy of the image with its brightness values between 0 and 255.
    """
    # Voxel intensity normalization
    img = np.floor(image)
    img /= np.max(img)
    img *= 255
    img = np.array(img, dtype=np.int16)
    return img

def normalize_dimension(mri, max_width, max_height):

    exam = np.rollaxis(mri, 2) #put frames at index 0
    new_exam = np.zeros((0, max_width, max_height), dtype=np.int16)
    for cut in exam:
        new_cut = np.zeros((0,max_height), dtype=np.int16)
        for line in cut:
            new_line = np.resize(line, (max_height,))
            new_cut = np.append(new_cut, [new_line], axis=0)
        for i in range(max_width - exam.shape[1]):
            new_cut = np.append(new_cut, [np.zeros((max_height,), dtype=np.int16)], axis=0)
        new_exam = np.append(new_exam, [new_cut], axis=0)

    new_exam = np.rollaxis(new_exam,0,3)
    
    return new_exam

def write_to_file(mri, file_name):
    img = nib.Nifti1Image(mri, np.eye(4))
    #Get file name without the file extension in order to add _norm to the end of the name
    new_file_name = file_name.split("/")[-1].split(".")[0]
    img.to_filename(os.path.join("",new_file_name + "_norm.nii.gz"))	

loaded_files, files_list, max_width, max_height = load_files()
for i in range(4):
    lf = loaded_files[i]
    mri = np.zeros((0, lf.shape[1], lf.shape[2]), dtype=np.int16)
    for image in lf:
        img = normalize_brightness(image)
        mri = np.append(mri, [img], axis=0)
    norm_mri = normalize_dimension(mri, max_width, max_height)
    write_to_file(norm_mri, files_list[i])
