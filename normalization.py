import numpy as np
import os
import nibabel as nib
from nibabel.testing import data_path
from hist_eq import histogram_equalization

def normalize_image(image):
    """
    Normalize the brightness values of an image.

    Params:
        image - The image to be normalized.

    Returns:
        A copy of the image with its brightness values between 0 and 255.
    """
    img = np.floor(image)
    img /= np.max(img)
    img *= 255
    img = np.array(img, dtype=np.int16)
    return img
