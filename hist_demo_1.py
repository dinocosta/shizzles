import numpy as np
import os
import nibabel as nib
from nibabel.testing import data_path
from hist_eq import histogram_equalization

import nibabel as nib
import numpy as np
import os
from hist_eq import histogram_equalization

x_train_path = '/home/dl_skull/ex2.nii.gz'
x_train = nib.load(x_train_path)

train_images = np.rollaxis(x_train.get_data(), 2)

# Apply histogram equalization.

for i in range(len(train_images)):
    image                 = train_images[i]
    image_histogram, bins = np.histogram(image, 256, normed=True)
    cdf = image_histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]
    image_equalized = np.interp(image, bins[:-1], cdf)
    train_images[i] = image_equalized

# This code normalizes brightness values between 0 and 255

# for i in range(len(train_images)):
#     tmp = np.floor(train_images[i])
#     tmp /= np.max(tmp)
#     tmp *= 255
#     tmp = np.array(tmp, dtype=np.int16)
#     train_images[i] = tmp

# for i in range(len(train_images)):
#     tmp = train_images[i] * 255.0
#     new_img, h, new_h, sk = histogram_equalization(tmp)
#     train_images[i] = new_img

# To get the image back to a state where it can be written to disk we roll the axis again.

train_images_to_disk = np.rollaxis(train_images, 0, 3)
img = nib.Nifti1Image(train_images_to_disk, np.eye(4))
img.to_filename(os.path.join("","equalized.nii.gz"))

