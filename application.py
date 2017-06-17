import os
import numpy as np 
from keras.models import load_model
import nibabel as nib
from skull_extract import dice_coef_loss

trained_models = sorted(os.listdir("logs"))
for i in range(len(trained_models)):
    print(str(i) + " - " + trained_models[i])

try:
    model_number = int(input('Model number: '))
except ValueError:
    print("Not a number")

model = load_model("logs/" + trained_models[model_number], custom_objects={'dice_coef_loss': dice_coef_loss})

mri = nib.load("/mnt/disk3/datasets_rm/data_set_skull/dl_skull_trab/mris/101_str_crop.nii.gz")
affine = mri.affine
mri = mri.get_data()

mri = np.rollaxis(mri, 2).reshape(mri.shape[2], 176, 256, 1)
prediction = model.predict(mri)
prediction = np.rollaxis(prediction, 0, 3).reshape(176, 256, mri.shape[0])

prediction_img = nib.Nifti1Image(prediction, affine)
nib.save(prediction_img, "prediction_img.nii")
generated_mask = nib.Nifti1Image(np.round(prediction), affine)
nib.save(generated_mask, "generated_mask.nii")
