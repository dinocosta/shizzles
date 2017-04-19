import os
import numpy as np 
from keras.models import load_model
import normalization as norm
import nibabel as nib

trained_models = os.listdir("logs")
trained_models.remove("log.txt")
for i in range(len(trained_models)):
    print(str(i) + " - " + trained_models[i])

try:
    model_number = int(input('Model number: '))
except ValueError:
    print("Not a number")

model = load_model("logs/" + trained_models[model_number])

mri = nib.load("/home/dl_skull/normalized_images/SW3944C_str_crop_norm.nii.gz")
affine = mri.affine
mri = mri.get_data()

mri = np.rollaxis(norm.normalize_image(mri), 2).reshape(mri.shape[2], 176, 256, 1)
prediction = model.predict(mri)
prediction_img = nib.Nifti1Image(prediction, affine)
nib.save(prediction_img, "teste.nii")
