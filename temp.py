import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# from tensorflow.keras.models import load_model

# # Load problematic h5 file (ignore compile)
# model = load_model("models/model.h5", compile=False)

# # Save as SavedModel format (safe for future use)
# model.save("models/mri_model_converted")

import os
import shutil
from glob import glob

base_path = "data"
output_path = os.path.join(base_path, "classifier_dataset")

# 1. X-ray folders (COVID, Lung_Opacity, Viral Pneumonia, Normal)
xray_classes = {
    "COVID": "covid",
    "Lung_Opacity": "lung_opacity",
    "Viral Pneumonia": "viral_pneumonia",
    "Normal": "normal_xray"  # renamed to avoid conflict
}

for folder_name, target_name in xray_classes.items():
    src_folder = os.path.join(base_path, "COVID-19_Radiography_Dataset", folder_name, "images")
    dest_folder = os.path.join(output_path, target_name)
    os.makedirs(dest_folder, exist_ok=True)

    for img in glob(os.path.join(src_folder, "*.png")):
        shutil.copy(img, dest_folder)

print("✅ X-ray categories processed.")


# 2. MRI images: split notumor to normal_mri, others to mri/
mri_base = os.path.join(base_path, "MRI-dataset", "Training")
mri_dest = os.path.join(output_path, "mri")
normal_mri_dest = os.path.join(output_path, "normal_mri")
os.makedirs(mri_dest, exist_ok=True)
os.makedirs(normal_mri_dest, exist_ok=True)

mri_classes = ["glioma", "meningioma", "pituitary"]
for cls in mri_classes:
    src = os.path.join(mri_base, cls)
    for img in glob(os.path.join(src, "*")):
        shutil.copy(img, mri_dest)

# Handle 'notumor' as 'normal_mri'
notumor_src = os.path.join(mri_base, "notumor")
for img in glob(os.path.join(notumor_src, "*")):
    shutil.copy(img, normal_mri_dest)

print("✅ MRI images and normal_mri split done.")


# 3. Real-life normal photos
normal_photos_src = os.path.join(base_path, "Normal_Photos")
normal_photos_dest = os.path.join(output_path, "normal_photos")
os.makedirs(normal_photos_dest, exist_ok=True)

for img in glob(os.path.join(normal_photos_src, "*.jpg")):
    shutil.copy(img, normal_photos_dest)

print("✅ Real-world normal photos moved.")
