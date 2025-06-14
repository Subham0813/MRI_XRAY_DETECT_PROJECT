import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from scripts.model import build_cnn_model, build_mri_model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from scripts.utils import load_model_and_encoder, detect_xray


import gdown
import zipfile

def download_and_extract_models():
    zip_url = "https://drive.google.com/uc?id=1TZGWa8tkztjI_d_ouW3MWzfxPC7QDtjc"
    zip_path = "models.zip"

    # Downloads the zip file if not already present
    if not os.path.exists("models"):
        print("[INFO] Downloading models.zip from Google Drive...")
        gdown.download(zip_url, zip_path, quiet=False)

        print("[INFO] Extracting models.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

        print("[INFO] Models extracted to ./models/")

# === Calling this before loading models ===
download_and_extract_models()


# === Flask App Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Model Paths ===
X_RAY_MODEL_PATH = "models/CNN_Covid19_Xray_Version.h5"
X_RAY_ENCODER_PATH = "models/Lebel_encoder.pkl"
MRI_MODEL_PATH = "models/model.h5"

# === Class Labels ===
MRI_LABELS = ['giloma', 'meningioma', 'notumor', 'pituitary']

# === Load Models ===
xray_model, xray_encoder = load_model_and_encoder(X_RAY_MODEL_PATH, X_RAY_ENCODER_PATH)
mri_model = build_mri_model(num_classes=len(MRI_LABELS))  # Rebuild model
mri_model.load_weights(MRI_MODEL_PATH)  # Load only weights
xray_mri_check_model = load_model("models/xray_mri_classifier.h5")

def is_xray_or_mri(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = xray_mri_check_model.predict(img)[0]  
    class_idx = np.argmax(pred)
    class_labels = ['covid', 'lung_opacity', 'mri', 'normal_mri', 'normal_photos', 'normal_xray', 'viral_pneumonia']  
    class_name = class_labels[class_idx]
    confidence = np.max(pred)

    return class_name, confidence




# === Predict Functions ===
def predict_mri(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = mri_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if MRI_LABELS[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {MRI_LABELS[predicted_class_index]}", confidence_score


# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    file_path = None

    if request.method == 'POST':
        detection_type = request.form.get('detection_type')
        file = request.files['file']
        if file and detection_type:
            file_path_local = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path_local)

            # if detection_type == 'xray':
            #     if not is_xray_or_mri(file_path_local):
            #         result = "This doesn't appear to be a chest X-ray. Please upload a valid image."
            #         return render_template('index.html', result=result)
            #     result, confidence = detect_xray(file_path_local, xray_model, xray_encoder)
            # elif detection_type == 'mri':
            #     if is_xray_or_mri(file_path_local): 
            #         result = "This appears to be an X-ray. Please upload an MRI scan."
            #         return render_template('index.html', result=result)
            #     result, confidence = predict_mri(file_path_local)

            predicted_class, confidence = is_xray_or_mri(file_path_local)
            print(f"[DEBUG] Predicted class: {predicted_class}, confidence: {confidence}")

            if detection_type == 'xray':
                if predicted_class == 'normal_photos' or predicted_class == 'mri' or  predicted_class == 'normal_mri' or confidence < 0.7:
                    result = "This doesn't appear to be a valid chest X-ray. Please upload a proper X-ray image."
                    return render_template('index.html', result=result)
                result, confidence = detect_xray(file_path_local, xray_model, xray_encoder)

            elif detection_type == 'mri':
                if predicted_class == 'normal_photos' or predicted_class == 'covid' or  predicted_class == 'normal_xray' or  predicted_class == 'viral_pneumonia' or  predicted_class == 'lung_opacity'or confidence < 0.7:
                    result = "This doesn't appear to be a valid MRI scan. Please upload a proper MRI image."
                    return render_template('index.html', result=result)
                result, confidence = predict_mri(file_path_local)

            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f"/uploads/{file.filename}", detection_type=detection_type)

    return render_template('index.html', result=result)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    app.run(debug=True)
