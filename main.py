import os
import tensorflow as tf
from scripts.data_loader import load_and_preprocess_data
from scripts.model import build_cnn_model
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.utils import save_model_and_encoder, load_model_and_encoder, detect_xray

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))



# === GPU Check ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Using GPU: {gpus}")
    except RuntimeError as e:
        print(f"[ERROR] GPU setup failed: {e}")
else:
    print("[INFO] No GPU found, using CPU.")

# === CONFIGURATION ===
MODE = "predict"  # Options: "train", "evaluate", "predict"
DATASET_DIR = "data/COVID-19_Radiography_Dataset"
MODEL_PATH = "models/CNN_Covid19_Xray_Version.h5"
ENCODER_PATH = "models/Lebel_encoder.pkl"
PREDICT_IMAGE_PATH = "data/COVID-19_Radiography_Dataset/Normal/images/Normal-10019.png"


if MODE == "train":
    # Load and preprocess data
    trainX, valX, testX, trainY, valY, testY, label_encoder = load_and_preprocess_data(DATASET_DIR)

    # Build and train model
    model = build_cnn_model()
    history = train_model(model, trainX, trainY, valX, valY, epochs=25, batch_size=40)

    # Save model and label encoder
    os.makedirs("models", exist_ok=True)
    save_model_and_encoder(model, label_encoder, MODEL_PATH, ENCODER_PATH)

elif MODE == "evaluate":
    # Load data and model
    _, _, testX, _, _, testY, label_encoder = load_and_preprocess_data(DATASET_DIR)
    model, label_encoder = load_model_and_encoder(MODEL_PATH, ENCODER_PATH)

    # Evaluate model
    evaluate_model(model, testX, testY, label_encoder)

elif MODE == "predict":
    # Load model and encoder
    model, label_encoder = load_model_and_encoder(MODEL_PATH, ENCODER_PATH)

    # Run prediction on a new image
    predicted_label, confidence = detect_xray(PREDICT_IMAGE_PATH, model, label_encoder)
    print(f"Predicted Label: {predicted_label}, Confidence: {confidence * 100:.2f}%")

else:
    print("Invalid MODE selected. Choose from: 'train', 'evaluate', or 'predict'.")
