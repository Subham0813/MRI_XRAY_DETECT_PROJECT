import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

def save_model_and_encoder(model, lebel_encoder, model_path, encoder_path):
    model.save(model_path)
    with open(encoder_path, 'wb') as f:
        pickle.dump(lebel_encoder, f)
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")


def load_model_and_encoder(model_path, encoder_path):
    model = load_model(model_path)
    with open(encoder_path, 'rb') as f:
        lebel_encoder = pickle.load(f)
    return model, lebel_encoder


def detect_xray(image_path, model, lebel_encoder, image_size=150):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (image_size, image_size))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)

    predictions = model.predict(image_input)
    predicted_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_index]

    predicted_label = lebel_encoder.inverse_transform([predicted_index])[0]

    # Show image with prediction
    plt.imshow(image_resized)
    plt.title(f"Predicted: {predicted_label}, Confidence: {confidence_score * 100:.2f}%")
    plt.axis('off')
    

    return predicted_label, confidence_score
