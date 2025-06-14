import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(data_dir='/kaggle/input', image_size=150):
    imagePaths = []
    for dirname, _, filenames in os.walk(data_dir):
        if 'images' in dirname:
            for filename in filenames:
                if filename.endswith('png'):
                    imagePaths.append(os.path.join(dirname, filename))

    Data = []
    Target = []
    label_map = {'Viral Pneumonia': 'Pneumonia', 'Normal': 'Normal', 'COVID': 'Covid-19'}

    for imagePath in tqdm(imagePaths, desc="Processing images"):
        label = imagePath.split(os.path.sep)[-3]
        if label not in label_map:
            continue
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size)) / 255.0
        Data.append(image)
        Target.append(label_map[label])

    le = LabelEncoder()
    labels = le.fit_transform(Target)
    labels = to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(Data, labels, test_size=0.2, stratify=labels, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    trainX = np.array(x_train)
    valX = np.array(x_val)
    testX = np.array(x_test)
    trainY = np.array(y_train)
    valY = np.array(y_val)
    testY = np.array(y_test)

    print(f"Processed {len(Data)} images with corresponding labels.")
    print("Trainning data shape: ", trainX.shape )
    print("Validation data shape: ", valX.shape)
    print("Testing data shape: ", testX.shape)
    print("Trainning labels shape: ", trainY.shape)
    print("Validation labels shape: ", valY.shape)
    print("Testing labels shape: ", testY.shape)

    return trainX, valX, testX, trainY, valY, testY, le
